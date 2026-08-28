# Module 2b — LangGraph Architecture

*Sub-topic: LangGraph Architecture (30m)*

---

## Where we left off

In 02a you placed LangGraph in the ecosystem: LangChain supplies the components, LangGraph decides the control flow, LangSmith watches the result. LangGraph is standalone but composes with the other two.

Now we open the engine. By the end of this lesson you should be able to say, precisely, what happens when you call `graph.invoke()` — which is knowledge you'll need in Module 5 when you start building real graphs and they start misbehaving in interesting ways.

---

## The running example: Anita and Karthik's support bot

Anita is an Enterprise Solutions Architect at a SaaS company. Karthik is the senior AI engineer who owns their LangChain and LangGraph integrations. They have a problem, and it's the same problem you'd have.

> **Anita:** Karthik, we keep getting the same feedback about our support chatbot — it feels rigid. It doesn't adapt to different customer intents, and it never escalates when it should. Can LangGraph help us build something more dynamic?
>
> **Karthik:** Yes, and specifically because we stop chaining prompts and start defining *stateful interactions*. We get to control how the conversation flows, when tools get invoked, and when a human should step in.
>
> **Anita:** So we're not building a chain any more. We're building a state machine?

That is exactly the right instinct, and it's worth pausing on, because it's the correct mental model for everything that follows.

A **state machine** is three things: a set of states, a set of transitions between them, and a rule for picking which transition to take. LangGraph is a state machine where:

- the **nodes** are the work — an LLM call, a tool call, a database lookup, plain Python
- the **edges** are the transitions — and one of them can be *"ask the LLM which way to go"*
- the **state** is a single object carried along every transition, readable and writable by every node

If you've used XState in the JS world, this will feel immediately familiar. The novelty is only that a transition function is allowed to be a language model.

Anita and Karthik want four things out of their bot. Hold onto these — each maps onto a piece of the architecture:

1. Route differently depending on customer intent (billing vs. bug report vs. how-to)
2. Plug in tools for search, summarisation, and escalation — swappable, independently maintained
3. Pause and wait for a human agent when a case is sensitive, then carry on
4. Inspect how the conversation state evolved, to debug and to satisfy compliance

---

## Why a graph? (Five reasons, and the third one is the real one)

Before the layers, the shape. LangGraph could have picked a different central abstraction — a task DAG, an event bus, a plain agent loop. It picked the graph. Here's why that's not arbitrary.

### 1. Agentic workflows are already graph-shaped

Write down what an agent actually does:

```
Think → Act → Observe → Reflect → Decide → (Continue or Stop)
```

Or in more concrete terms:

```
LLM call → tool call → replan → new LLM call → log → final answer
```

Look at the shape of that. It isn't a line. It's a **loop with branches** that carries context forward each time round. "Decide → Continue" points *backwards*. "Decide → Stop" exits.

A pipeline cannot express this. `A | B | C` runs forwards, once. To get looping behaviour out of a pipeline you end up writing a `while` loop around it, hand-rolling a dict to carry state between iterations, and adding `if` statements to decide whether to go again. At which point — congratulations — you've written a graph engine, badly, inside your application code.

A graph expresses it natively, because "an edge that points backwards" is just an edge.

```
                  ┌──────────────────────────┐
                  │                          │ (retry / continue)
                  ▼                          │
 START ──▶ [ classify ] ──▶ [ retrieve ] ──▶ [ answer ] ──▶ END
                  │                                │
                  └────────▶ [ escalate ] ─────────┘
```

### 2. Control flow becomes visible instead of implied

Most LLM apps are a black box: prompt goes in, output comes out, and when it's wrong your only lever is rewording the prompt.

With a graph you explicitly declare:

- the order of operations
- the branching conditions
- the loops and retries
- the termination logic

That last one matters more than it sounds. "When does this stop?" is one of the hardest questions to answer about an autonomous agent, and in a graph it's a line of code you wrote.

> You can **see** the workflow instead of hoping the model figures it out.

Same behaviour, different epistemics: an unpredictable AI pipeline becomes an inspectable system.

### 3. Reliability and recoverability become possible at all

This is the reason that actually justifies the architecture, and the one the source undersells.

Because execution is a graph, the runtime always knows three facts at every moment:

- **where** it is (which node)
- **what** caused it to get there (which edge, from which node)
- **what the state looked like** when it arrived

Those three facts are what a checkpoint *is*. And once you can checkpoint, you get things that are otherwise impossible:

| Capability | Why the graph makes it possible |
|-----------|-------------------------------|
| **Resume from failure** | The API died at node 7 of 9. Restart from the node-7 checkpoint, not from the beginning. No re-running six LLM calls. |
| **Replay from any node** | Reload a past checkpoint and run forward again. This is "time travel" (Module 5). |
| **Human-in-the-loop** | Pausing is just *not scheduling the next node* and persisting state. Resuming is loading it back. |
| **Deterministic transitions despite a non-deterministic LLM** | The LLM's *output* varies. Which edges exist, and how state merges, do not. Randomness is confined to node bodies; the skeleton is fixed. |

That last row is the sharpest idea in the module. LangGraph doesn't make the LLM predictable — nothing does. It **contains** the unpredictability inside nodes, so the structure around it stays deterministic and auditable.

For anything running in production, this is the difference between a demo and a system.

### 4. It forces modular, reusable design

When a workflow is a graph:

- nodes are self-contained building blocks — one function, one job, testable in isolation
- graphs nest inside other graphs (**subgraphs** — Module 6)
- different teams own different nodes and ship independently

Karthik's "can we make it modular — plug in tools for search, summarisation, escalation?" is this property. Escalation is a subgraph. The team that owns escalation policy owns that subgraph and nothing else.

It's the microservices trade, at the workflow level: you can swap one node without rewriting everything, in exchange for having to think about the interface between them. Here the interface is the state schema.

### 5. It mirrors how reasoning actually works

Human problem-solving isn't a straight line. It's iterative, branching, and contextual — form a hypothesis, test it, observe, reflect, revise the plan, go again.

```
observe → reflect → plan → act → observe → …
```

A graph with cycles is a natural encoding of that. A linear pipeline is a lossy one. If you're modelling a reasoning process, use the structure that has the same shape as the thing you're modelling.

---

## The architecture: three layers

LangGraph is built as three layers. Learn the layers and you can place any feature in the course into one of them.

```
┌─────────────────────────────────────────────────────────────┐
│  A. GRAPH DEFINITION LAYER          — what you write        │
│     StateGraph API: nodes, edges, conditional edges          │
├─────────────────────────────────────────────────────────────┤
│  B. EXECUTION LAYER                 — what runs it          │
│     Pregel engine: super-steps, parallelism, checkpointing   │
├─────────────────────────────────────────────────────────────┤
│  C. STATE MANAGEMENT LAYER          — what it carries       │
│     Channels + reducers, short/long-term memory, time travel │
└─────────────────────────────────────────────────────────────┘
```

Note the division: **you only ever write in layer A.** Layers B and C are what you get for free — and understanding them is what stops layer A from surprising you.

---

## Layer A — Graph Definition

This is the `StateGraph` API. Three concepts:

| Concept | What it is | LangChain analogue |
|---------|-----------|--------------------|
| **Node** | A function: takes state, returns a state update | `RunnableLambda` — but it can be looped back to |
| **Edge** | A transition from one node to another | The `|` operator — but it can be conditional, and can point backwards |
| **State** | A typed object carried through the whole run | Nothing equivalent. This is the new idea. |

A node can be an LLM call, a tool invocation, or arbitrary custom logic. LangGraph does not care.

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# --- the state: the single object every node reads and writes ---
class SupportState(TypedDict):
    messages: Annotated[list, add_messages]   # reducer: append, don't overwrite
    intent: str                                # plain field: overwrite
    escalated: bool


# --- nodes: plain functions, state in, partial update out ---
def classify(state: SupportState):
    # a LangChain chain would live right here
    return {"intent": "billing"}

def answer(state: SupportState):
    return {"messages": [("ai", "Here's your invoice breakdown…")]}

def escalate(state: SupportState):
    return {"escalated": True}


# --- edges: the control flow ---
builder = StateGraph(SupportState)
builder.add_node("classify", classify)
builder.add_node("answer", answer)
builder.add_node("escalate", escalate)

builder.add_edge(START, "classify")          # static edge: always
builder.add_conditional_edges(               # conditional edge: decided at runtime
    "classify",
    lambda s: "escalate" if s["intent"] == "billing" else "answer",
)
builder.add_edge("answer", END)
builder.add_edge("escalate", END)

graph = builder.compile()
```

Four things to notice, because each is a real conceptual shift from LCEL:

**1. Nodes return *partial* updates, not new state.** `classify` returns `{"intent": "billing"}` — it says nothing about `messages` or `escalated`, and those are left alone. You're describing a delta, not rebuilding the world. (This is why the merge rules in layer C matter.)

**2. Edges are declared separately from nodes.** In LCEL, `a | b` fuses "what runs" and "what runs next" into one expression. Here they're separate: `add_node` declares capability, `add_edge` declares flow. That separation is what lets a node have two incoming edges, or an edge point backwards.

**3. A conditional edge is a router function that reads state and returns the *name* of the next node.** That's the whole mechanism. It's a plain Python function — so it can be a threshold check, a lookup table, or an LLM call. **This one function is where "partial control flow definition" from Module 1 physically lives.** You defined that `classify` runs first and that both branches end; the router decides which branch. Developer owns the skeleton, model owns the choice.

**4. `START` and `END` are real sentinel nodes.** `START` is where input enters, `END` is where execution halts. An edge to `END` is how a path terminates — termination is explicit, not implicit.

### A note on the Functional API

The dialogue mentions decorators — `@task` and `@entrypoint`:

> **Karthik:** That's where the functional API shines. We use decorators like `@task` and `@entrypoint` to define logic blocks, then attach tools and memory as needed. It's clean, testable, and scalable.

LangGraph offers **two front-ends to the same engine**:

- **Graph API** (`StateGraph`) — you declare nodes and edges explicitly. The structure is data, so it can be visualised and reasoned about.
- **Functional API** (`@entrypoint`, `@task`) — you write ordinary Python with `if`/`while`/`await`, and control flow is implicit in the code. You still get checkpointing, streaming, and interrupts.

```python
from langgraph.func import entrypoint, task

@task
def classify(text: str) -> str:
    return "billing"

@entrypoint(checkpointer=checkpointer)
def support_bot(text: str):
    intent = classify(text).result()
    return escalate(text).result() if intent == "billing" else answer(text).result()
```

Both compile down to the same Pregel execution. Rule of thumb: **Graph API when the structure is the point** (you want to see it, share it, nest it); **Functional API when you have existing procedural code** you want to make durable without restructuring it. Module 5 covers both properly — for now just know they aren't rival frameworks.

---

## Layer B — Execution: the Pregel engine

> LangGraph is inspired by **Pregel**, a graph-parallel computation model from Google.

The source states this and moves on. Don't let it — this is the layer that explains behaviour you will otherwise find baffling.

Pregel is Google's system for computing over graphs too large for one machine (PageRank over the web, for instance). Its execution model is called **BSP — Bulk Synchronous Parallel** — and its unit of work is the **super-step**.

### What a super-step is

One super-step is three phases:

```
  ┌──── SUPER-STEP N ─────────────────────────────────────────┐
  │                                                            │
  │  1. PLAN   Which nodes were activated by the last step?    │
  │                                                            │
  │  2. EXECUTE   Run all of them — in parallel.               │
  │               Each gets the state as it was at the start.  │
  │               Each returns a partial update.               │
  │               Nobody sees anybody else's update yet.       │
  │                                                            │
  │  3. UPDATE (the barrier)                                   │
  │               All updates land together.                   │
  │               Reducers merge conflicts.                    │
  │               → checkpoint written here                    │
  │               → next nodes activated                       │
  └────────────────────────────────────────────────────────────┘
                              ↓
                     SUPER-STEP N+1
```

The **barrier** is the important part. Nodes running in the same super-step are isolated from each other — they all read the same starting state and their writes are applied together at the end. There's no read-your-neighbour's-write, no ordering dependency, no race.

That single design choice is where five separate LangGraph features come from:

| Feature | Falls out of super-steps because… |
|---------|-----------------------------------|
| **Parallel node execution** | Everything activated in a super-step runs concurrently by default. Fan out to three retrievers and they all go at once. |
| **Reducers are necessary** | Two parallel nodes can write the same key. Something must decide how to merge. That something is a reducer. |
| **Checkpoint granularity** | The barrier is the only moment state is consistent and complete — so that's exactly when a checkpoint is written. One checkpoint per super-step. |
| **Interrupts / human-in-the-loop** | Pausing means stopping at a barrier. State is coherent there, so it's safe to persist, walk away, and resume tomorrow. |
| **Deterministic replay** | Reload the checkpoint from super-step N and the engine can rebuild exactly which nodes were pending. |

Retries and error handling live here too: a node can carry a retry policy, and because the engine knows the last good checkpoint, a failure doesn't cost you the whole run.

### Concretely, in Anita's bot

Suppose after classifying an intent you want to search the docs *and* look up the customer's account at the same time:

```
super-step 1:  [classify]                        → {"intent": "billing"}
super-step 2:  [search_docs] ‖ [fetch_account]   → both run in parallel
                                                  → both write to "context"
                                                  → reducer merges at the barrier
super-step 3:  [answer]                           → reads merged context
```

Super-step 2 is one step, not two, even though two nodes ran. Wall-clock cost is the slower of the two, not the sum. And `answer` sees both results, merged, without you writing any coordination code.

---

## Layer C — State Management: channels and reducers

Layer C is where the state object lives. Two ideas.

### Channels

Each key in your state schema is a **channel** — an independently-tracked slot that nodes read from and write to. `messages` is a channel. `intent` is a channel. Nodes communicate *only* through channels; there are no direct node-to-node calls. That indirection is what makes nodes independently swappable.

### Reducers

A **reducer** is the merge rule for a channel: *given the current value and an incoming update, what's the new value?*

Every channel has one. If you don't specify it, the default is **overwrite** — last write wins.

```python
class State(TypedDict):
    intent: str                                   # default reducer: overwrite
    messages: Annotated[list, add_messages]       # reducer: append messages
    retrieved: Annotated[list, operator.add]      # reducer: concatenate lists
    attempts: Annotated[int, operator.add]        # reducer: sum
```

Read `Annotated[list, operator.add]` as "this channel is a list, and updates are combined with `+`". Nothing more mysterious than that.

Why it matters, in one example. `answer` returns `{"messages": [("ai", "…")]}`. With the default overwrite reducer, that **replaces the entire conversation history with a single message** — the bot instantly forgets everything. With `add_messages`, it appends. Same node code, opposite behaviour, and the only difference is the annotation on the state schema.

> Most "my LangGraph agent lost its memory" bugs are a missing reducer.

`add_messages` is worth knowing specifically: it's smarter than `+`. It appends new messages, and it *replaces* existing ones when IDs match — which is what makes editing history (and therefore time travel over a conversation) work.

And back to layer B: reducers are not optional bookkeeping, they're the mechanism that makes parallelism safe. If two nodes in the same super-step write to the same channel and there's no reducer to merge them, LangGraph doesn't silently pick one — it raises an error, because there's no defensible answer.

### Short-term vs. long-term memory

The state layer supports two distinct scopes, and conflating them is a common design mistake:

| | Short-term | Long-term |
|---|-----------|-----------|
| **Scope** | One thread / conversation | Across all sessions for a user |
| **Holds** | Messages, working variables, flags | Preferences, learned facts, history |
| **Backed by** | Checkpointer (state per super-step) | Store (a key-value namespace) |
| **Question it answers** | "What did we just say?" | "Who is this person?" |

Anita's bot needs both: short-term to keep the current ticket coherent, long-term to know this customer is on the enterprise plan and has been burned by this bug before. Module 5's Memory topic goes deep here.

### Checkpointing and time travel

> **Anita:** Can we inspect how the state evolves? Debug transitions, replay a session?
>
> **Karthik:** Yes — LangGraph has time travel and state inspection. You can trace how state changed at each step, which is great for debugging *and* compliance.

The mechanism, now that you have super-steps: attach a **checkpointer**, and state is persisted at every barrier, tagged with a thread ID.

```python
from langgraph.checkpoint.memory import InMemorySaver   # legacy alias: MemorySaver

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ticket-4711"}}

graph.invoke({"messages": [("user", "Why was I charged twice?")]}, config)

# every super-step, newest first
for snapshot in graph.get_state_history(config):
    print(snapshot.next, snapshot.values["intent"])
```

That gives you three capabilities from one mechanism:

- **Inspection** — the exact state at every step of a past run
- **Replay** — reload any checkpoint and run forward from there
- **Recovery** — a crashed run resumes from its last barrier

The compliance angle Karthik raises is not a throwaway. In a regulated setting, "show me why the system decided to refund this customer" is an audit requirement. A checkpoint history is an answer to it. A pipeline of prompts is not.

---

## StateGraph vs. LangChain's AgentExecutor

You've already used LangChain's prebuilt agent path. Here's the relationship, since the source raises it.

`AgentExecutor` (and `create_react_agent` on top of it) is a **fixed loop that someone else wrote**: call the model → if it requested a tool, run it → feed the result back → repeat until it stops. Great loop. You just can't change it. You can't insert an approval gate before the tool runs, can't cap retries per tool, can't route to a different model when confidence is low, can't checkpoint mid-loop.

`StateGraph` is a **lower-level API** — it doesn't give you the loop, it gives you the primitives the loop is made of. Which means more code for the simple case and full control for everything else.

```
create_react_agent   →  the loop, prebuilt         (LangChain course, Module 8)
StateGraph           →  the primitives it's built from  (this course)
```

Module 4 closes this circle: it revisits `create_react_agent` — which by then you'll recognise as a small, ordinary LangGraph graph, and you'll be able to read its source and see the nodes.

---

## Seeing the shape: a graph is just nodes and edges

Before writing real LangGraph code (Module 3), it's worth building the picture by hand. A graph, stripped of everything LangGraph adds, is a set of nodes and a set of directed edges — nothing more exotic.

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()                      # DiGraph = *directed* graph: edges have a direction

G.add_node("Input")
G.add_node("LLM")
G.add_node("Output")

G.add_edge("Input", "LLM")
G.add_edge("LLM", "Output")

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="lightblue",
        node_size=2000, font_size=12, arrows=True)
plt.title("Simple LangGraph Workflow")
plt.show()
```

```
   (Input) ──▶ (LLM) ──▶ (Output)
```

This is a chain drawn as a graph — every node has one way in and one way out, so it behaves exactly like `Input | LLM | Output`. Which is the point: **a chain is a degenerate graph.** The graph model doesn't replace pipelines, it contains them.

Now add one line:

```python
G.add_edge("LLM", "LLM")      # or: G.add_edge("Output", "LLM")
```

```
        ┌──────┐
        ▼      │
   (Input) ──▶ (LLM) ──▶ (Output)
```

One edge, and you have something no pipeline can express: a reflection loop, a retry, a self-correcting agent. That edge is the entire difference between the two paradigms.

Two caveats so you don't over-read the snippet: `networkx` is a **visualisation and graph-theory library, not part of LangGraph** — it's here as a drawing aid only. And real LangGraph has its own visualisation built in, which you'll use from Module 3 onwards:

```python
graph.get_graph().draw_mermaid_png()   # renders your actual compiled graph
```

---

## Putting the layers back together

The full picture, with Anita and Karthik's four requirements mapped onto it:

```
 THEIR REQUIREMENT              LAYER      MECHANISM
 ────────────────────────────────────────────────────────────────────
 Route by customer intent   →   A          conditional edges
 Pluggable tools/modules    →   A          nodes + subgraphs
 Pause for a human agent    →   B          interrupt at a super-step barrier
 Inspect state evolution    →   B + C      checkpoint per super-step
 Remember the conversation  →   C          messages channel + add_messages reducer
 Remember the customer      →   C          long-term store
 Parallel doc + account     →   B          same super-step, merged by reducer
 lookup
 Survive a mid-run crash    →   B          resume from last checkpoint
```

Every one of those is a Module 5 or 6 topic. You now know which layer each lives in and why it exists — which means the rest of the course is filling in mechanisms you already have a slot for, rather than accumulating unrelated features.

> **Karthik:** I'll set up the initial graph and walk the team through the decorators and the state model.

That's Module 3.

---

## Self-check

1. What are the three layers of LangGraph, and which one do you actually write code in?
2. A node returns `{"intent": "billing"}`. What happens to the other keys in state?
3. What are the three phases of a super-step, and what happens at the barrier?
4. Two nodes run in the same super-step and both write to `context`. What decides the result — and what happens if you haven't specified one?
5. Why is a checkpoint written once per super-step rather than once per node?
6. Explain "deterministic transitions even with non-deterministic LLMs" without using the word "deterministic".
7. Where in the code does "partial control flow definition" from Module 1 physically live?
8. Why is a chain a special case of a graph rather than a different thing?
9. When would you reach for the Functional API over the Graph API?
10. What's the difference between short-term and long-term memory, and which one is the checkpointer?
