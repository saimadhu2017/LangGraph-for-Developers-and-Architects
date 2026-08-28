# Module 2a — LangGraph in the LangChain Ecosystem

*Sub-topic: LangGraph in the LangChain Ecosystem (15m)*

---

## Where we left off

Module 1 answered **why** LangGraph exists.

The short version: *control flow* is the question of **who decides what runs next**. A chain answers "the developer, at build time" — same path every invocation, predictable but rigid. An autonomous agent answers "the LLM, at runtime" — flexible but unpredictable. Neither is good enough for production, because agency and reliability pull against each other.

LangGraph's answer is **partial control flow definition**: you hard-wire the steps that must not vary, and let the LLM decide the rest. That's the innovation. Everything else in the course is machinery for expressing it.

This module moves from *why* to **how**. And it starts with the question every developer asks first.

---

## The question you're actually asking

You just finished a LangChain course. You already have `create_react_agent` working. So:

> "Is LangGraph a replacement for LangChain? A layer on top of it? Do I need both? Did I just learn the wrong thing?"

Here is the honest answer, and it's worth reading twice:

**LangGraph is a standalone framework. It does not require LangChain. But it composes with LangChain perfectly, and in practice you'll use both.**

Let's unpack why both halves of that are true.

### Why LangGraph doesn't need LangChain

LangGraph's job is orchestration: nodes, edges, state, checkpoints. None of that knows or cares what a node *does*. A node is just a Python function that takes state and returns a state update.

```python
def my_node(state):
    return {"count": state["count"] + 1}
```

There is no LLM in that node. No `ChatPromptTemplate`, no `Runnable`, no LangChain import at all. LangGraph will happily run it. You could build a graph that calls the OpenAI SDK directly, or `requests.post`, or a local model, or nothing but pure Python.

So LangGraph is not a LangChain plugin. It's an orchestration engine that happens to be built by the same team.

### Why you'll use LangChain anyway

Because orchestration is the *only* thing LangGraph does. It has no prompt templates. No output parsers. No document loaders. No embeddings. No retrievers. No `@tool` decorator with automatic schema generation.

All of that already exists, works well, and you already know it. LangGraph doesn't reinvent it — it calls it.

So the real relationship is:

```
LangChain  →  the components (what happens inside a node)
LangGraph  →  the control flow (which node runs, when, and with what state)
```

**LangGraph orchestrates LangChain components. It does not replace them.** Your LCEL knowledge is not wasted — a chain becomes the *body of a node*, and now that node can loop, branch, pause for a human, and resume from a crash.

That's the whole pivot of this course in one sentence.

---

## The three-tool ecosystem

The LangChain team ships three products. They are frequently confused, because the naming doesn't help. Here's the division of labour, in the order you'd reach for them:

| Tool | One-line job | Reach for it when… |
|------|--------------|--------------------|
| **LangChain** | High-level framework for building LLM apps | You want a working agent or RAG chain *today*, with any model provider |
| **LangGraph** | Low-level orchestration for agentic workflows | The prebuilt agent isn't enough — you need custom control flow, real state, loops, human approval |
| **LangSmith** | Observability, evaluation, deployment | It works on your laptop and now has to work in production, measurably |

A useful way to hold it: **LangChain gets you to a demo. LangGraph gets you to a system. LangSmith keeps you honest about whether that system is any good.**

They're layered, not competing:

```
        ┌─────────────────────────────────────────────┐
        │  LangSmith — traces, evals, deployment      │  ← observes everything below
        ├─────────────────────────────────────────────┤
        │  LangGraph — nodes, edges, state, checkpoints│  ← decides what runs next
        ├─────────────────────────────────────────────┤
        │  LangChain — prompts, models, tools, retrieval│ ← does the actual work
        └─────────────────────────────────────────────┘
```

You can use any one without the others. You'll usually use all three.

---

## The comparison, aspect by aspect

The source material gives a full comparison grid. Here it is, with the LangSmith column folded in where it's meaningful and dropped where it isn't (LangSmith doesn't *have* a control flow model — it watches yours).

| Aspect | LangChain | LangGraph |
|--------|-----------|-----------|
| **Primary purpose** | High-level framework for LLM apps | Low-level orchestration for agentic workflows |
| **Control flow** | Linear chains, simple prebuilt agents | Graph-based, dynamic routing driven by state |
| **Flexibility** | Limited — fixed chains, basic agents | High — custom graphs, conditional edges, nested subgraphs |
| **State management** | Minimal — session-scoped chat memory | Advanced — explicit shared state, reducers, checkpointing |
| **Human-in-the-loop** | Possible, but you build the plumbing | First-class: pause the graph, wait, resume |
| **Streaming** | Yes — token streaming from the LLM | Yes — plus node-by-node and state-delta streaming for long tasks |
| **Loops & retries** | Awkward — a chain runs forward only | Native — an edge can point backwards |
| **Debuggability** | Print statements and hope | Inspect state at every step; replay from any checkpoint |
| **Licence** | MIT, open source | MIT, open source |
| **Best fit** | Prototyping, RAG, simple tool-using agents | Complex, stateful, multi-agent, production systems |

Two rows deserve a closer look, because the difference is bigger than the table makes it sound.

**State management.** In LangChain, memory is a thing you bolt on: `RunnableWithMessageHistory` wraps your chain and injects past messages into a `MessagesPlaceholder`. It's memory *for the prompt*. In LangGraph, state is the thing the whole graph is built around — every node reads it, every node writes to it, and it can hold anything, not just messages. Retry counters, retrieved documents, a confidence score, a flag saying "a human already approved this". Routing decisions read from it. That's a category difference, not a feature difference.

**Loops.** This is the one that matters most in practice. A LangChain chain is a pipeline — `A | B | C`. There is no way to make `C` send work back to `B`, because the pipe operator only composes forwards. If step C discovers the retrieved documents were irrelevant, the chain's only options are "return a bad answer" or "raise". A LangGraph edge can point from C back to B, so the graph retries the retrieval with a rewritten query — carrying forward everything it learned on the first attempt, because that's in state.

---

## Straight answers to the FAQ

The source poses these as questions. A few of its answers are thin, so here are fuller ones.

**Is LangChain required to use LangGraph?**
No. LangGraph is standalone. `pip install langgraph` and you have nodes, edges, state, and checkpointing. You'll usually want LangChain too, for everything *inside* the nodes.

**What makes LangGraph different from other agent frameworks?**
Most agent frameworks hand you a black box: you configure an agent, call `.run()`, and hope. That's fine for generic tasks and hostile to customisation — when the behaviour is wrong, there's no seam to reach into. LangGraph inverts it: the control flow *is* your code. You wrote which node follows which, so when it misbehaves you can see exactly where and change exactly that. The differentiator isn't a feature list; it's that nothing is hidden.

**Does LangGraph slow my application down?**
Not meaningfully, and it's worth knowing *why* rather than taking "no" on faith. Graph overhead is function dispatch and dict merging — microseconds. Your LLM call is hundreds of milliseconds to seconds. The graph is rounding error next to the model.

The one place you *can* pay is **checkpointing**: if you persist state to Postgres after every super-step, that's real I/O on every step. It's usually worth it (that's what buys you resumability and time travel), but it's a deliberate trade, and for a fast in-process graph you'd use an in-memory checkpointer instead. Streaming exists precisely so long-running graphs stay responsive while they work.

**Is it free and open source?**
Yes — MIT licensed, like LangChain. Free for commercial use, no strings.

**Do I need LangSmith to deploy LangGraph?**
The source says deployment "requires LangSmith". Treat that as marketing shorthand and don't build a mental model on it. LangGraph is a Python library — you can deploy it inside a FastAPI service, a Lambda, a Celery worker, or anything else that runs Python, with no LangChain-hosted product involved. What the hosted offering (**LangGraph Platform**, sold alongside LangSmith) actually gives you is managed infrastructure: a persistence layer, task queues for long-running runs, horizontal scaling, and a UI. Convenient, not mandatory. Self-hosting is a fully supported path.

---

## What this means for the rest of the course

You now know where LangGraph sits. From here on, the mental split is:

- When a lesson talks about **prompts, models, tools, retrieval** — that's LangChain, and you already know it. It'll be used without ceremony.
- When a lesson talks about **nodes, edges, state, reducers, checkpoints, interrupts, subgraphs, streaming** — that's LangGraph, and that's the new material.

Next up is the architecture itself: the three layers LangGraph is built from, the Pregel execution model underneath it, and why a graph is the right shape for this problem in the first place.

---

## Self-check

1. Can you build a LangGraph graph with zero LangChain imports? Why or why not?
2. Where does a LangChain LCEL chain live in a LangGraph application?
3. Name the one-line job of each of LangChain, LangGraph, and LangSmith.
4. LangChain has memory. Why is LangGraph's state described as a *category* difference rather than a better version of the same thing?
5. Why can't a LangChain chain retry a failed step, structurally?
6. Where does LangGraph actually cost you performance — and what buys you in return?
