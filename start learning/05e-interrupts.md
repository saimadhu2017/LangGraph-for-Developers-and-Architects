# Module 5.5 — Interrupts

## Where we left off

The Research Assistant runs from question to answer in one unbroken sweep. `understand` extracts keywords, `search` fetches sources, `summarize` drafts the answer, `evaluate` decides if it's good enough. The human sees the final answer — nothing in between.

That's fine for a demo. It's a liability in production. If `search` found weak or off-topic sources, `summarize` will confidently produce a wrong answer. By the time the human reads it, the damage is done.

The remaining debt: nothing pauses for a human to review the sources before the LLM drafts the answer.

---

## The problem — why you can't just split runs manually

An obvious workaround: run the graph up to `search`, extract the sources, show them to the user, then start a *second* graph run for `summarize`. But this breaks down immediately:

- You'd have to re-submit all the accumulated state (keywords, sources, attempt count, question) as new input to the second run
- The second run re-executes `understand` and `search` — wasting tokens and time, and possibly getting different results
- You've lost the checkpoint history and time-travel capability for the paused run

Interrupts are the built-in solution. The graph pauses mid-run, persists full state via the checkpointer, waits for external input, and resumes exactly where it stopped — same thread, same state, same graph instance.

---

## One requirement: a checkpointer

Interrupts require a checkpointer. Without one, LangGraph raises an error the moment execution tries to pause. The checkpointer is what persists the state at the pause point so resume is possible. We have `InMemorySaver` from Module 5.2 — same mechanism, new use.

---

## Two ways to interrupt

LangGraph offers two distinct approaches. The right choice depends on whether the pause is structural or conditional.

### Approach 1 — `interrupt_before` at compile time

Declare which nodes are interrupt points when building the graph. Zero changes to node code.

```python
graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["summarize"],
)
```

When the graph reaches `summarize`, it pauses *before* running it. State is checkpointed. The caller gets control back with `snapshot.next == ("summarize",)` — the node hasn't run yet.

To resume, re-invoke with `None` as input (no new state to inject):

```python
graph.invoke(None, config=config)   # resumes from the checkpoint
```

Or pass a dict if you want to inject updated state on resume (uses the normal reducer rules):

```python
# Human wants to try different keywords before summarizing
graph.update_state(config, {"keywords": ["checkpointing", "persistence"]})
graph.invoke(None, config=config)
```

**When to use:** You always want to pause at this node. The pause is a deployment concern — QA, compliance, review — not business logic. The node code doesn't need to know an interrupt happened.

### Approach 2 — `interrupt()` inside a node

The node itself calls `interrupt(payload)`. The payload is what you want the human to see. On resume, `interrupt()` returns the human's response — and the node can act on it.

```python
from langgraph.types import interrupt

def approve_sources(state: ResearchState) -> dict:
    human_input = interrupt({
        "question": "Approve these sources for the summary?",
        "sources": state["sources"],
        "verdict_so_far": state["verdict"],
    })
    # human_input is whatever was passed to Command(resume=...)
    if human_input == "approve":
        return {}
    return {"verdict": "rejected — human declined sources"}
```

To resume, pass the human's response via `Command`:

```python
from langgraph.types import Command

graph.invoke(Command(resume="approve"), config=config)
```

The string `"approve"` becomes the return value of `interrupt()` inside the node.

**When to use:** The pause is conditional (only pause if the confidence score is low). Or the node needs to receive and act on the human's input — edit the summary, provide missing data, approve/reject with a reason.

---

## What happens when `interrupt()` is called

Step by step:

1. Execution halts at the `interrupt()` call — not at a node boundary, at that exact line
2. Full state is checkpointed by the configured checkpointer
3. The payload is returned to the caller under `result["__interrupt__"]` — a list of `Interrupt` objects
4. The graph waits indefinitely — no timeout

```python
result = graph.invoke({"question": QUESTION, "sources": [], "attempts": 0}, config=config)

# result["__interrupt__"] contains what the node passed to interrupt()
for item in result["__interrupt__"]:
    print(item.value)  # → {"question": "Approve...", "sources": [...]}
```

On resume with `graph.invoke(Command(resume="approve"), config=config)`:

1. The node **restarts from the beginning**
2. `interrupt()` returns `"approve"` instead of pausing again
3. Execution continues from that point

---

## The re-execution gotcha

This is the most important operational detail. When a node resumes, it **restarts from line 1**. Any code before `interrupt()` runs again.

```python
def my_node(state):
    result = call_expensive_llm(state)   # ← RUNS AGAIN on resume
    approved = interrupt(result)
    return {"result": result}
```

On resume, `call_expensive_llm` fires a second time. You're billed twice, you get a different response, and the `result` the human approved may not match the one the node uses.

**Fix:** Write expensive results to state before calling `interrupt()`. On resume, read from state instead of recomputing:

```python
def my_node(state):
    # On resume, expensive_result is already in state from the first run
    if not state.get("expensive_result"):
        result = call_expensive_llm(state)
        # interrupt() will checkpoint this write before pausing
        # ... but we can't write before interrupt() returns ...
```

Actually the cleaner fix is to split the work across two nodes: one node does the expensive work and writes to state, the next node calls `interrupt()`. The first node's result is checkpointed between them.

```
compute_node → interrupt_node → continue_node
```

The `interrupt_before` approach naturally gives you this split — you put the interrupt *between* two nodes without touching either.

---

## `Command` — two uses

`Command` from `langgraph.types` has two distinct uses:

**1. Carrying the resume value:**
```python
graph.invoke(Command(resume="approve"), config=config)
```
The value passed to `resume=` becomes the return value of `interrupt()` inside the node. Can be any JSON-serializable object — string, bool, dict, list.

**2. Routing after a node (via `goto`):**
```python
def approval_node(state):
    decision = interrupt("Approve or reject?")
    if decision == "approve":
        return Command(goto="summarize")
    return Command(goto="search")   # try again with different sources
```

`Command(goto="node_name")` returned from a node redirects the graph to that node next, bypassing the normal edge logic. Use this when the interrupt's response should override the routing — not as a general-purpose router (that's what conditional edges are for).

---

## Four interrupt design patterns (from the source)

| Pattern | What it does | Approach |
|---|---|---|
| **Approval Workflow** | Pause before irreversible action (DB write, payment, email) | `interrupt()` inside node, reject path routes to abort |
| **Review and Edit** | Human can see and modify LLM output before it flows downstream | `interrupt()` with LLM output as payload; resume value can be edited version |
| **Tool Invocation Control** | Interrupt inside a `@tool` — show tool args before they fire | `interrupt()` inside the `@tool` function itself |
| **Input Validation** | Pause to collect or verify user input mid-workflow | `interrupt_before` a node that requires validated input; inject via `update_state` |

### Interrupts inside tool functions

The source highlights this pattern. You can put `interrupt()` inside a `@tool` function:

```python
from langchain_core.tools import tool
from langgraph.types import interrupt

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    # Pause before sending — show the human what's about to go out
    response = interrupt({
        "action": "send_email",
        "to": to, "subject": subject, "body": body,
        "message": "Approve sending this email?",
    })
    if response.get("action") == "approve":
        # Human can override any of the original args
        final_to = response.get("to", to)
        final_body = response.get("body", body)
        return f"Email sent to {final_to}"
    return "Email cancelled by user"
```

The tool becomes interactive — it halts execution and waits for a response before doing anything irreversible. This applies the approval pattern at the tool level rather than the graph level.

---

## `interrupt_before` vs `interrupt_after`

`compile()` accepts both:

```python
graph = builder.compile(
    checkpointer=saver,
    interrupt_before=["summarize"],    # pause before the node runs
    interrupt_after=["evaluate"],      # pause after the node runs
)
```

`interrupt_before` is more common — you pause to check preconditions and possibly inject new input before the node executes. `interrupt_after` is useful for post-hoc inspection or triggering an external async workflow after a node completes.

---

## Project v0.7 — pause before summarize

`interrupt_before=["summarize"]` requires one change to the graph builder and one new demo function. No node code changes.

### Updated `research_assistant/graph.py`

```python
def build_graph(checkpointer=None, interrupt_before=None):
    builder = StateGraph(ResearchState)

    builder.add_node("understand", understand)
    builder.add_node("search", search)
    builder.add_node("summarize", wrap_with_hooks(summarize, pre=pre_summarize, post=post_summarize))
    builder.add_node("evaluate", evaluate)

    builder.add_edge(START, "understand")
    builder.add_edge("understand", "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "evaluate")

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"retry": "search", "done": END, "give_up": END},
    )

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )
```

One new parameter. `interrupt_before=None` means no interrupts — all existing `run_v3()` / `run_v4()` / `run_v5()` calls are unaffected.

### New `run_v7_interrupt()` in `main.py`

```python
def run_v7_interrupt():
    print("=" * 60)
    print("v0.7 — Interrupt: pause before summarize for source review")
    print("=" * 60)

    checkpointer = InMemorySaver()
    graph = build_graph(
        checkpointer=checkpointer,
        interrupt_before=["summarize"],
    )
    config = {"configurable": {"thread_id": "interrupt-demo"}}

    # Phase 1: run until the graph pauses before summarize
    print("--- phase 1: searching (will pause before summarize) ---")
    for step in graph.stream(
        {"question": QUESTION, "sources": [], "attempts": 0},
        config=config,
    ):
        for node, update in step.items():
            keys = ", ".join(f"{k}={_short(v)}" for k, v in update.items())
            print(f"  [{node}] -> {keys}")

    snapshot = graph.get_state(config)
    print(f"\n--- paused: next={snapshot.next} ---")
    print("  sources ready for review:")
    for s in snapshot.values.get("sources", []):
        print(f"    - {s['title']}")

    # Human reviews sources here. We simulate approval.
    print("\n--- human decision: APPROVED — resuming ---")

    # Phase 2: resume — None input, same config
    for step in graph.stream(None, config=config):
        for node, update in step.items():
            keys = ", ".join(f"{k}={_short(v)}" for k, v in update.items())
            print(f"  [{node}] -> {keys}")

    final = graph.get_state(config)
    print("\n--- final ---")
    print("  verdict :", final.values.get("verdict"))
    print("  sources :", len(final.values.get("sources", [])))
    print("  summary :", _short(final.values.get("summary", ""), 120))
```

Run it and you see two distinct phases:
```
[understand] -> keywords=...
[search]     -> sources=..., attempts=1
--- paused: next=('summarize',) ---
  sources ready for review:
    - LangGraph Docs — Core Concepts
    - Pregel and Super-Steps
    - Channels and Reducers
--- human decision: APPROVED — resuming ---
[summarize]  -> summary=...
[evaluate]   -> verdict=ok — 3 sources
```

The graph ran `understand` and `search`, checkpointed, handed back control. The human inspected `snapshot.values["sources"]`. We resumed with `None`. The graph picked up exactly where it stopped — ran `summarize` on the already-found sources.

---

## File layout after v0.7

```
research_assistant/
├── state.py          — unchanged
├── corpus.py         — unchanged
├── tools.py          — unchanged (v0.6)
├── nodes.py          — unchanged
├── hooks.py          — unchanged
├── graph.py          — updated: interrupt_before parameter added
├── agent.py          — unchanged
├── controller_demo.py — unchanged
├── tools_demo.py     — unchanged (v0.6)
└── main.py           — updated: run_v7_interrupt() added
```

---

## Dynamic vs. static pausing — one more distinction

The source draws a line between "static breakpoints" (compile-time `interrupt_before`) and "dynamic interrupts" (`interrupt()` inside a node). The key difference:

| | `interrupt_before` | `interrupt()` in node |
|---|---|---|
| Declared | At compile time | At runtime, inside node logic |
| Conditional | Always pauses at that node | Can be inside an `if` — pauses only when the condition holds |
| Resume value | No value flows back (resume with `None` or `update_state`) | Resume value = return of `interrupt()` |
| Node re-execution | Node hasn't started; starts fresh | Node restarts from line 1 — re-execution gotcha applies |
| Use for | QA gates, deployment policies | Conditional approval, mid-node data collection |

---

## Self-check

1. Why does an interrupt require a checkpointer? What exactly does the checkpointer save at the pause point?
2. `interrupt_before=["summarize"]` — what is `next` on the state snapshot after the graph pauses?
3. How do you resume a graph paused with `interrupt_before`? How does it differ from resuming a graph paused by `interrupt()` inside a node?
4. A node calls `expensive_llm()` and then `interrupt()`. What happens on resume and why is this a problem?
5. `Command(resume="approve")` vs `Command(goto="summarize")` — when would you use each?
6. The Research Assistant uses `interrupt_before` rather than `interrupt()` inside a node. Why is that the right choice here?
