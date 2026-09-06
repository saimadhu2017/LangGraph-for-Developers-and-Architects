# Module 5.2 — Persistence

## Where we left off

v0.3 is an agent. It loops, retries, and stops when satisfied. But `build_graph()` calls `compile()` with no arguments. No checkpointer. Every run starts from a blank slate.

Here is the scenario that hurts. The graph is on its second search attempt:

```
superstep 1: understand   → keywords extracted
superstep 2: search       → 3 docs found, saved to sources
superstep 3: summarize    → summary drafted
superstep 4: evaluate     → "weak — only 3 sources, wanted 4"
  router: "retry"
superstep 5: search (2nd) → process crashes
```

Restart. The graph starts from zero. Attempt 1's three documents are gone. You pay for that search twice.

Persistence is the fix. Like every feature in the rest of Module 5 — Interrupts, Time Travel, Memory — it is one argument to `compile()`.

---

## What a checkpointer does

A checkpointer is a storage backend. When you compile with one, LangGraph writes the full state to storage at the end of every **superstep** — the Pregel barrier, the moment all active nodes have finished and their writes have been merged. Not per-node. Per-superstep, because that is the only moment state is internally consistent.

For a two-attempt Research Assistant run:

```
[understand]   → superstep 1 → CHECKPOINT  {keywords: [...]}
[search]       → superstep 2 → CHECKPOINT  {sources: [3 docs], attempts: 1}
[summarize]    → superstep 3 → CHECKPOINT  {summary: "..."}
[evaluate]     → superstep 4 → CHECKPOINT  {verdict: "weak — only 3 sources"}
  router: "retry"
[search]       → superstep 5 → CHECKPOINT  {sources: [4 docs], attempts: 2}
[summarize]    → superstep 6 → CHECKPOINT  {summary: "..."}
[evaluate]     → superstep 7 → CHECKPOINT  {verdict: "ok — 4 sources"}
  router: "done"
END
```

Crash between supersteps 4 and 5? Re-invoke with the same thread ID. LangGraph loads superstep 4's checkpoint — 3 docs in `sources`, `attempts=1`, `verdict="weak"` — and picks up from there. The retry is not starting blind.

---

## What a checkpoint holds

Each checkpoint is a `StateSnapshot` object with five fields:

| Field | What it contains |
|---|---|
| `values` | The full state dict at this point in time |
| `next` | Tuple of node names queued to run next. Empty tuple means the graph reached END. |
| `config` | The run config that produced this checkpoint |
| `metadata` | Step number, source ("loop" / "input" / "fork"), and what was written this step |
| `tasks` | PregelTask objects — what ran this superstep, plus error info if the step failed |

`next` is the field to watch. `snapshot.next == ()` means the graph finished cleanly. `snapshot.next == ("search",)` means it was paused before the search node ran — that is the Interrupts sub-topic in 5.5.

---

## What a thread is

A thread is a named container for a sequence of checkpoints. Every checkpoint belongs to exactly one thread. The ID is just a string you choose:

```python
config = {"configurable": {"thread_id": "research-session-42"}}
```

All invocations sharing the same `thread_id` read from and write to the same checkpoint history. That is the entire persistence model.

**Rule of thumb: one thread = one conversation, one task instance, or one document being processed.**

For the Research Assistant: each question gets its own thread ID. Different questions should not share checkpoint history, because `sources` uses `operator.add` — reusing a thread would accumulate documents across unrelated questions.

---

## Checkpointer implementations

Three options, in ascending order of durability:

| Checkpointer | State lives in | Survives process restart | Use when |
|---|---|---|---|
| `InMemorySaver` | Python process RAM | No | Development, unit tests |
| `SqliteSaver` | SQLite file on disk | Yes | Single-machine production |
| `PostgresSaver` | PostgreSQL database | Yes, across servers | Distributed / multi-instance |

All three implement the same `BaseCheckpointSaver` interface. **Switching from dev to production is one import change.** Nodes, edges, `get_state`, `get_state_history` — none of that changes.

```python
# Development
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# Production, single machine
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Production, distributed
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
```

> **On naming:** The course source uses `MemorySaver`. That is a legacy alias for `InMemorySaver` — both live in `langgraph.checkpoint.memory` and work identically. `InMemorySaver` is the canonical name in current LangGraph.

---

## The two-line mechanic

**Line 1 — compile with a checkpointer:**

```python
graph = builder.compile(checkpointer=InMemorySaver())
```

**Line 2 — pass a thread ID in config on every invoke or stream call:**

```python
config = {"configurable": {"thread_id": "research-session-42"}}
result = graph.invoke({"question": "..."}, config=config)
```

That is it. LangGraph writes the checkpoint after each superstep automatically. You do not touch the checkpointer in your nodes.

**One gotcha:** if you have a checkpointer but omit `config`, the call still runs — but writes to the `None` thread, which silently collides across requests in a server environment. Always pass config when your graph has a checkpointer.

---

## Configuration: separate from state

`thread_id` goes in `config["configurable"]`, not in the state `TypedDict`. That is intentional.

**State** = data the graph computes and routes on. Nodes read it, nodes write it, reducers merge it, checkpoints preserve it.

**Config** = execution parameters that influence behaviour but are not part of the output. Thread ID, user ID, model choice, document limit. These travel alongside the graph run without being stored as application data. Nodes can read config via a `RunnableConfig` parameter — that is covered in 5.4 when we wire up real retrievers — but they do not write to it.

```python
# Wrong — runtime parameters embedded in application state
class ResearchState(TypedDict):
    question: str
    thread_id: str   # does not belong here
    user_id: str     # does not belong here

# Right — state holds only what the graph computes
class ResearchState(TypedDict):
    question: str
    keywords: list[str]
    sources: Annotated[list[Source], operator.add]
    summary: str
    verdict: str
    attempts: Annotated[int, operator.add]

# Runtime parameters go in config
config = {"configurable": {
    "thread_id": "research-001",
    "user_id": "alice",
}}
```

The checkpointer reads `config["configurable"]["thread_id"]` automatically. You never plumb it through nodes.

---

## Reading state

Two methods. Both take the config to identify the thread:

```python
# Latest checkpoint for this thread
snapshot = graph.get_state(config)

snapshot.values     # full state dict: {'question': '...', 'sources': [...], 'verdict': 'ok...'}
snapshot.next       # () if graph finished, ('search',) if interrupted before search
snapshot.metadata   # {'step': 7, 'source': 'loop', 'writes': {'evaluate': {...}}}
```

```python
# All checkpoints for this thread, newest first
for snapshot in graph.get_state_history(config):
    step = snapshot.metadata.get("step")
    print(step, snapshot.next, snapshot.values.get("attempts"))

# Example output for a 2-attempt run:
# 7  ()              2
# 6  ('evaluate',)   2
# 5  ('summarize',)  2
# 4  ('search',)     1   ← the retry branched from here
# 3  ('evaluate',)   1
# 2  ('summarize',)  1
# 1  ('search',)     0
# 0  ('understand',) 0
```

`get_state_history` returns a generator of `StateSnapshot` objects. Module 5.3 (Time Travel) is built on this list: pick any snapshot, fork the run from that point, observe how the outcome changes.

---

## Single-turn vs multi-turn memory

The checkpointer is the same mechanism for both purposes.

**Single-turn (fault tolerance):** Within one `invoke`/`stream` call, a crash mid-loop is recoverable. Re-invoke with the same thread ID and the graph resumes from the last written checkpoint. From the user's perspective: the call failed, you retry it, it continues where it stopped rather than starting over.

**Multi-turn (conversational continuity):** Two separate `invoke` calls with the same thread ID share state. The second call loads the first call's final checkpoint, merges the new input via reducers, and continues. This is how a support bot remembers context from message 1 when processing message 5 — the `messages` channel keeps accumulating via `add_messages`.

For the Research Assistant, single-turn fault tolerance is the relevant use. Use a fresh thread per question.

---

## How resume works — the merge step

When you invoke with an existing thread ID, LangGraph does three things in order:

1. **Load** — finds the latest checkpoint for that thread, uses its `values` as the base state
2. **Merge** — for each key in your new input dict, applies the key's reducer against the loaded base
3. **Run** — executes from the `next` nodes in the checkpoint

The merge step is where reducers appear again:

```
Checkpointed state: {question: "A", sources: [d1,d2,d3], attempts: 1}
New input:          {question: "B"}

After merge:
  question → default reducer (overwrite): "B" replaces "A"
  sources  → operator.add: [d1,d2,d3] + nothing = [d1,d2,d3] (key absent in input, unchanged)
  attempts → operator.add: 1 + nothing = 1 (unchanged)
```

Keys not present in the new input are left exactly as checkpointed. That is why a crash-resume works with `{}` as the new input — nothing is overwritten, and the graph continues from exactly where it stopped.

---

## Clearing ephemeral state in multi-turn graphs

In a chatbot that reuses one thread across turns, some state keys are per-turn and should reset between turns — the course calls these "ephemeral state." The pattern: your final node returns `{"per_turn_key": reset_value}` before reaching END. That checkpoint becomes the base for the next turn, with those fields already cleared.

For the Research Assistant this is not needed because we use one thread per question. For a support bot where `draft_response` or `tool_results` should not bleed between messages, clearing in the final node is the right place.

---

## Source note: node mutation pattern

The course's code examples for this section show nodes mutating state in place and returning the full dict:

```python
def node_a(state):
    state["a"] = "Hello"   # in-place mutation
    return state            # full state returned
```

We corrected this in 5.1. The checkpointer itself does not care — it snapshots whatever state looks like at the superstep barrier — but in-place mutation causes the duplication bug under `operator.add` reducers (the reducer sees the already-mutated value and adds it again). Always return only what changed:

```python
def node_a(state):
    return {"a": "Hello"}   # partial update
```

---

## Project → v0.4

Two changes. `graph.py` gains a `checkpointer` parameter. `main.py` gains `run_v4()`.

**`graph.py`** — one-line change to `build_graph()`:

```python
def build_graph(checkpointer=None):
    # ... nodes and edges unchanged ...
    return builder.compile(checkpointer=checkpointer)
```

**`main.py`** — `run_v4()` builds with `InMemorySaver`, runs one question with a thread ID, then prints the final snapshot and walks the full checkpoint history:

```python
def run_v4():
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "research-001"}}

    print("=" * 60)
    print("v0.4 — InMemorySaver + thread_id")
    print("=" * 60)

    for step in graph.stream(
        {"question": QUESTION, "sources": [], "attempts": 0},
        config=config,
    ):
        for node, update in step.items():
            keys = ", ".join(f"{k}={_short(v)}" for k, v in update.items())
            print(f"  [{node}] -> {keys}")

    # Latest snapshot
    snapshot = graph.get_state(config)
    print("\n--- final snapshot ---")
    print("  verdict  :", snapshot.values.get("verdict"))
    print("  attempts :", snapshot.values.get("attempts"))
    print("  sources  :", len(snapshot.values.get("sources", [])))
    print("  next     :", snapshot.next)   # () means finished cleanly

    # Full checkpoint history, newest first
    print("\n--- checkpoint history (newest first) ---")
    for snap in graph.get_state_history(config):
        step_n = snap.metadata.get("step", "?")
        nxt    = snap.next or ("__end__",)
        srcs   = len(snap.values.get("sources", []))
        att    = snap.values.get("attempts", 0)
        print(f"  step {step_n:>2} | next={nxt} | sources={srcs} | attempts={att}")
```

Run it, and the checkpoint history makes the superstep sequence visible as a data structure — not just console output but a queryable history. That same history is what Time Travel (5.3) navigates.

---

## Self-check

- What is the difference between a checkpointer and a thread?
- Why are checkpoints written per superstep rather than per node?
- Which five fields does a `StateSnapshot` have, and which one tells you whether the graph finished?
- What is the difference between `InMemorySaver` and `SqliteSaver`? How much code changes when you switch?
- Why does `thread_id` go in `config["configurable"]` rather than in the state `TypedDict`?
- What does LangGraph do when you invoke with a thread ID that already has a checkpoint?
- How do reducers affect the merge step during resume?
- What is the difference between single-turn (fault tolerance) and multi-turn (conversational continuity) uses of the checkpointer?
- What is `get_state_history` used for, and which sub-topic builds on it directly?
