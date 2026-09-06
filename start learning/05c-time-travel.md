# Module 5.3 — Time Travel

## Where we left off

v0.4 gave us 8 checkpoints in history — one per superstep — stored in `InMemorySaver` and printed by `run_v4()`. We can read them, but we can't navigate them yet. That's what this lesson adds.

---

## The core idea: `checkpoint_id` is an address

Every `StateSnapshot` has a config that includes its checkpoint ID:

```python
snap.config["configurable"]["checkpoint_id"]  # e.g. "1eff2d18-3a71-6fcb-bfde-..."
```

You already saw this in v0.4's history output. The new idea: pass that ID **back** to the graph on a subsequent `invoke` or `stream` call, and the graph re-runs forward from that exact snapshot.

One rule: the graph only runs **forward** from the checkpoint. It re-runs whatever `snap.next` says to run, and nothing before that snapshot is touched. The state it starts from is exactly what was stored.

This is the mechanism behind every "rewind" feature: debugging, fault recovery, human-in-the-loop approval, experimentation. They all do the same two things — find a checkpoint, then re-invoke from it.

---

## The three operations

### 1. `get_state_history` — read with intent

You know this from v0.4. The difference now is reading it with a purpose: find the snapshot whose `next` contains `"search"`. That's the checkpoint after `understand` ran but before the first `search` ran — the ideal fork point, because `understand`'s keyword extraction is already done but no retrieval has happened yet.

```python
history = list(graph.get_state_history(config))
# history is newest-first; reversed() gives oldest-first
fork_snap = next(
    (s for s in reversed(history) if "search" in s.next),
    None,
)
```

`fork_snap.config` already contains the `checkpoint_id`. Pass it directly to the next two operations — no need to extract the ID string manually.

---

### 2. Re-invoke from a past checkpoint — Replay

```python
result = graph.invoke(None, config=fork_snap.config)
```

`None` as the first argument means "don't inject new user input; just continue from the checkpoint." LangGraph loads the state stored at `fork_snap.config["configurable"]["checkpoint_id"]`, checks `fork_snap.next` for which node runs first, and executes forward from there.

This is **Replay**: same state, same path, same result. Useful for:
- Debugging a failed node without re-running the expensive LLM calls that already succeeded.
- Auditing: proving the system produces the same output given the same state.
- Restarting after a network failure without repeating earlier work.

---

### 3. Inject new state, then re-invoke — Time Travel

```python
# update_state writes a NEW checkpoint on top of fork_snap, returns the new config
travel_config = graph.update_state(
    fork_snap.config,            # base: the past checkpoint
    {"keywords": new_keywords},  # the override — applied with each channel's reducer
    as_node="understand",        # "as if understand wrote this" → next = ("search",)
)

result = graph.invoke(None, config=travel_config)
```

`update_state` does three things:

1. Takes the state stored at `fork_snap.config` as the base.
2. Merges `values` into it using each channel's reducer (default = overwrite; `operator.add` channels accumulate as usual).
3. Writes a **new checkpoint** to the thread and returns the config pointing to it.

The `as_node` argument tells LangGraph which node "made" the update. The graph routes to whatever comes after that node. Since `understand → search` is a fixed edge in our graph, `as_node="understand"` ensures `travel_config` has `next=("search",)` — so `invoke(None, ...)` picks up at `search`, not at `understand`.

**Use the returned config.** `update_state` returns the config pointing to the NEW checkpoint. Pass that returned config (not the original `fork_snap.config`) to your subsequent `invoke` or `stream`.

---

## Replay vs Time Travel

| | Replay | Time Travel |
|---|---|---|
| **State at fork point** | Exactly as stored | Modified by `update_state` |
| **Execution path** | Same as original | Can diverge |
| **Trigger** | `invoke(None, config=snap.config)` | `update_state(snap.config, vals, as_node=...)` → `invoke(None, config=returned_config)` |
| **Use cases** | Debugging, auditing, re-running failed steps | Experimentation, HITL, adaptive workflows |

---

## Four use cases

| Use case | What it means concretely |
|---|---|
| **Debugging** | A node raised an exception mid-run. Find the checkpoint just before that node, fix the node, replay. The expensive LLM calls before the failure don't re-run. |
| **Fault recovery** | Same as debugging, automated: the checkpointer already saved the state; just re-invoke the same thread after the issue is resolved. |
| **Human-in-the-loop** | After `evaluate` runs, a human reviews the verdict. If they override it, `update_state` injects their decision and the graph continues from that point. (Module 5.5 — Interrupts — builds this properly with a pause mechanism.) |
| **Experimentation** | "What if the planner had extracted different keywords?" Fork from the post-`understand` checkpoint, inject new keywords, re-run `search` forward. Compare two outcomes without re-running `understand`. |

---

## The fork model

Time travel doesn't rewrite history. The original checkpoints stay. `update_state` adds a new checkpoint on top of the fork point, and subsequent invocations add more checkpoints after it. The same thread now has two forward paths:

```
START → understand → search → summarize → evaluate → search → summarize → evaluate → END
                  ↑  (original path, checkpoints 0–8)
                  └─ [update_state: new keywords] → search → summarize → evaluate → END
                          (time-travel path, new checkpoints on same thread)
```

All checkpoints — original and forked — live on the same `thread_id`. The `checkpoint_id` at the fork point is the address that identifies where each branch starts. You can replay the original path and the time-travel path independently by passing the right `checkpoint_id`.

---

## Source errors corrected

### 1. `get_state_history` returns `StateSnapshot` objects, not tuples or dicts

The source's iteration code branches on `isinstance(entry, tuple)` vs `isinstance(entry, dict)`. Neither is correct. `get_state_history` yields `StateSnapshot` namedtuples with fields `.config`, `.values`, `.next`, `.metadata`, `.tasks`. Iterate directly:

```python
for snap in graph.get_state_history(config):
    cid = snap.config["configurable"]["checkpoint_id"]
    print(snap.values, snap.next, snap.metadata)
```

No `isinstance` check needed. The source's branching code would silently fall through both branches on every entry, printing nothing.

### 2. Nodes mutating state in place

The source's `greet_node` and `personalize_node` do `state["key"] = value; return state`. We corrected this in v0.3: nodes return **partial dicts**. Returning the full state triggers the reducer on every key, which doubles lists under `operator.add`. This debt is already paid in our project.

### 3. The source's "time travel" demo doesn't actually re-run nodes

The source calls `update_state(config, {"name": "User_1"})` on a **finished thread** (where `next=()`), then re-invokes with just `thread_id` and no `checkpoint_id`. The output shows `message: 'Hello, ABC!'` — meaning `personalize_node` did **not** re-run with the new name. The graph returned the latest snapshot unchanged, because `next=()` means "graph is done, nothing to run."

For nodes to re-run, you must fork from a checkpoint where `next` is non-empty — found via `get_state_history`, not by calling `update_state` on the finished latest snapshot. The source demo is calling `update_state` correctly but then re-invoking the wrong checkpoint.

---

## Project → v0.5

Only `main.py` changes. The graph, state, nodes, and corpus are all unchanged — time travel was already wired in at v0.4 via `build_graph(checkpointer=...)`. This is the payoff of teaching `compile(checkpointer=)` as the plug-in point: persistence, replay, and time travel all come from the one line added in 5.2. Nothing new to wire.

`run_v5_time_travel()` runs four steps on one thread:

1. The original run (2-attempt, 4 sources, ok) — identical to v0.4.
2. Print the full checkpoint history and identify the fork point.
3. Inject different keywords via `update_state` and re-run from the fork point.
4. Compare: original path vs time-travel path, different keywords → different docs.

```python
def run_v5_time_travel():
    print("=" * 60)
    print("v0.5 — Time Travel: fork from a past checkpoint")
    print("=" * 60)

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "travel-demo"}}

    # Step 1: original run — same 2-attempt result as v0.4
    print("--- original run ---")
    for step in graph.stream(
        {"question": QUESTION, "sources": [], "attempts": 0},
        config=config,
    ):
        for node_name, update in step.items():
            keys = ", ".join(f"{k}={_short(v)}" for k, v in update.items())
            print(f"  [{node_name}] -> {keys}")

    original_final = graph.get_state(config)
    history = list(graph.get_state_history(config))

    print(f"\n--- {len(history)} checkpoints in history ---")
    for snap in history:
        step_n = snap.metadata.get("step", "?")
        nxt = snap.next or ("__end__",)
        cid = snap.config["configurable"]["checkpoint_id"][:10]
        print(f"  step {step_n:>2} | next={nxt!s:<32} | id={cid}...")

    # Step 2: find the fork point — earliest checkpoint where next contains "search"
    # reversed(history) = oldest-first; first match = after 'understand', before first 'search'
    fork_snap = next(
        (s for s in reversed(history) if "search" in s.next),
        None,
    )
    if not fork_snap:
        print("No fork point found.")
        return

    print(f"\n--- fork point: step {fork_snap.metadata.get('step')} ---")
    print(f"  original keywords : {fork_snap.values.get('keywords')}")
    print(f"  next              : {fork_snap.next}")

    # Step 3: time travel — inject new keywords, re-run from fork point
    # Simulates the user clarifying "I meant checkpointing and state, not general architecture"
    new_keywords = ["checkpointing", "persistence", "state", "channels", "reducers"]
    print(f"\n--- time travel: injecting keywords {new_keywords} ---")

    # update_state writes a new checkpoint on top of fork_snap; returns its config
    travel_config = graph.update_state(
        fork_snap.config,
        {"keywords": new_keywords},
        as_node="understand",  # edges from 'understand' determine next → ("search",)
    )
    travel_final = graph.invoke(None, config=travel_config)

    # Step 4: compare outcomes — different keywords → different docs found
    print("\n--- comparison ---")
    orig_srcs = original_final.values.get("sources", [])
    trav_srcs = travel_final.get("sources", [])
    print(f"  original    : {len(orig_srcs)} source(s) | {original_final.values.get('verdict')}")
    print(f"  time-travel : {len(trav_srcs)} source(s) | {travel_final.get('verdict')}")
    print("\n  original sources:")
    for s in orig_srcs:
        print(f"    - {s['title']}")
    print("\n  time-travel sources:")
    for s in trav_srcs:
        print(f"    - {s['title']}")
```

**Expected comparison output:**

```
--- comparison ---
  original    : 4 source(s) | ok — 4 sources
  time-travel : 3 source(s) | weak — only 3 source(s), wanted 4

  original sources:
    - LangGraph Docs — Core Concepts
    - Pregel and Super-Steps
    - Channels and Reducers
    - Checkpointing Guide

  time-travel sources:
    - Channels and Reducers
    - Checkpointing Guide
    - LangGraph Docs — Core Concepts
```

The original run extracted `["langgraph", "architecture"]` from the question and found 4 docs across two attempts. The time-travel run started with `["checkpointing", "persistence", "state", "channels", "reducers"]` — a narrower, domain-specific query — found 3 docs and hit `MAX_ATTEMPTS` before reaching the threshold. Same graph, same code, different state at the fork point → different path, different outcome. The fork model in action.

---

## Complete updated `main.py`

```python
from langgraph.checkpoint.memory import InMemorySaver

from research_assistant import agent as v2
from research_assistant import controller_demo
from research_assistant.graph import build_graph

QUESTION = "Explain LangGraph architecture using 3 reliable sources."


def run_v5_time_travel():
    print("=" * 60)
    print("v0.5 — Time Travel: fork from a past checkpoint")
    print("=" * 60)

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "travel-demo"}}

    # Step 1: original run — same 2-attempt result as v0.4
    print("--- original run ---")
    for step in graph.stream(
        {"question": QUESTION, "sources": [], "attempts": 0},
        config=config,
    ):
        for node_name, update in step.items():
            keys = ", ".join(f"{k}={_short(v)}" for k, v in update.items())
            print(f"  [{node_name}] -> {keys}")

    original_final = graph.get_state(config)
    history = list(graph.get_state_history(config))

    print(f"\n--- {len(history)} checkpoints in history ---")
    for snap in history:
        step_n = snap.metadata.get("step", "?")
        nxt = snap.next or ("__end__",)
        cid = snap.config["configurable"]["checkpoint_id"][:10]
        print(f"  step {step_n:>2} | next={nxt!s:<32} | id={cid}...")

    # Step 2: find the fork point — earliest checkpoint where next contains "search"
    # reversed(history) = oldest-first; first match = after 'understand', before first 'search'
    fork_snap = next(
        (s for s in reversed(history) if "search" in s.next),
        None,
    )
    if not fork_snap:
        print("No fork point found.")
        return

    print(f"\n--- fork point: step {fork_snap.metadata.get('step')} ---")
    print(f"  original keywords : {fork_snap.values.get('keywords')}")
    print(f"  next              : {fork_snap.next}")

    # Step 3: time travel — inject new keywords, re-run from fork point
    new_keywords = ["checkpointing", "persistence", "state", "channels", "reducers"]
    print(f"\n--- time travel: injecting keywords {new_keywords} ---")

    travel_config = graph.update_state(
        fork_snap.config,
        {"keywords": new_keywords},
        as_node="understand",
    )
    travel_final = graph.invoke(None, config=travel_config)

    # Step 4: compare outcomes
    print("\n--- comparison ---")
    orig_srcs = original_final.values.get("sources", [])
    trav_srcs = travel_final.get("sources", [])
    print(f"  original    : {len(orig_srcs)} source(s) | {original_final.values.get('verdict')}")
    print(f"  time-travel : {len(trav_srcs)} source(s) | {travel_final.get('verdict')}")
    print("\n  original sources:")
    for s in orig_srcs:
        print(f"    - {s['title']}")
    print("\n  time-travel sources:")
    for s in trav_srcs:
        print(f"    - {s['title']}")


def run_v4():
    print("=" * 60)
    print("v0.4 — InMemorySaver + thread_id (persistence)")
    print("=" * 60)

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "research-001"}}

    print("--- streaming run ---")
    for step in graph.stream(
        {"question": QUESTION, "sources": [], "attempts": 0},
        config=config,
    ):
        for node, update in step.items():
            keys = ", ".join(f"{k}={_short(v)}" for k, v in update.items())
            print(f"  [{node}] -> {keys}")

    snapshot = graph.get_state(config)
    print("\n--- final snapshot ---")
    print("  verdict  :", snapshot.values.get("verdict"))
    print("  attempts :", snapshot.values.get("attempts"))
    print("  sources  :", len(snapshot.values.get("sources", [])))
    print("  next     :", snapshot.next)

    print("\n--- checkpoint history (newest first) ---")
    for snap in graph.get_state_history(config):
        step_n = snap.metadata.get("step", "?")
        nxt    = snap.next or ("__end__",)
        srcs   = len(snap.values.get("sources", []))
        att    = snap.values.get("attempts", 0)
        print(f"  step {step_n:>2} | next={nxt} | sources={srcs} | attempts={att}")


def run_v3():
    print("=" * 60)
    print("v0.3 — StateGraph with a conditional edge (the loop)")
    print("=" * 60)
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())

    print("--- step by step ---")
    for step in graph.stream({"question": QUESTION, "sources": [], "attempts": 0}):
        for node, update in step.items():
            keys = ", ".join(f"{k}={_short(v)}" for k, v in update.items())
            print(f"  [{node}] -> {keys}")

    final = graph.invoke({"question": QUESTION, "sources": [], "attempts": 0})
    print("\nKEYWORDS:", final["keywords"])
    print("ATTEMPTS:", final["attempts"])
    print("\nSOURCES:")
    for s in final["sources"]:
        print("  -", s["title"])
    print("\nSUMMARY:\n", final["summary"])
    print("\nVERDICT:", final["verdict"])


def _short(value, width: int = 60) -> str:
    text = str(value)
    return text if len(text) <= width else text[:width] + "…"


def run_v2():
    print("\n" + "=" * 60)
    print("v0.2 — Prebuilt create_react_agent (2-node ReAct loop + hooks)")
    print("=" * 60)
    answer = v2.run(QUESTION)
    print("\nANSWER:\n", answer)


def run_controller_demo():
    print("\n" + "=" * 60)
    print("Demo — the LLM picks the edge (controller_demo.py)")
    print("=" * 60)
    final = controller_demo.run()
    print("DECISION:", final["decision"])
    print("\n--- annotated messages ---")
    for msg in final["messages"]:
        who = msg.name or msg.type
        print(f"  {who:<11} | {_short(msg.content, 80)} | {msg.additional_kwargs}")


def main():
    run_v5_time_travel()
    run_v4()
    run_v3()
    run_v2()
    run_controller_demo()


if __name__ == "__main__":
    main()
```

---

## Self-check

1. What two things does `update_state` do before time travel can run? (Hint: base state + new checkpoint)
2. Why does `as_node="understand"` matter? What would happen if you omitted it or passed `as_node="search"`?
3. What's the difference between `invoke(None, config=snap.config)` (with `checkpoint_id`) and `invoke(None, config={"configurable": {"thread_id": "..."}})` (thread only, no `checkpoint_id`)?
4. Why did the source's time travel demo not re-run nodes even after calling `update_state`?
5. If a `search` node raised a network exception mid-run, which checkpoint would you fork from to retry just that node?
