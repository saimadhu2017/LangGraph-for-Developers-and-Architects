# Module 4.1 — Hooks

## Where We Left Off

Module 3 ended with v0.1 of the Research Assistant running end-to-end: `understand → search → summarize → evaluate`, a clean four-node linear graph. It also ended with an honest gap in the anatomy table:

| Agent Component | LangGraph Equivalent |
|---|---|
| LLM Core | Inside a node body |
| Prompt Template | Inside a node body |
| Memory Module | State channel + checkpointer |
| Toolset | Tool nodes |
| State | `TypedDict` passed to `StateGraph` |
| **Hooks & Transitions** | **Edges (transitions done) + Hooks ← owed** |

Transitions landed — that's `add_edge` and `add_conditional_edges`, fully covered in Modules 2 and 3. But *hooks* had no LangChain equivalent, and the table deferred them to Module 4. This sub-topic pays that debt.

---

## The Problem Without Hooks

Your `summarize` node calls an LLM. In production it misbehaves — slow, wrong, occasionally bizarre. You want to know:

- What question and sources did it receive?
- What did the model actually return?
- How long did it take?
- Was the result worth the API call?

Without hooks, your options are:

1. **Scatter `print()` statements inside node functions.** Works, but observability code is now tangled with business logic. Turning it off in prod means editing every node.
2. **Use LangSmith.** Works, but requires account setup, has per-trace cost at scale, and is remote — debugging locally means waiting for the trace to appear in a browser tab.

Hooks give you a third option: **structured intercept points that fire around a node, in-process, in plain Python, with zero external dependencies.**

---

## What a Hook Is

A hook is a plain function with one contract:

```python
def my_hook(state: MyState) -> MyState:
    # read, log, or modify state
    return state  # always return state
```

LangGraph defines two firing points:

| Hook | When it fires | What you can do |
|---|---|---|
| `pre_model_hook` | Immediately **before** the node (LLM/tool) executes | Read/log the input; inject extra context; stamp a timestamp; validate and abort early |
| `post_model_hook` | Immediately **after** the node executes | Read/log the output; trigger a side effect (alert, metric); note errors |

The execution order:

```
Input state
      │
      ▼
 [pre_model_hook]   ◄── intercept: see/modify what enters the node
      │
      ▼
  Node executes
  (LLM call, tool call, or any logic)
      │
      ▼
 [post_model_hook]  ◄── intercept: see what left the node, trigger side effects
      │
      ▼
Output state
```

The key word is **separate**. A hook is attached to a node from the *outside*. The node function has no idea it exists. That means you can:

- Add logging to any node without touching its code
- Use different hooks per environment (verbose in dev, quiet in prod)
- Stack multiple hooks in sequence on the same node

---

## pre_model_hook

Runs before the LLM or tool is invoked. Receives the current state. Returns the state (modified or not) — and that returned state is what the node actually sees.

```python
def pre_summarize(state: ResearchState) -> ResearchState:
    print(f"\n[PRE]  question : '{state['question']}'")
    print(f"[PRE]  keywords : {state.get('keywords', [])}")
    print(f"[PRE]  sources  : {len(state.get('sources', []))} doc(s)")
    return state
```

Because the return value is used as the node's input, the pre-hook can actually *modify* what the node sees — not just observe. Useful patterns:

- **Logging** — record what's about to go to the LLM
- **Enrichment** — stamp a `started_at` timestamp, inject a request ID
- **Validation** — check the input and return early or set an error flag
- **Context injection** — append extra system instructions before the prompt is built

---

## post_model_hook

Runs after the node has executed. In the **wrapper pattern** (explained next), it receives the merged state after the node's updates have been applied — so it can see both what was there before and what the node added.

```python
def post_summarize(state: ResearchState) -> ResearchState:
    summary = state.get("summary", "")
    print(f"[POST] summary  : {len(summary)} chars")
    return state
```

Typical uses:

- **Logging** — record what the LLM returned
- **Alerting** — if the response looks wrong, fire a Slack/PagerDuty notification
- **Metrics** — increment a token-cost counter, write a latency record
- **Sanitization** — strip PII before it reaches the next node (modify state and return it)

---

## Wiring Hooks to a Node

There are two ways, depending on whether you're on a custom `StateGraph` or a prebuilt agent.

### Pattern 1 — Manual wrapper (for custom `StateGraph`)

Write a wrapper function that sequences `pre → node → post`, then register the *wrapper* as the node:

```python
def wrap_with_hooks(node_fn, pre=None, post=None):
    def wrapped(state):
        if pre:
            state = pre(state)          # pre-hook runs; its return is what the node sees
        result = node_fn(state)         # the real node logic
        if post:
            post({**state, **result})   # post-hook sees the merged state (for observation)
        return result                   # return only the partial update LangGraph expects
    return wrapped
```

Two details worth pinning:

1. **`node_fn` returns a partial update dict**, not full state. `summarize` returns `{"summary": "..."}`, not the whole `ResearchState`. So before calling the post-hook you merge `{**state, **result}` so it can see the new values. The wrapper still returns `result` so LangGraph's reducer logic runs unchanged.

2. **In this pattern, modifications inside `post_model_hook` do not propagate.** The wrapper returns `result` (computed before post-hook ran), so any changes post-hook makes to state are local to it — they're for observation and side effects only. If you need post-hook to mutate what reaches the next node, see the `create_react_agent` pattern below.

Register it:

```python
builder.add_node(
    "summarize",
    wrap_with_hooks(summarize, pre=pre_summarize, post=post_summarize)
)
```

`summarize` is unchanged. The hook is entirely outside it.

### Pattern 2 — `create_react_agent` API (for prebuilt agents)

When you use the prebuilt agent (next sub-topic), hooks are keyword arguments:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[search_tool],
    pre_model_hook=pre_model_hook,
    post_model_hook=post_model_hook,
)
```

LangGraph calls them for you around every LLM invocation in the agent's reasoning loop. No wrapper needed. In this API, `post_model_hook` *can* return a partial update dict that propagates to state — the prebuilt runtime handles the merge. This is covered properly in the next sub-topic.

### A note on the source

The source material shows `graph.add_pre_model_hook(...)` and `builder.add_post_model_hook(...)` being called on a `StateGraph`. That method does not exist on `StateGraph` — the hook kwargs belong to `create_react_agent`. The source's actual running code (the `llm_node_with_hooks` wrapper) uses Pattern 1 correctly. Follow the working code.

---

## Hooks vs. Express Middleware

If you've written Node.js, this pattern will look familiar:

```
Express:    req  → [middleware]  → handler  → [middleware] → res
LangGraph: state → [pre-hook]   → node     → [post-hook]  → state
```

The difference is what flows through. Express middleware sees HTTP request/response objects. LangGraph hooks see the typed state dict the whole graph shares — `state["sources"]`, `state["verdict"]`, everything every node has produced so far. That makes hooks much richer: a pre-hook can read the full conversation history before deciding whether to call the LLM at all.

---

## Project: Adding Hooks to the Research Assistant

We're targeting the `summarize` node — the only node that calls an LLM, so the only one where latency and output quality are genuinely opaque. The other three nodes (`understand`, `search`, `evaluate`) are pure Python; debugging them with a stack trace is enough.

One new file, one changed file. `nodes.py` is untouched — that's the point.

### New file: `research_assistant/hooks.py`

```python
from .state import ResearchState


def pre_summarize(state: ResearchState) -> ResearchState:
    print(f"\n[PRE-HOOK]  question : '{state['question']}'")
    print(f"[PRE-HOOK]  keywords : {state.get('keywords', [])}")
    print(f"[PRE-HOOK]  sources  : {len(state.get('sources', []))} doc(s)")
    return state


def post_summarize(state: ResearchState) -> ResearchState:
    summary = state.get("summary", "")
    print(f"[POST-HOOK] summary  : {len(summary)} chars generated")
    return state


def wrap_with_hooks(node_fn, pre=None, post=None):
    """Returns a new node function that calls pre → node_fn → post.

    The pre-hook's return is used as the node's input (so it can modify state).
    The post-hook receives the merged state for observation; its return is ignored
    in this pattern, so use it for logging and side effects only.
    """
    def wrapped(state: ResearchState) -> dict:
        if pre:
            state = pre(state)
        result = node_fn(state)
        if post:
            post({**state, **result})
        return result
    return wrapped
```

### Modified: `research_assistant/graph.py`

```python
from langgraph.graph import END, START, StateGraph

from .hooks import post_summarize, pre_summarize, wrap_with_hooks
from .nodes import evaluate, search, summarize, understand
from .state import ResearchState


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("understand", understand)
    builder.add_node("search", search)
    builder.add_node("summarize", wrap_with_hooks(summarize, pre=pre_summarize, post=post_summarize))
    builder.add_node("evaluate", evaluate)

    builder.add_edge(START, "understand")
    builder.add_edge("understand", "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "evaluate")
    builder.add_edge("evaluate", END)

    return builder.compile()
```

### What you'll see when you run it

```
$ python -m research_assistant.main

graph TD;
    __start__ --> understand;
    understand --> search;
    search --> summarize;
    summarize --> evaluate;
    evaluate --> __end__;

[PRE-HOOK]  question : 'Explain LangGraph architecture using 3 reliable sources.'
[PRE-HOOK]  keywords : ['langgraph', 'architecture', 'reliable', 'sources']
[PRE-HOOK]  sources  : 3 doc(s)

... LLM call happens here ...

[POST-HOOK] summary  : 438 chars generated

KEYWORDS: ['langgraph', 'architecture', 'reliable', 'sources']

SOURCES:
  - LangGraph Docs — Core Concepts
  - Pregel and Super-Steps
  - Channels and Reducers

SUMMARY:
 LangGraph models an application as a graph...

VERDICT: ok — 3 sources
```

The hook lines bracket the LLM call. Everything else is exactly as it was in v0.1 — same graph, same nodes, same output format.

---

## Self-Check

1. What is the function signature of a LangGraph hook?
2. A `pre_model_hook` receives the state *before* the node runs and returns state. Can that returned state differ from what was passed in? What does that mean for the node?
3. In the `wrap_with_hooks` implementation above, the post-hook receives `{**state, **result}` instead of `state`. Why?
4. Why does `wrapped` return `result` instead of the post-hook's return value? What's the consequence for post-hook modifications?
5. In Pattern 2 (`create_react_agent`), post-hook modifications *can* propagate. What must the post-hook return for that to happen?
6. Name two things a pre-hook can do that you could not do cleanly by putting the same code inside the node function.
