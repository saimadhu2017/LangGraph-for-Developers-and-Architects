# Module 5.4 — Tool

## Where we left off

Since Module 3, `corpus.py` has carried this comment at the top:

```python
"""A stand-in for a real search tool. Module 5's *Tool* sub-topic swaps this for a real retriever."""
```

The graph has been running on a keyword-overlap stub. That was intentional — tools belong in their own lesson. This lesson pays that debt.

But before we wire anything in, we need to understand what LangGraph means by a "tool" and why it treats tools differently from ordinary function calls inside a node.

---

## What a tool is in LangGraph

A **tool** is a function the agent can call to interact with the world outside its own reasoning. The boundary is explicit:

| Inside a node (reasoning) | Outside (tool) |
|---|---|
| Parsing input, deciding what to do | Searching documents |
| Formatting an LLM prompt | Querying a database |
| Reading from state | Calling an external API |
| Writing to state | Running code, doing math |

A plain Python function called inside a node is not a "tool" in LangGraph's sense — it's just code. A **tool** in LangGraph means three things:

1. It's wrapped with `@tool` (from `langchain_core.tools`), giving it a **name, description, and typed input schema** the LLM can read
2. It can be passed to an LLM via `.bind_tools([...])` so the LLM can request calls to it
3. It can be registered with `ToolNode`, which executes those LLM-requested calls automatically

You already used this in Module 4.2 — `agent.py`'s `search_docs` is a `@tool`. The prebuilt `create_react_agent` wired it through `ToolNode` for you. Today we learn to do that wiring explicitly in a custom `StateGraph`.

---

## The `@tool` decorator — three points for the graph context

You used this in the LangChain course. Three things matter specifically in a LangGraph context:

```python
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """Search the research corpus for documents matching the query.

    Returns formatted document titles and text.
    Use specific technical keywords for best results.
    """
    # ... implementation
    return formatted_string
```

**The docstring is the tool's description.** When this tool is bound to an LLM, the LLM reads the description to decide when and how to call it. "Use specific technical keywords for best results" is prompt engineering inside the tool definition — it shapes the LLM's call arguments without any extra prompt work.

**The type annotations are the input schema.** `ToolNode` validates inputs against them. If the LLM constructs a call with the wrong argument names or types, `ValidationNode` (Module 4.2) can catch that before execution.

**Return a string.** Tool results become `ToolMessage` content — a string the LLM reads in the next step. If you return a dict or list it gets stringified unpredictably. Return a formatted string the LLM can reason about.

---

## Two patterns for calling a tool in a custom graph

This is the core of the lesson. The choice comes down to one question: **does the LLM need to decide whether or which tool to call?**

### Pattern A — Direct call

The node calls the tool explicitly. No LLM involved in tool selection.

```python
def search(state: ResearchState) -> dict:
    results = search_docs.invoke({"query": " ".join(state["keywords"])})
    sources = _parse_results(results)
    return {"sources": sources, "attempts": 1}
```

The node knows exactly which tool to call and with what arguments. The tool runs as a deterministic step. The LLM plays no role in the decision.

**When to use:** Your workflow already knows which tool to call. There's one tool and you always want it. The Research Assistant's `search` node is this — `understand` computes the keywords, `search` always calls `search_docs`. Nothing to decide.

### Pattern B — LLM-driven via ToolNode

The LLM node calls the model with `.bind_tools([...])`. The LLM may or may not request a tool call. `ToolNode` reads the LLM's response and executes any tool calls it finds.

```python
llm_with_tools = model.bind_tools([search_docs, calculator, weather_api])

def agent_node(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

After this node, a router checks whether the LLM's response contains tool requests. If yes, `ToolNode` runs them and loops back. If no, the graph moves on.

**When to use:** You have multiple tools and the LLM needs to decide which to use, when to use them, and when to stop. This is the ReAct loop we saw inside `create_react_agent` — now built explicitly.

---

## `ToolNode` — what it does

`ToolNode` (from `langgraph.prebuilt`) is a prebuilt node that executes tool calls. It was running inside `create_react_agent` in Module 4.2. Here's exactly what it does on every invocation:

```
state["messages"][-1]          ← reads the last message (expected: AIMessage)
    .tool_calls                ← list of {"name": "...", "args": {...}, "id": "..."}
        ↓
    runs each tool by name, in parallel (Pregel parallel super-step)
        ↓
    ToolMessage(content=result, tool_call_id=id)   ← one per tool call
        ↓
return {"messages": [ToolMessage1, ToolMessage2, ...]}
```

Three things follow from this design:

**`ToolNode` requires `messages` in state.** It reads and writes the `messages` channel. Your state must have `messages: Annotated[list, add_messages]`. If you're using a custom `TypedDict` with no `messages` key, you either add one or use Pattern A instead.

**Parallel execution is built in.** If the LLM requests two tools in one response, both run in the same super-step. The checkpoint is written after both complete. You get parallelism for free.

**`handle_tool_errors=True` (default).** If a tool raises an exception, `ToolNode` catches it, formats it as a `ToolMessage` with the error text, and returns it. The LLM reads the error and can retry or adjust. This is what makes LLM-driven tool use recoverable without graph-level error handling.

---

## `tools_condition` — the router between LLM node and ToolNode

After an LLM node in Pattern B, you don't know in advance whether the LLM decided to call a tool or to finish. `tools_condition` is the prebuilt router for exactly this:

```python
from langgraph.prebuilt import tools_condition

builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "__end__": END},
)
```

`tools_condition` reads `state["messages"][-1]`. If that message has `.tool_calls`, it returns `"tools"`. Otherwise it returns `"__end__"`.

Two things to note:

`"__end__"` (double-underscore) is `tools_condition`'s vocabulary, not a node name — you translate it to `END` in the path_map. Same pattern as the `route_after_evaluate` router in the Research Assistant: the router speaks domain vocabulary, the path_map speaks node names.

This function is a pure router — it never writes to state. The same rule applies: do the work in a node, decide in the edge.

---

## The full Pattern B graph — built explicitly

Here's the complete LLM-driven tool loop as a custom `StateGraph`. This is what `create_react_agent` builds for you automatically:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm_with_tools = model.bind_tools([search_docs])

def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode([search_docs])

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "__end__": END},
)
builder.add_edge("tools", "agent")  # ← loop: after tools run, ask the LLM again

graph = builder.compile()
```

Compare this to Module 4.2's `create_react_agent` internals — they're the same graph. The difference now is that every seam is reachable: you can insert a node between `tools` and `agent`, replace `tools_condition` with your own router, add state keys beyond `messages`, or checkpoint between tool calls.

```
START → [agent] → tools_condition → [tools] → [agent] → ...
                       ↓
                      END
```

---

## Tool invocation lifecycle — mapped to LangGraph

The source describes a five-step lifecycle. Here's what each step is in actual LangGraph terms:

| Step | In LangGraph |
|---|---|
| Intent Recognition | LLM output: `AIMessage` with `tool_calls` populated |
| Tool Selection | `tool_calls[i]["name"]` — the LLM picks by name |
| Input Preparation | `tool_calls[i]["args"]` — LLM formats args to match the `@tool` schema |
| Execution | `ToolNode` dispatches by name, runs in parallel |
| Result Handling | `ToolMessage` appended to `messages`; LLM reads it in the next turn |

For Pattern A, steps 1–3 collapse into your node's code. You handle Intent Recognition (always call this tool), Tool Selection (hardcoded), and Input Preparation (you build the args). `tool.invoke(args)` handles Execution. You handle Result Handling in the same node.

---

## Three integration patterns (from the source) — concretely

The source names three patterns. Here's what they mean in code:

**Single Tool Node** — one node, one tool, always called. Pattern A. The Research Assistant's `search` node is this. Deterministic, predictable, no LLM in the tool-selection loop.

**Tool Router Node** — one LLM node that can call multiple tools. Pattern B. The "router" is `tools_condition` + the LLM's choice of which tool to name in `tool_calls`. You have multiple tools registered on the same `ToolNode`; the LLM picks one (or more) per step.

**Tool + Memory Node** — a tool node that also writes results to long-term memory (the `store` argument to `compile()`). Module 5.6 covers this. The mechanism is: after the tool runs, a node reads the `ToolMessage` and writes a summary to the cross-session store.

---

## Project v0.6 — formalizing the search tool

Three changes. The main graph (`graph.py`, `nodes.py`) is **unchanged** — it's Pattern A, which is correct for this use case.

### 1. New `research_assistant/tools.py`

The `@tool search_docs` definition moves here from `agent.py`. It now has a proper home:

```python
from langchain_core.tools import tool

from .corpus import search_corpus


@tool
def search_docs(query: str) -> str:
    """Search the research corpus for documents matching the query.

    Returns formatted document titles and text, or a 'no results' message.
    Use specific technical keywords for best results.
    """
    keywords = query.lower().split()
    results = search_corpus(keywords)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(f"[{r['title']}]\n{r['text']}" for r in results)
```

`corpus.py` stays — it's the implementation the tool wraps. What changes is the *interface*: `search_docs` is now a proper LangGraph tool with a name, description, and schema. It can be used in either pattern.

### 2. Updated `research_assistant/agent.py`

One import change — `search_docs` now comes from `tools.py`:

```python
from .tools import search_docs   # ← was defined inline here before
```

No behaviour change. Removes the duplicate definition.

### 3. New `research_assistant/tools_demo.py`

Shows Pattern B end-to-end — the explicit LLM-driven loop with `ToolNode` and `tools_condition`:

```python
"""Module 5.4 demo — Pattern B: LLM-driven tool use via ToolNode + tools_condition.

Builds the same loop that create_react_agent builds automatically,
but explicitly, so you can see every seam.
"""
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .tools import search_docs

load_dotenv()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class ToolDemoState(TypedDict):
    messages: Annotated[list, add_messages]


def _get_model():
    endpoint = HuggingFaceEndpoint(
        repo_id=MODEL,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        max_new_tokens=512,
        temperature=0.3,
    )
    return ChatHuggingFace(llm=endpoint)


def build_tool_demo_graph():
    llm_with_tools = _get_model().bind_tools([search_docs])

    def agent_node(state: ToolDemoState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode([search_docs])

    builder = StateGraph(ToolDemoState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile()


def run(question: str = "What are LangGraph's core architectural concepts?") -> dict:
    graph = build_tool_demo_graph()
    return graph.invoke({
        "messages": [HumanMessage(content=question)],
    })
```

### Updated `research_assistant/main.py`

Add `run_tools_demo()` and include it in `main()`:

```python
def run_tools_demo():
    print("=" * 60)
    print("v0.6 — Tool: Pattern B (LLM + bind_tools + ToolNode)")
    print("=" * 60)
    from research_assistant import tools_demo
    result = tools_demo.run()
    print("\n--- message thread ---")
    for msg in result["messages"]:
        who = getattr(msg, "name", None) or msg.__class__.__name__
        print(f"  {who:<20} | {_short(msg.content, 80)}")
```

---

## File layout after v0.6

```
research_assistant/
├── state.py              — unchanged
├── corpus.py             — unchanged (implementation detail)
├── tools.py              — NEW: @tool search_docs (the public tool interface)
├── nodes.py              — unchanged (Pattern A: calls search_corpus directly)
├── hooks.py              — unchanged
├── graph.py              — unchanged
├── agent.py              — updated: imports search_docs from tools.py
├── controller_demo.py    — unchanged
├── tools_demo.py         — NEW: Pattern B demo (LLM + ToolNode + tools_condition)
└── main.py               — updated: run_tools_demo() added
```

Why doesn't `nodes.py` change? Because Pattern A is the right choice for the Research Assistant's `search` node. The node always calls `search_docs`, always with the same kind of argument, and the query strategy (strict vs broad, deduplication) is workflow logic that belongs in the node — not something the LLM should decide. Pattern B would be the right choice if `search` needed to decide between `search_docs`, `web_search`, and `database_query` based on context. It doesn't.

---

## What's still not wired (intentional debts)

| Debt | Lesson |
|---|---|
| Nothing pauses for a human to approve citations | 5.5 — Interrupts |
| Nothing is remembered between questions | 5.6 — Memory |
| `tools_demo.py` uses the same corpus stub | Out of scope — real replacement would be FAISS/Chroma |

---

## Self-check

1. What three things does `@tool` give a function that a plain Python function doesn't have?
2. When should you use Pattern A (direct call) vs Pattern B (LLM-driven via ToolNode)?
3. `ToolNode` requires a specific key in your state `TypedDict`. What is it, and why?
4. `tools_condition` returns two string labels. What are they, and what does each mean?
5. In Pattern B, the graph has an edge from `tools` back to `agent`. Why does that edge exist?
6. Why does the Research Assistant's `search` node use Pattern A rather than Pattern B?
7. The source's "Tool Router Node" pattern — which LangGraph component is the actual "router"?
