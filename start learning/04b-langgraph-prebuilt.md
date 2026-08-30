# Module 4.2 — langgraph-prebuilt

## Where We Left Off

v0.1+hooks runs end-to-end. We hand-wrote every piece: the `TypedDict` state, four node functions, every `add_edge`, and then in 4.1 we bolted hooks around `summarize` with a manual wrapper. You also used `create_react_agent` in the LangChain course — called it, passed in some tools, and got answers back, but had no idea what was inside.

This sub-topic opens that box.

---

## The Prebuilt Ecosystem

`langgraph-prebuilt` ships several pre-wired agent templates. The one we care about today is `create_react_agent`. The others are worth naming so you recognize them in documentation:

| Package / Class | What it does |
|---|---|
| `create_react_agent` | ReAct tool-calling agent — the main prebuilt |
| `ToolNode` | Prebuilt node that runs tool calls from an `AIMessage` |
| `tools_condition` | Prebuilt router: tool calls present → tools, else → END |
| `ValidationNode` | Validates tool-call arguments against Pydantic schemas before execution |
| `HumanInterrupt` / `HumanResponse` | Pause + resume for human-in-the-loop approval |
| `langgraph-supervisor` | Hierarchical multi-agent orchestration (next sub-topic) |
| `Trustcall` | Structured extraction with high reliability |
| `LangMem` | Long-term memory across sessions |
| `LangGraph Swarm` | Decentralized multi-agent collaboration |

Today's focus: `create_react_agent`, `ToolNode`, `tools_condition`, `ValidationNode`, and the hook API.

---

## What create_react_agent Actually Builds

The ReAct pattern — Reason → Act → Observe → Loop — maps directly onto a two-node graph. Before looking at `create_react_agent`, here is that graph written by hand:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Bind tools to the model so it can emit tool_calls
model_with_tools = llm.bind_tools(tools)

# Node 1: the LLM. Decides whether to call a tool or answer directly.
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Node 2: the tool executor. Reads tool_calls from the last AIMessage, runs them.
tool_node = ToolNode(tools)

graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # prebuilt router
graph.add_edge("tools", "agent")                       # always loop back

agent = graph.compile()
```

The graph shape:

```
START
  │
  ▼
agent  ──[tool_calls present?]──▶  tools
  ▲                                  │
  └──────────────────────────────────┘  (always)
  │
  ▼ (no tool_calls)
 END
```

`create_react_agent` builds exactly this and returns the compiled graph:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=tools)
```

That one call replaces the 15 lines above. The same nodes, the same edges, the same loop. The difference is that you can no longer reach into the graph to add a node between `agent` and `tools`, or change what `tools_condition` routes on. The loop is sealed.

---

## MessagesState

The manual graph above uses `MessagesState`. `create_react_agent` uses it too. Here is what it is:

```python
from langgraph.graph import MessagesState

# Equivalent to:
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

Everything flows through the `messages` list: the user's question as a `HumanMessage`, the model's reasoning and tool requests as `AIMessage` objects, tool results as `ToolMessage` objects, and the final answer as another `AIMessage`. The `add_messages` reducer appends (and replaces on matching IDs, which enables time travel later).

This is different from our `ResearchState`, which has five named keys (`question`, `keywords`, `sources`, `summary`, `verdict`). Both are valid. The structured state is better when you want precise control over what each node reads and writes. The message-based state is better for conversational agents where the whole history is the context.

---

## ToolNode

`ToolNode` is the prebuilt executor for tool calls. It does what you would otherwise write yourself:

1. Reads `tool_calls` from the last `AIMessage` in state
2. Finds each tool by name
3. Runs them — in parallel if there are multiple calls
4. Wraps each result in a `ToolMessage` (with the matching `tool_call_id`)
5. Returns `{"messages": [ToolMessage, ...]}`

Usage:

```python
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools=[search_docs, get_weather], handle_tool_errors=True)
```

`handle_tool_errors=True` (the default) catches exceptions from tools and returns them as `ToolMessage` content so the LLM can reason about them rather than crashing the graph.

`tools_condition` is the prebuilt router that pairs with `ToolNode`. It reads the last message: if it has `.tool_calls`, route to `"tools"`; otherwise route to `END`. It is what you pass to `add_conditional_edges` in the manual version — and what `create_react_agent` wires for you.

---

## The Hook API on create_react_agent

In Module 4.1 we built `wrap_with_hooks`, a wrapper function that sequenced `pre → node → post` and returned the node's partial update dict. The `create_react_agent` hook API is different in one key way.

**Module 4.1 wrapper pattern:** hook returns the full state (because it is composing inside a plain Python function).

**create_react_agent hook API:** hook returns `None` or a partial update dict. LangGraph handles the merge.

```python
def pre_hook(state: dict) -> dict | None:
    messages = state.get("messages", [])
    print(f"[PRE]  {len(messages)} message(s) in state")
    return None            # None = no state change

def post_hook(state: dict) -> dict | None:
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        print(f"[POST] {type(last).__name__} added")
    return None

agent = create_react_agent(
    model=llm,
    tools=[search_docs],
    pre_model_hook=pre_hook,
    post_model_hook=post_hook,
)
```

Returning `None` is cleaner than returning the whole state object unchanged — it signals "pass through" explicitly. If you need to modify state before the model sees it, return a partial update dict:

```python
def pre_hook(state: dict) -> dict | None:
    # Inject a system message if none present
    messages = state.get("messages", [])
    if not any(m.type == "system" for m in messages):
        from langchain_core.messages import SystemMessage
        return {"messages": [SystemMessage(content="Be concise.")]}
    return None
```

The hooks fire on every iteration of the ReAct loop — before and after each LLM call. If the agent makes three tool calls, the hooks run three times each.

---

## ValidationNode

LLMs sometimes produce tool arguments that do not match the expected types. A user says "book a meeting for one hour" — the LLM emits `duration_minutes: "60"` (a string) instead of `duration_minutes: 60` (an integer). The tool schema requires `int`. The execution fails.

`ValidationNode` catches this before the tool runs. It sits between the agent node and the tool node and validates `tool_calls` arguments against Pydantic schemas. On failure, instead of crashing, it returns a `ToolMessage` containing the validation error. The agent node reads that error message on the next step and can correct its arguments and retry.

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, ValidationNode

class ScheduleInput(BaseModel):
    title: str
    duration_minutes: int = Field(description="Must be a whole number, e.g. 60")
    attendees: list[str]

@tool(args_schema=ScheduleInput)
def schedule_meeting(title: str, duration_minutes: int, attendees: list[str]) -> str:
    """Schedule a meeting in the calendar."""
    return f"Scheduled '{title}' for {duration_minutes} minutes."

tools = [schedule_meeting]
tool_node = ToolNode(tools)
validation_node = ValidationNode([ScheduleInput])
```

The graph shape with validation:

```
                          ┌─────────────────────────────────┐
                          │  fail: error as ToolMessage      │
                          ▼                                  │
START → agent → validation → tool_node → agent → END
              ↗ pass: AIMessage forwarded
```

If validation passes, `ValidationNode` forwards the original `AIMessage` to `tool_node`. If it fails, it injects a `ToolMessage` with the schema error back into the message history, and the router sends the agent back to re-plan with that error in context.

---

## HumanInterrupt

`HumanInterrupt` is a prebuilt object from `langgraph.prebuilt` that pauses graph execution at a node boundary and sends a structured request to a human operator before resuming. It requires a checkpointer — the graph needs to persist its state while waiting. A typical use case: an agent drafts an email, reaches a `human_review` node, emits a `HumanInterrupt` with the draft, and waits. A human reads it in a UI or terminal, constructs a `HumanResponse` (approved, edited, or rejected), and the graph resumes from the checkpoint with the human's decision in state.

This is what Module 5's Interrupts sub-topic covers in full. For now: the mechanism exists, it is in `langgraph-prebuilt`, and it requires a checkpointer that you have not added yet.

---

## The Trade-off: v0.1 vs. v0.2 Side-by-Side

This is the payoff of running both versions.

| | v0.1 — Custom StateGraph | v0.2 — create_react_agent |
|---|---|---|
| State schema | Custom `TypedDict` (5 keys) | `MessagesState` (messages list) |
| Nodes | 4 explicit (`understand` / `search` / `summarize` / `evaluate`) | 2 auto-wired (`agent` / `tools`) |
| Control flow | Every edge declared by you | ReAct loop pre-wired |
| Loop | Not yet (v0.3 adds it) | Built in |
| Hooks | `wrap_with_hooks` wrapper | `pre_model_hook` / `post_model_hook` kwargs |
| What you control | Everything | Model, tools, hooks, system prompt |
| What you cannot do without rebuilding | — | Insert approval gate between specific steps · cap retries per tool · swap models mid-loop · route on custom state keys |

The last row is the point. `create_react_agent` is faster to wire up, but the loop is inside a sealed box. Our v0.1 `StateGraph` IS the loop — every edge, every conditional, every node is reachable and replaceable. When you need the approval gate in Module 5, you add it to v0.1. You cannot add it to v0.2 without reconstructing the graph from scratch — at which point you are back to v0.1.

---

## vs. LangChain's create_tool_calling_agent

You used LangChain's `create_tool_calling_agent` (via `initialize_agent` with `"chat-conversational-react-description"`) in the previous course. Here is the meaningful comparison:

| Feature | `create_react_agent` (LangGraph) | `create_tool_calling_agent` (LangChain) |
|---|---|---|
| Framework | Graph-based — a real `StateGraph` | Chain-based — an `AgentExecutor` |
| Tool use strategy | ReAct: Thought → Act → Observe → loop | Direct function call, structured output |
| State | Explicit `MessagesState` channel with reducer | Implicit, managed by `AgentExecutor` |
| Embeddable in a larger graph | Yes — it is a compiled graph, can be a node | No — chain, not a node |
| Customization | Hooks, subgraphs, conditional edges, checkpoints | Prompt templates, tool schemas |
| Multi-step reasoning | Native loop | Possible but harder to inspect |
| Best for | Multi-step reasoning, graph workflows, supervisor patterns | Single-step tool invocation, chatbots |

Rule of thumb: if your agent needs to loop — reason about a tool result, try again, compare multiple results — use `create_react_agent`. If you need a single structured tool call with validated output and no loop, `create_tool_calling_agent` is simpler.

---

## Project: v0.2 Research Assistant

Same question, two approaches running side by side. The prebuilt version uses `create_react_agent` with `search_docs` as a tool — the LLM decides when and how many times to call it. The v0.1 custom graph calls each step explicitly.

### New file: `research_assistant/agent.py`

```python
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import create_react_agent

from .corpus import search_corpus

load_dotenv()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _get_model():
    endpoint = HuggingFaceEndpoint(
        repo_id=MODEL,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        max_new_tokens=512,
        temperature=0.3,
    )
    return ChatHuggingFace(llm=endpoint)


@tool
def search_docs(query: str) -> str:
    """Search the research corpus for documents matching the query."""
    keywords = query.lower().split()
    results = search_corpus(keywords)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(f"[{r['title']}]\n{r['text']}" for r in results)


def _pre_hook(state: dict) -> dict | None:
    messages = state.get("messages", [])
    print(f"\n[PRE-HOOK]  agent step — {len(messages)} message(s) in state")
    return None


def _post_hook(state: dict) -> dict | None:
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "")
        print(f"[POST-HOOK] {type(last).__name__} — {len(str(content))} chars")
    return None


def build_agent():
    return create_react_agent(
        model=_get_model(),
        tools=[search_docs],
        pre_model_hook=_pre_hook,
        post_model_hook=_post_hook,
    )


def run(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Research the following question and provide a detailed answer with citations:\n\n"
            + question
        ))]
    })
    return result["messages"][-1].content
```

### Updated: `research_assistant/main.py`

```python
from research_assistant import agent as v2
from research_assistant.graph import build_graph

QUESTION = "Explain LangGraph architecture using 3 reliable sources."


def run_v1():
    print("=" * 60)
    print("v0.1 — Custom StateGraph (4 explicit nodes + hooks)")
    print("=" * 60)
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())
    final = graph.invoke({"question": QUESTION, "sources": []})
    print("\nKEYWORDS:", final["keywords"])
    print("\nSOURCES:")
    for s in final["sources"]:
        print("  -", s["title"])
    print("\nSUMMARY:\n", final["summary"])
    print("\nVERDICT:", final["verdict"])


def run_v2():
    print("\n" + "=" * 60)
    print("v0.2 — Prebuilt create_react_agent (2-node ReAct loop + hooks)")
    print("=" * 60)
    answer = v2.run(QUESTION)
    print("\nANSWER:\n", answer)


def main():
    run_v1()
    run_v2()


if __name__ == "__main__":
    main()
```

### What you will see

v0.1 prints the Mermaid graph, then the structured output (keywords, sources, summary, verdict). v0.2 prints hook lines around each LLM step as the agent reasons, then the final answer.

The same question, answered two ways. v0.1 uses explicit steps the code defined. v0.2 uses the ReAct loop the model navigates — it may call `search_docs` once, or multiple times, or not at all if it thinks it already knows. The outputs will differ.

---

## Self-Check

1. `create_react_agent` builds a two-node graph. Name the two nodes and describe what each one does.
2. `tools_condition` is the prebuilt router. What does it check, and what are the two possible destinations it routes to?
3. In Module 4.1, `wrap_with_hooks` returned the full state. In `create_react_agent`, a hook returns `None` or a partial dict. Why is returning `None` cleaner than returning the full state?
4. If the LLM emits an `AIMessage` with three tool calls in one step, what does `ToolNode` do with them?
5. An LLM says `duration_minutes: "sixty"` when the schema requires `int`. What does `ValidationNode` do, and what happens next?
6. Name one thing v0.1 can do that v0.2 cannot do without reconstructing the graph.
