# Module 4.3 — langgraph-supervisor

## Where We Left Off

Module 3 introduced four graph shapes that cover nearly every agentic use case:

| Shape | What it does | Example |
|---|---|---|
| **route** | Conditional edge picks one of several paths | Customer support triage |
| **loop** | Node sends work backward for another pass | Re-search on weak results |
| **delegate** | Central controller dispatches to specialist agents | **← this one** |
| **remember** | State accumulates across turns | Multi-session assistant |

v0.1 and v0.2 of the Research Assistant are loop shapes. The delegate shape has been named but not built. This sub-topic builds it.

---

## The Problem With One Agent and Many Tools

v0.2 uses `create_react_agent` with a `search_docs` tool. That works fine for one domain. Now imagine the Research Assistant needs to also book travel for the researcher, query a company database, and summarize Slack threads. You add more tools:

```python
agent = create_react_agent(
    model=llm,
    tools=[search_docs, book_flight, book_hotel, query_database, summarize_slack],
)
```

This breaks down in two ways:

1. **Tool selection degrades.** The more tools the LLM sees, the harder it is to pick the right one. At 20+ tools, accuracy drops noticeably.
2. **You can't scope tools to their domain.** A flight-booking tool and a research tool shouldn't share the same system prompt or retry logic. A flight booking that fails mid-transaction needs different handling than a failed search.

The fix is to give each domain its own agent, and put a routing LLM above them that decides which agent to call.

---

## The Supervisor Pattern

A supervisor system has two levels:

```
User → Supervisor (LLM) → Flight Agent (LLM + tools)
                        → Hotel Agent  (LLM + tools)
                        → Research Agent (LLM + tools)
```

The **supervisor** knows about the worker agents and decides which one to call for a given task. Each **worker agent** knows only about its own tools and its own system prompt. Workers don't know each other exists.

The mechanism underneath is identical to what you already know: **tool-based handoff**. The supervisor treats each worker agent as a callable tool. When it decides to delegate, it produces an `AIMessage` with `tool_calls` pointing to a worker by name. That worker runs, returns a result (as a `ToolMessage`), and the supervisor reads it before deciding what to do next.

```
Supervisor (LLM with agents as tools)
     │
     ▼ tool_call: "flight_assistant"
Flight Agent runs (its own ReAct loop)
     │
     ▼ ToolMessage with result
Supervisor reads result → decides to call Hotel Agent
     │
     ▼ tool_call: "hotel_assistant"
Hotel Agent runs
     │
     ▼ ToolMessage with result
Supervisor reads result → no more tool_calls → END
```

It's the same `tools_condition` loop from 4.2, one level up. The only new idea is that the "tools" being called are entire agents, not Python functions.

---

## What `create_supervisor` Builds

`langgraph-supervisor` is a separate package (install with `pip install langgraph-supervisor`). It provides `create_supervisor`, which encapsulates the supervisor graph in the same way `create_react_agent` encapsulates the ReAct loop.

The graph it builds:

```
START → supervisor (LLM with worker agents as tools)
supervisor → [agent_condition] → flight_agent  → supervisor
                               → hotel_agent   → supervisor
                               → END (supervisor decides task is done)
```

Each worker agent is a full `create_react_agent` graph running inside a node of the outer graph. The interface between layers is `MessagesState` — messages flow down to the worker and results flow back up as `ToolMessage` objects.

---

## The Code

Here is the source's flight + hotel booking example, structure-first:

```python
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# Step 1: Define tools — one per domain
def book_hotel(hotel_name: str) -> str:
    """Book a hotel by name."""
    return f"Hotel booked at {hotel_name}."

def book_flight(from_airport: str, to_airport: str) -> str:
    """Book a flight between two airports."""
    return f"Flight booked from {from_airport} to {to_airport}."

# Step 2: Create worker agents — each gets its own tools and prompt
flight_agent = create_react_agent(
    model=llm,
    tools=[book_flight],
    prompt="You are a flight booking assistant. Only book flights.",
    name="flight_assistant",   # ← supervisor uses this name to call it
)

hotel_agent = create_react_agent(
    model=llm,
    tools=[book_hotel],
    prompt="You are a hotel booking assistant. Only book hotels.",
    name="hotel_assistant",
)

# Step 3: Create the supervisor
supervisor = create_supervisor(
    agents=[flight_agent, hotel_agent],
    model=llm,
    prompt="You manage a flight and hotel booking assistant. Assign tasks accordingly.",
).compile()

# Step 4: Run
result = supervisor.invoke({
    "messages": [{
        "role": "user",
        "content": "Book a flight from BOS to JFK and a stay at McKittrick Hotel."
    }]
})
```

That one request triggers two delegations: the supervisor calls `flight_assistant`, gets a confirmation, then calls `hotel_assistant`, gets a confirmation, then synthesizes a final reply.

### Two roles of `prompt`

| Argument | On a worker agent | On the supervisor |
|---|---|---|
| `prompt` | Constrains scope ("you only book flights") | Describes routing strategy |
| Effect | Worker ignores unrelated requests | Supervisor knows which agent handles what |

### The `name` parameter

`name="flight_assistant"` on `create_react_agent` is how the supervisor discovers the worker. Under the hood, the supervisor's LLM sees a tool called `transfer_to_flight_assistant` and calls it when it needs a flight booked. If you don't set `name`, the supervisor has no stable identifier for that agent.

---

## When to Use Supervisor vs. Single Agent

| Scenario | Use |
|---|---|
| All tools are in one domain (e.g., all research tools) | Single `create_react_agent` |
| Tools need different system prompts / retry logic | Supervisor + separate worker agents |
| Task reliably decomposes into distinct sub-tasks | Supervisor |
| You need domain isolation (different teams own different agents) | Supervisor |
| You have >10 tools and tool selection is degrading | Supervisor with grouped workers |
| You're prototyping and want to move fast | Single agent first, promote to supervisor later |

The smell test: if you'd want different system prompts for different tools, you want a supervisor.

---

## Connection to the Graph Shape

From Module 3's use-case table, the delegate shape looked like this in pseudocode:

```
controller → decides → specialist_A
                     → specialist_B
                     → specialist_C
```

`create_supervisor` IS that shape, pre-wired. The supervisor node IS the conditional edge function — it reads the task and returns the name of the agent to activate next. Workers are subgraphs inside nodes. The pattern scales: you can nest supervisors (a supervisor whose workers are themselves supervisors), giving you a full hierarchy for large multi-agent systems.

---

## What's Next

With all three Module 4 sub-topics done (Hooks · langgraph-prebuilt · langgraph-supervisor), you've seen every prebuilt surface. Module 5 opens the custom StateGraph fully: the backward edge (turning v0.1's chain into a real agent), persistence, time travel, tools, interrupts, and memory. That's where v0.3+ of the Research Assistant is built.

---

## Self-Check

1. The supervisor calls worker agents using the same mechanism it uses to call regular tools. What mechanism is that, and what does it produce in the message history?
2. What does the `name` parameter on `create_react_agent` control?
3. You have an agent with 15 tools covering three domains (billing, shipping, returns). What's the signal that it's time to promote this to a supervisor architecture?
4. A worker agent runs its own ReAct loop. What does the supervisor see as the result of calling a worker — a raw Python return value, or something else?
5. Can a supervisor's worker be another supervisor? What use case does that enable?
