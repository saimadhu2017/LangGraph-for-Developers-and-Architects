# Course Context — Rolling Snapshot

> Single source of truth for the tutor. Read this **before** writing any lesson.
> Two jobs: (1) the running log of every concept already covered, so nothing gets re-taught;
> (2) the latest full project source, so each new lesson can extend it without guessing.

## Carried Over From "Generative AI Essentials With LangChain" (assume known)
LCEL + `|` pipe operator · Runnables (`RunnableLambda`, `RunnableMap`) · `ChatPromptTemplate`, `PromptTemplate`, `MessagesPlaceholder`, `FewShotChatMessagePromptTemplate` · `StrOutputParser` · `ChatHuggingFace` + `HuggingFaceEndpoint` model setup · chat memory (`InMemoryChatMessageHistory`, `RunnableWithMessageHistory`, `trim_messages`) · document loaders → `RecursiveCharacterTextSplitter` → `HuggingFaceEmbeddings` → FAISS/Chroma → similarity search (RAG) · guardrails (regex jailbreak patterns, toxicity classifier, Pydantic output validation) · `@tool` decorator · `create_react_agent` from `langgraph.prebuilt` used as a **black box**.

**Where this course picks up:** open that black box. Explicit state, nodes, edges, cycles, checkpoints.

## Concepts Covered
- Module 1 > The Rise of Agentic Workflows and the Emergence of LangGraph — *all 4 sub-topics collapsed into one lesson, see "Teaching Style" note below*. **Core idea: control flow = who decides what runs next.** Three answers: you-in-code / you-in-a-chain / the-LLM-at-runtime. Chain = same sequence every invocation (predictable but rigid; Express-middleware analogy). Agent ≈ **control flow defined by the LLM** — path differs per invocation; buys unpredictability, hard debugging, no guardrails. **Agency↔reliability tradeoff curve**: x-axis = control (Code → LLM Call/Chain → Fully Autonomous), reliability line falls as agency line rises; enterprises need both. **LangGraph's one real innovation = partial control flow definition**: developer defines critical steps (reliability), LLM decides non-critical transitions (flexibility) — "bends the reliability curve" upward mid-spectrum. Six mechanisms that follow, each mapped to a concrete chain failure: stateful execution (multi-turn memory), conditional routing (edges pick next node from state), human-in-the-loop/interrupts (pause + resume), subgraphs (modular nested graphs), checkpointing (save/restore state), streaming. Graphs can **loop back**, pipelines structurally cannot — retry without restarting. LangGraph vs LangChain: directed graph vs linear chains, explicit+persistent state vs implicit, loops/conditionals/retries vs basic branching, graph visualization vs basic debugging — LangGraph orchestrates LangChain components, doesn't replace them. vs CrewAI/AutoGen/Haystack: differentiator is graph orchestration + native LangChain integration (table noted as low-value/stale-prone). Mental models: state machine whose transition function can be an LLM call (XState), CI/CD with conditional stages, executable flowchart.

- Module 2a > LangGraph in the LangChain Ecosystem — **LangGraph is standalone, not a LangChain plugin**: a node is just `state -> partial update`, no LangChain import required. But orchestration is *all* LangGraph does (no prompts/parsers/loaders/embeddings/retrievers/`@tool`), so in practice you use both: **LangChain = what happens inside a node, LangGraph = which node runs next**. An LCEL chain becomes the *body of a node* — and now that node can loop, branch, pause, resume. Three-tool layering: LangChain gets you to a demo → LangGraph to a system → LangSmith keeps you honest (LangSmith observes the layers below; has no control-flow model of its own). Full comparison grid (purpose / control flow / flexibility / state / HITL / streaming / loops / debuggability / MIT / best fit). Two rows singled out as *category* differences, not feature differences: **state** (LangChain memory = bolted-on, prompt-scoped via `RunnableWithMessageHistory`+`MessagesPlaceholder`; LangGraph state = the thing the graph is built around, holds retry counters/docs/scores/approval flags, and routing reads from it) and **loops** (`A | B | C` composes forwards only — C structurally cannot send work back to B; a LangGraph edge can point backwards, carrying forward what attempt 1 learned). FAQ answered properly, with two source claims corrected: **perf** — graph overhead is dict-merge/dispatch microseconds vs. hundreds of ms of LLM latency, so it's rounding error; the *real* cost is checkpointer I/O per super-step (deliberate trade, buys resumability + time travel); **deployment** — "requires LangSmith" is marketing shorthand, LangGraph is a Python library deployable in FastAPI/Lambda/Celery; **LangGraph Platform** (sold alongside LangSmith) sells managed persistence + task queues + scaling + UI, self-hosting fully supported. Differentiator vs other agent frameworks = nothing is hidden (control flow *is* your code, so there's a seam to reach into).
- Module 2b > LangGraph Architecture — **running example: Anita (Enterprise Solutions Architect) + Karthik (Sr AI Engineer)** rebuilding a rigid SaaS support chatbot; their 4 requirements (route by intent · pluggable tools · pause for human · inspect state evolution) are mapped onto the architecture at lesson end. Correct mental model = **state machine** (nodes = work, edges = transitions, state = carried object) where a transition function may be an LLM call; XState analogy. **Why graphs, 5 reasons:** (1) agentic workflows are already graph-shaped — `Think→Act→Observe→Reflect→Decide→Continue|Stop` is a *loop with branches*; hand-rolling it around a pipeline = writing a graph engine badly inside app code; (2) control flow becomes visible — order/branching/loops/**termination logic** explicitly declared; (3) **reliability + recoverability — the reason that actually justifies the architecture**: runtime always knows *where* it is, *what node* caused the transition, *what state* looked like — those three facts ARE a checkpoint, which buys resume-from-failure, replay, HITL, and "deterministic transitions despite non-deterministic LLMs" (randomness **contained inside nodes**, skeleton stays fixed/auditable); (4) modular/reusable — nodes as blocks, nested subgraphs, independent team ownership, microservices trade where the interface is the state schema; (5) mirrors real cognition (observe→reflect→plan→act loops). **Three layers — you only ever write in layer A:** **(A) Graph Definition** = `StateGraph`: `add_node`/`add_edge`/`add_conditional_edges`/`START`/`END`/`compile()`; nodes return **partial updates** not new state; edges declared *separately* from nodes (unlike LCEL's `a | b` which fuses them) which is what permits multiple in-edges and backward edges; **a conditional edge is a router fn reading state and returning the next node's NAME — this is where Module 1's "partial control flow definition" physically lives**; `START`/`END` are real sentinel nodes so termination is explicit. Also: **two front-ends to the same engine** — Graph API (`StateGraph`, structure-as-data, visualisable/nestable) vs **Functional API** (`@entrypoint`/`@task` from `langgraph.func`, ordinary Python control flow, still gets checkpointing/streaming/interrupts); rule of thumb = Graph API when structure is the point, Functional when making existing procedural code durable. **(B) Execution = Pregel engine** (Google's graph-parallel model, BSP = Bulk Synchronous Parallel); unit of work = **super-step**, three phases **PLAN → EXECUTE (all activated nodes in parallel, each reading the same start state, nobody sees anybody's writes) → UPDATE at the BARRIER (updates land together, reducers merge, checkpoint written, next nodes activated)**. Five features fall out of that one design choice: parallel execution by default · reducers become *necessary* · checkpoint granularity = one per super-step (barrier is the only consistent moment) · interrupts (barrier is safe to persist and walk away from) · deterministic replay. Retries/error handling live here too. Worked example: `[search_docs] ‖ [fetch_account]` is ONE super-step, wall-clock = slower of the two, `answer` sees both merged with zero coordination code. **(C) State Management** = **channels** (each state key is an independently-tracked slot; nodes communicate *only* through channels, no direct node-to-node calls — that indirection is what makes nodes swappable) + **reducers** (merge rule per channel; **default is overwrite/last-write-wins**). `Annotated[list, add_messages]`, `Annotated[list, operator.add]`, `Annotated[int, operator.add]`. **"Agent lost its memory" bugs are almost always a missing reducer** — `{"messages": [...]}` under the default reducer *replaces the whole history*. `add_messages` is smarter than `+`: appends, but replaces on matching IDs, which is what makes history editing / conversational time travel work. Reducers aren't bookkeeping — they're what makes parallelism safe; two same-super-step writes to one unreduced channel **raise**, not silently pick. **Short-term (thread-scoped, checkpointer, "what did we just say?") vs long-term (cross-session, store, "who is this person?")** — conflating them is a common design mistake. Checkpointing: `compile(checkpointer=InMemorySaver())` (legacy alias `MemorySaver`) + `{"configurable": {"thread_id": ...}}` + `graph.get_state_history(config)` → inspection / replay / recovery from one mechanism; the compliance angle ("show me why the system refunded this customer") is real, not a throwaway. **`StateGraph` vs `AgentExecutor`**: `create_react_agent` = a fixed loop someone else wrote (can't insert approval gates, cap per-tool retries, swap models on low confidence, or checkpoint mid-loop); `StateGraph` = the primitives that loop is built from. `networkx`/`DiGraph` snippet used purely as a drawing aid (NOT part of LangGraph; real tool is `graph.get_graph().draw_mermaid_png()`) to land the punchline: **a chain is a degenerate graph** (one way in, one way out per node) — add a single backward edge and you get retry/reflection, which is the entire difference between the paradigms.

## Running Project: Research Assistant (`start learning/research_assistant/`)
_Named and scaffolded at Module 3. Fresh build, graph-first — the LangChain course's Developer Documentation Assistant is not carried over. Chosen because it is the smallest project that genuinely needs **every** mechanism in the syllabus: a loop (re-search on weak sourcing), a tool (search), accumulating state (sources), a checkpoint (searches are slow), and an interrupt (human approves citations)._

**Version roadmap:** v0.1 Module 3 (linear 4-node graph) → v0.2 Module 4 (prebuilt agent + hooks) → v0.3+ Module 5 (backward edge, persistence, time travel, interrupts, memory) → v0.4+ Module 6 (subgraphs, streaming).

### Current State (v0.1 — linear, runs end-to-end)

Files are **real on disk**, not just in the lesson. `pip install -r "start learning/requirements.txt"`, then `python -m research_assistant.main` from inside `start learning/`.

```
start learning/
├── .env                      # gitignored — HUGGINGFACEHUB_API_TOKEN
├── .env.example
├── requirements.txt          # langgraph, langchain-core, langchain-huggingface, python-dotenv
└── research_assistant/
    ├── __init__.py
    ├── state.py
    ├── corpus.py
    ├── nodes.py
    ├── hooks.py              # NEW (Module 4.1) — pre_summarize, post_summarize, wrap_with_hooks
    ├── graph.py
    └── main.py
```

**`research_assistant/state.py`**
```python
import operator
from typing import Annotated, TypedDict


class Source(TypedDict):
    title: str
    text: str


class ResearchState(TypedDict):
    question: str                                    # what the user asked
    keywords: list[str]                              # what `understand` decided to look for
    sources: Annotated[list[Source], operator.add]   # ACCUMULATES across searches
    summary: str                                     # the drafted answer
    verdict: str                                     # evaluator's judgement
```

**`research_assistant/corpus.py`**
```python
"""A stand-in for a real search tool. Module 5 swaps this for a real retriever."""

CORPUS = [
    {
        "title": "LangGraph Docs — Core Concepts",
        "text": "LangGraph models an application as a graph. Nodes do work, edges decide "
                "what runs next, and a shared state object is threaded through both.",
        "tags": ["langgraph", "architecture", "graph", "nodes", "edges", "state"],
    },
    {
        "title": "Pregel and Super-Steps",
        "text": "Execution follows the Pregel model. Each super-step plans which nodes are "
                "active, runs them in parallel, then merges their writes at a barrier.",
        "tags": ["langgraph", "architecture", "pregel", "execution", "parallel"],
    },
    {
        "title": "Channels and Reducers",
        "text": "State keys are channels. A reducer defines how a new write merges with the "
                "existing value. The default reducer overwrites; add_messages appends.",
        "tags": ["langgraph", "state", "channels", "reducers", "architecture"],
    },
    {
        "title": "Checkpointing Guide",
        "text": "A checkpointer persists state after every super-step, which is what enables "
                "resume-after-crash, replay, and human-in-the-loop pauses.",
        "tags": ["langgraph", "checkpointing", "persistence", "reliability"],
    },
]


def search_corpus(keywords: list[str], limit: int = 3) -> list[dict]:
    """Score every document by keyword overlap, return the best `limit` matches."""
    scored = []
    for doc in CORPUS:
        hits = len(set(keywords) & set(doc["tags"]))
        if hits:
            scored.append((hits, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{"title": doc["title"], "text": doc["text"]} for _, doc in scored[:limit]]
```

**`research_assistant/nodes.py`**
```python
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .corpus import search_corpus
from .state import ResearchState

load_dotenv()

MIN_SOURCES = 3
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_client = None


def get_client() -> InferenceClient:
    """Built once, lazily — so importing this module doesn't need a token."""
    global _client
    if _client is None:
        _client = InferenceClient(
            provider="novita",
            api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )
    return _client


STOPWORDS = {
    "explain", "using", "what", "how", "why", "the", "a", "an", "and", "or",
    "with", "for", "of", "in", "to", "is", "are", "reliable", "sources", "me",
}

SYSTEM_PROMPT = (
    "You are a research assistant. Answer ONLY from the sources provided. "
    "Cite each source by its title in square brackets. If the sources do not "
    "cover the question, say so plainly."
)


# ── planner ──────────────────────────────────────────────────────────────
def understand(state: ResearchState) -> dict:
    words = (w.strip("?.,\"'").lower() for w in state["question"].split())
    keywords = [w for w in words if w and w not in STOPWORDS and len(w) > 2]
    return {"keywords": keywords}


# ── retriever ────────────────────────────────────────────────────────────
def search(state: ResearchState) -> dict:
    found = search_corpus(state["keywords"])
    # `sources` has an `operator.add` reducer, so this APPENDS.
    return {"sources": found}


# ── executor ─────────────────────────────────────────────────────────────
def summarize(state: ResearchState) -> dict:
    sources_text = "\n\n".join(
        f"[{s['title']}] {s['text']}" for s in state["sources"]
    ) or "(no sources found)"

    response = get_client().chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {state['question']}\n\nSources:\n{sources_text}"},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    summary = response.choices[0].message.content
    return {"summary": summary}


# ── evaluator ────────────────────────────────────────────────────────────
def evaluate(state: ResearchState) -> dict:
    count = len(state["sources"])
    if count >= MIN_SOURCES:
        return {"verdict": f"ok — {count} sources"}
    return {"verdict": f"weak — only {count} source(s), wanted {MIN_SOURCES}"}
```

**`research_assistant/hooks.py`** (added Module 4.1)
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

**`research_assistant/graph.py`** (updated Module 4.1 — wraps summarize with hooks)
```python
from langgraph.graph import END, START, StateGraph

from .hooks import post_summarize, pre_summarize, wrap_with_hooks
from .nodes import evaluate, search, summarize, understand
from .state import ResearchState


def build_graph():
    builder = StateGraph(ResearchState)

    # What can run.
    builder.add_node("understand", understand)
    builder.add_node("search", search)
    builder.add_node("summarize", wrap_with_hooks(summarize, pre=pre_summarize, post=post_summarize))
    builder.add_node("evaluate", evaluate)

    # What runs after what. (Module 5 replaces the last edge with a conditional one.)
    builder.add_edge(START, "understand")
    builder.add_edge("understand", "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "evaluate")
    builder.add_edge("evaluate", END)

    return builder.compile()
```

**`research_assistant/main.py`**
```python
from research_assistant.graph import build_graph

QUESTION = "Explain LangGraph architecture using 3 reliable sources."


def main():
    graph = build_graph()

    # The structure, as text. No extra dependencies needed.
    print(graph.get_graph().draw_mermaid())

    final = graph.invoke({
        "question": QUESTION,
        "sources": [],          # reducer channels: seed them explicitly
    })

    print("\nKEYWORDS:", final["keywords"])
    print("\nSOURCES:")
    for s in final["sources"]:
        print("  -", s["title"])
    print("\nSUMMARY:\n", final["summary"])
    print("\nVERDICT:", final["verdict"])


if __name__ == "__main__":
    main()
```

**Deliberate v0.1 debts, to be paid in later modules — do not "fix" them early:**
- `evaluate` computes a verdict nobody acts on. That string becomes a **routing decision** in Module 5 (`add_conditional_edges`), replacing `add_edge("evaluate", END)`. This is the one backward edge that turns the chain into an agent.
- `sources` already carries `operator.add` even though nothing loops yet — written a module early on purpose, so the loop doesn't silently drop results later.
- `corpus.py` is a keyword-overlap stub standing in for a retriever; swapped in Module 5's *Tool* sub-topic.
- No checkpointer on `compile()` yet — added in Module 5's *Persistence* sub-topic.

## Concepts Covered (cont.)
- Module 4.1 > Hooks — paid the anatomy-table debt: the "Hooks & Transitions" row had "edges" for transitions but left hooks unexplained. **Hooks = structured intercept points that fire around a node, in-process, in plain Python.** Two types: `pre_model_hook(state) -> state` runs BEFORE the node (can modify what the node sees); `post_model_hook(state) -> state` runs AFTER (for observation and side effects). Flow: `Input → [pre-hook] → Node (LLM/tool) → [post-hook] → Output`. Two wiring patterns: (1) **manual wrapper** for custom `StateGraph` — `wrap_with_hooks(node_fn, pre, post)` composes pre → node → post inside one registered function; pre-hook return IS used (modifies node input); post-hook return is ignored (observation-only in this pattern — wrapped returns `result`, not post-hook return); (2) **`create_react_agent` kwargs** — pass `pre_model_hook=` and `post_model_hook=` as kwargs, LangGraph calls them around every LLM step; post-hook CAN return a propagating partial update in this mode (covered next sub-topic). Source showed `graph.add_pre_model_hook()` / `builder.add_post_model_hook()` on `StateGraph` — these do not exist; hook kwargs belong to `create_react_agent`; source's actual working code used Pattern 1 correctly. Node.js/Express analogy: same shape (`req → middleware → handler → middleware → res`) but hooks operate on the full typed state dict, not HTTP objects. Project extended with `hooks.py` (`pre_summarize`, `post_summarize`, `wrap_with_hooks`) and `graph.py` updated to wrap the `summarize` node — `nodes.py` unchanged, which is the point. `hooks.py` detail: post-hook receives `{**state, **result}` (merged) so it can see the node's output; wrapper returns `result` so LangGraph reducer logic is unaffected.

- Module 3 > Getting Started with LangGraph: Building Your First AI Workflow Graph — **first code in the course; project named + scaffolded here.** Sub-topic *"What are Agents?"* compressed hard against Module 1 (it restates agent-as-LLM-defined-control-flow). **`Agent = LLM + Tools + Reasoning Loop + State`** — taught as four load-bearing terms, each with what breaks if dropped (LLM→hard-coded script · Tools→chatbot that can only talk · Loop→one-shot answer · State→amnesia). **The exit-condition contrast is the core framing:** a plain LLM call stops when *the text ran out*; an agent stops when *the goal was met* — and that single difference is what multi-step reasoning / error recovery / tool calls / long-running work / autonomous decisions all reduce to (one feature, four angles, not five bullets). Loop beats: `Reason → Act → Observe → Loop`. Flight-booking example ("cheapest flight to Dubai next Friday, notify on WhatsApp") used to make it concrete — **steps "compare prices" and "select an option" are decisions made after seeing data that didn't exist at request time**, which is why they couldn't be pre-written. Six characteristics (Autonomy · Reasoning · Tool Use · Memory · Modularity · Adaptability) taught as a table with a **cost column added** (unpredictability · latency+tokens · real side effects · storage+context pressure · interface to maintain · behaviour drift) and the point landed that *the cost column is the reason LangGraph exists* — every mechanism in the course buys a benefit while capping its cost.
  **The module's real payload = the Anatomy → LangGraph mapping table:** LLM Core → inside a node body (LangGraph never calls a model itself) · Prompt Template → inside a node body, `ChatPromptTemplate`/LCEL · Memory Module → a state channel (short-term) **+** checkpointer/store (long-term) · Toolset → nodes running tools, `@tool` wrapped or prebuilt `ToolNode` · State → the `TypedDict` passed to `StateGraph(...)` · Hooks & Transitions → edges + conditional edges. Three call-outs: the top two rows **are LangChain, not LangGraph** (2a's "LangChain inside a node" cashed in); "Memory Module" secretly covers two different things and that's where the short/long-term conflation starts; **Hooks & Transitions is the only row with no LangChain equivalent — which is why it gets its own module (4)**. "Why agents matter in LangGraph" (4 source claims) collapsed into an *agents-need-X → LangGraph-gives-X* table pointing back at 2b, whose real value is naming the **four node roles: planner / retriever / executor / evaluator** — the skeleton of nearly every agent graph.
  **Source correction made honestly:** the course's Research Assistant diagram (User Query → Planner → Search Tool → Summarizer → Evaluator → Result) has every arrow pointing down — **it is a chain, not an agent**; if the Evaluator finds only 1 source it has nowhere to send the work. Drew the version with the `evaluate → search` backward edge as what we're actually building toward, and framed v0.1 as deliberately shipping the straight line so mechanics land before branching.
  **Hands-on, taught in two steps.** (1) A no-LLM "hello graph" (`CounterState`, `double`/`describe`) to prove the machinery, with the four first-graph gotchas: nodes return **partial updates** (returning full state is the classic beginner habit) · **nodes never call each other, the edge knows** (the indirection that makes nodes swappable) · **`START`/`END` are real nodes** — `add_edge(START, x)` is what sets the entry point, there's no implicit first-node-wins · **`compile()` is a real validation step**, catches unreachable nodes/bad edge targets/no path from START, and is also where checkpointers attach later. LCEL contrast made explicit: **`a | b` fuses "what runs" with "what runs next"; `add_node` + `add_edge` split them** — two lines now buys cycles later. (2) Project v0.1. **Visualisation: `graph.get_graph().draw_mermaid()`** returns Mermaid text with zero extra deps (`draw_mermaid_png()` needs a renderer) — an architecture diagram that *cannot drift from the code because it is the code*; this is the real tool 2b's `networkx` snippet was only miming. Two `invoke` details taught before they bite: **`TypedDict` isn't runtime-enforced** so keys no node has written simply don't exist → use `state.get(k, default)`; and **seed reducer channels explicitly** (`"sources": []`). Five use cases (customer support · enterprise knowledge retrieval · research assistance · workflow automation · education/tutoring) taught via a graph-shape column, landing: **five domains, ~four shapes — route / loop / delegate / remember** — which is why the course teaches the graph, not a catalogue of agents. Closed with a first-graph mistakes table (KeyError → `.get()` · state resetting → full-state return or missing reducer · never terminates → no edge to END · unreachable node → forgot `add_edge(START, ...)` · parallel unreduced writes raise **by design**).

## Teaching Style — THE JOB (learner-clarified at Module 1; applies to every lesson)
**Rewrite the course in easy-to-learn form.** The learner found the source material boring and repetitive and asked for
it to be **re-taught simply** — not summarized. Read this as: *you are re-authoring the course for a developer.*

- **Teach fully.** Every real concept from the screenshots is covered and explained. Don't produce a digest.
- **Build up in a learnable order:** problem → concept → why it matters → how it's solved. Never dump conclusions first.
- **Simple language, short sentences, concrete examples.** Carry **one running example** through the lesson
  (e.g. Ravi's logistics support bot in Module 1) rather than scattering unrelated ones.
- **Remove repetition, not content.** The source says the same thing 4–5 times; say it once, well, in the right place.
  That is the only cutting to do.
- **No meta-commentary.** No "what I cut and why" sections — a course doesn't discuss its own editing. *(An earlier
  draft of Module 1 did this and was corrected.)*
- Persona dialogues in the source (Ravi, Riya/Karan) are **good teaching devices** — reuse them as worked examples.
- Tables and scannable blocks over prose walls; text diagrams over long descriptions.
- End each lesson with a **short self-check list** of questions the learner should be able to answer.

## Next Up
**Module 4.2 — langgraph-prebuilt** (15m): `create_react_agent` — the black box opened. Same four node roles (planner/retriever/executor/evaluator) wired into a loop by someone else. Use the real `pre_model_hook`/`post_model_hook` kwargs API. Project reaches v0.2: rebuild the Research Assistant as a prebuilt agent side-by-side with the hand-built v0.1+hooks.

Angle to take: the learner used `create_react_agent` as a **black box** in LangChain Module 8, and has now hand-built a `StateGraph` in Module 3. So teach Module 4 as **"here's what was actually running"** — the same four node roles (planner/retriever/executor/evaluator) wired into a loop by someone else — not as a new tool. 2b already set this up with the `StateGraph` vs `AgentExecutor` contrast (a fixed loop you can't insert approval gates into vs. the primitives that loop is built from); land the payoff, don't re-argue it.

**Hooks is the sub-topic to spend the time on** — Module 3's anatomy table left "Hooks & Transitions" as the *only* row with no LangChain equivalent and explicitly promised Module 4 would fill it in. That's a debt to pay.

Project should reach **v0.2**: rebuild the same Research Assistant job with a prebuilt agent + hooks, side by side with the hand-built v0.1, so the trade (less code vs. less control) is concrete rather than asserted. `langgraph-supervisor` (15m) maps onto the "delegate" graph shape named in Module 3's use-case table.

Curriculum is locked into `CLAUDE.md` from the syllabus screenshots — 6 modules, numbering final, **10h 20m of teaching content**. Prelude, Exercises, and all Quizzes are **intentionally out of scope** by the learner's choice; that accounts for the gap against the course's stated 15h 30m. Don't flag it, don't ask for more screenshots of it, don't write quiz questions unasked.
