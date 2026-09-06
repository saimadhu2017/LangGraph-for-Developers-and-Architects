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

**Version roadmap:** v0.1 Module 3 (linear 4-node graph) → v0.2 Module 4 (prebuilt agent + hooks) → v0.3 Module 5.1 (conditional edge + retry loop) → v0.4 Module 5.2 (persistence) → v0.5 Module 5.3 (time travel) → **v0.6 Module 5.4 (tools) ✅** → v0.7+ rest of Module 5 (interrupts, memory) → v0.8+ Module 6 (subgraphs, streaming).

### Current State (v0.6 — tools: @tool search_docs in tools.py, Pattern B demo in tools_demo.py)

Files are **real on disk**, not just in the lesson. `pip install -r "start learning/requirements.txt"`, then `python -m research_assistant.main` from inside `start learning/`.

```
start learning/
├── .env                      # gitignored — HUGGINGFACEHUB_API_TOKEN
├── .env.example
├── requirements.txt          # langgraph, langchain-core, langchain-huggingface, python-dotenv
└── research_assistant/
    ├── __init__.py
    ├── state.py              # Updated (5.1) — `attempts` loop counter
    ├── corpus.py             # Updated (5.1) — require_all + exclude_titles
    ├── tools.py              # NEW (5.4) — @tool search_docs (public tool interface)
    ├── nodes.py              # Updated (5.1) — retry-aware search, MIN_SOURCES=4, MAX_ATTEMPTS=3
    ├── hooks.py              # Module 4.1 — pre_summarize, post_summarize, wrap_with_hooks
    ├── agent.py              # Updated (5.4) — imports search_docs from tools.py
    ├── controller_demo.py    # NEW (5.1) — LLM writes `decision`, router reads it
    ├── tools_demo.py         # NEW (5.4) — Pattern B: LLM + bind_tools + ToolNode + tools_condition
    ├── graph.py              # Updated (5.2) — build_graph() takes optional checkpointer=
    └── main.py               # Updated (5.4) — run_tools_demo() added
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
    keywords: list[str]                              # OVERWRITTEN — the retry rewrites the query
    sources: Annotated[list[Source], operator.add]   # ACCUMULATES across searches
    summary: str                                     # the drafted answer
    verdict: str                                     # evaluator's judgement — now a ROUTING input
    attempts: Annotated[int, operator.add]           # NEW (v0.3) — how many times `search` has run
```

**`research_assistant/corpus.py`**
```python
"""A stand-in for a real search tool. Module 5's *Tool* sub-topic swaps this for a real retriever."""

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


def search_corpus(
    keywords: list[str],
    limit: int = 3,
    require_all: bool = False,
    exclude_titles: set[str] | None = None,
) -> list[dict]:
    """Score every document by keyword overlap, return the best `limit` matches.

    v0.3 added two parameters so the retry can ask a *different* question than the
    first attempt did:

    - `require_all=True`  → strict: a doc must match every keyword. Precise, few hits.
    - `exclude_titles`    → don't return documents we already have in state.
    """
    wanted = set(keywords)
    exclude = exclude_titles or set()

    scored = []
    for doc in CORPUS:
        if doc["title"] in exclude:
            continue
        hits = len(wanted & set(doc["tags"]))
        if require_all and hits < len(wanted):
            continue
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

MIN_SOURCES = 4       # v0.3: raised, so the first (strict) search deliberately falls short
MAX_ATTEMPTS = 3      # v0.3: the cycle's brake — without this the loop can run forever
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

# v0.3 — the retry must ask a DIFFERENT question than the first attempt.
# Re-running an identical query is how a cycle turns into an infinite loop.
BROADER_TERMS = {
    "architecture": ["execution", "persistence", "reliability"],
    "langgraph": ["graph", "nodes", "edges"],
    "state": ["channels", "reducers"],
    "memory": ["checkpointing", "persistence"],
}

SYSTEM_PROMPT = (
    "You are a research assistant. Answer ONLY from the sources provided. "
    "Cite each source by its title in square brackets. If the sources do not "
    "cover the question, say so plainly."
)


def _broaden(keywords: list[str]) -> list[str]:
    """Add parent/sibling terms for each keyword, preserving order and dropping dupes."""
    widened = list(keywords)
    for word in keywords:
        for extra in BROADER_TERMS.get(word, []):
            if extra not in widened:
                widened.append(extra)
    return widened


# ── planner ──────────────────────────────────────────────────────────────
def understand(state: ResearchState) -> dict:
    words = (w.strip("?.,\"'").lower() for w in state["question"].split())
    keywords = [w for w in words if w and w not in STOPWORDS and len(w) > 2]
    return {"keywords": keywords}


# ── retriever ────────────────────────────────────────────────────────────
# v0.3 — runs MORE THAN ONCE, so it reads which pass it is on from state
# (`attempts`), not from a variable it kept in memory: nodes are stateless.
def search(state: ResearchState) -> dict:
    attempt = state.get("attempts", 0)
    keywords = state["keywords"]

    if attempt == 0:
        # First pass: strict. A document must match every keyword.
        found = search_corpus(keywords, limit=3, require_all=True)
        return {"sources": found, "attempts": 1}

    # Retry: broaden the query, and skip documents we already collected.
    already = {s["title"] for s in state.get("sources", [])}
    widened = _broaden(keywords)
    found = search_corpus(widened, limit=3, require_all=False, exclude_titles=already)

    # Three keys, three different merge rules — all in one return:
    #   keywords -> default reducer, OVERWRITES (the query is replaced)
    #   sources  -> operator.add,    APPENDS   (attempt 1's results are kept)
    #   attempts -> operator.add,    SUMS      (0 + 1 + 1 = 2)
    return {"keywords": widened, "sources": found, "attempts": 1}


# ── executor ─────────────────────────────────────────────────────────────
# v0.3 — runs again after every retry, on the LARGER source set.
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
# v0.3 — the verdict is finally READ by something (`route_after_evaluate`).
# The node still doesn't know where the work goes; deciding is the edge's job.
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

**`research_assistant/graph.py`** (updated Module 5.2 — optional checkpointer)
```python
from langgraph.graph import END, START, StateGraph

from .hooks import post_summarize, pre_summarize, wrap_with_hooks
from .nodes import MAX_ATTEMPTS, evaluate, search, summarize, understand
from .state import ResearchState


# ── the router ───────────────────────────────────────────────────────────
# NOT a node. It does no work and writes nothing to state. It reads state and
# returns a LABEL saying where the work goes next. This function is the entire
# difference between v0.2's chain and v0.3's agent.
def route_after_evaluate(state: ResearchState) -> str:
    if state["verdict"].startswith("ok"):
        return "done"

    # The brake. A cycle with no exit condition is an infinite loop, and the
    # only thing that would stop it is LangGraph's recursion_limit blowing up.
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "give_up"

    return "retry"


def build_graph(checkpointer=None):
    builder = StateGraph(ResearchState)

    # What can run.
    builder.add_node("understand", understand)
    builder.add_node("search", search)
    builder.add_node("summarize", wrap_with_hooks(summarize, pre=pre_summarize, post=post_summarize))
    builder.add_node("evaluate", evaluate)

    # What runs after what — the fixed part of the control flow.
    builder.add_edge(START, "understand")
    builder.add_edge("understand", "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "evaluate")

    # v0.3 — the one decided-at-runtime part. Replaces `add_edge("evaluate", END)`.
    # The path_map translates the router's vocabulary into node names, so the router
    # never has to know the graph's topology.
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "retry": "search",   # <-- the BACKWARD edge. This is the cycle.
            "done": END,
            "give_up": END,
        },
    )

    # v0.4 — optional checkpointer. None = no persistence (v0.3 behaviour).
    # Pass InMemorySaver() for dev, SqliteSaver for production.
    return builder.compile(checkpointer=checkpointer)
```

**`research_assistant/tools.py`** (NEW Module 5.4 — public tool interface)
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

**`research_assistant/agent.py`** (updated Module 5.4 — imports search_docs from tools.py)
```python
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import create_react_agent

from .tools import search_docs

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

**`research_assistant/controller_demo.py`** (added Module 5.1 — the LLM picks the edge)
```python
"""Module 5.1 demo — the router's answer comes from the LLM, not from an `if`.

Rebuilt from the course's AWS Bedrock example on the HuggingFace client this project
already uses. Three things were changed on purpose:
  1. Nodes return PARTIAL UPDATES instead of mutating and returning the whole state.
  2. `messages` uses the `add_messages` reducer instead of a hand-rolled list append.
  3. The LLM's free-text answer is normalised before it is allowed to route.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .nodes import MODEL, get_client


class ControllerState(TypedDict):
    user_input: str
    summary: str
    messages: Annotated[list, add_messages]   # appends; never replaces history
    decision: str                             # "summarize" | "validate" | "end"


def _ask(prompt: str, max_tokens: int = 256) -> str:
    response = get_client().chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


CHOICES = ("summarize", "validate", "end")


# ── Node 1: the LLM decides what should happen next ──────────────────────
def controller_node(state: ControllerState) -> dict:
    prompt = f"""You are an AI workflow controller.
User said: {state["user_input"]}
Already summarised: {"yes" if state.get("summary") else "no"}

Decide the next step in the workflow.
Options:
1. summarize - if the user input is long text and has not been summarised
2. validate  - if there is already a summary
3. end       - if no action is needed

Answer with ONLY one word: summarize, validate, or end."""

    raw = _ask(prompt, max_tokens=5).lower()

    # NEVER route on raw model output. An LLM that replies "Summarize!" or
    # "I would summarize this" must not be able to crash the graph with a
    # KeyError on the path_map. Unrecognised answers fall back to a safe exit.
    decision = next((c for c in CHOICES if c in raw), "end")

    return {
        "decision": decision,
        "messages": [AIMessage(content=f"Decision: {decision}", name="controller")],
    }


# ── Node 2: summarise ────────────────────────────────────────────────────
def summarizer_node(state: ControllerState) -> dict:
    summary = _ask(f"Summarize this text briefly:\n\n{state['user_input']}")
    return {
        "summary": summary,
        "messages": [
            AIMessage(
                content=f"Summary: {summary}",
                name="summarizer",
                additional_kwargs={"node": "summarizer", "type": "summary"},
            )
        ],
    }


# ── Node 3: validate ─────────────────────────────────────────────────────
def validator_node(state: ControllerState) -> dict:
    summary = state.get("summary", "")
    verdict = _ask(
        f"Is the following summary under 80 words? Reply only with Yes or No.\n\n{summary}",
        max_tokens=5,
    )
    return {
        "messages": [
            AIMessage(
                content=f"Validation result: {verdict}",
                name="validator",
                additional_kwargs={"node": "validator", "type": "validation"},
            )
        ]
    }


def build_controller_graph():
    builder = StateGraph(ControllerState)

    builder.add_node("controller", controller_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("validator", validator_node)

    builder.add_edge(START, "controller")

    # The router here is a one-liner because controller_node already did the
    # thinking and parked its answer in state. Routers should stay this thin:
    # do the work in a node, decide in the edge.
    builder.add_conditional_edges(
        "controller",
        lambda state: state["decision"],
        {"summarize": "summarizer", "validate": "validator", "end": END},
    )

    builder.add_edge("summarizer", "validator")
    builder.add_edge("validator", END)

    return builder.compile()


TEXT = (
    "LangGraph is a Python framework that helps developers build complex, "
    "stateful, multi-step AI agents using a directed graph of nodes."
)


def run(text: str = TEXT) -> dict:
    app = build_controller_graph()
    return app.invoke({
        "user_input": text,
        "summary": "",
        "messages": [HumanMessage(content=text)],
        "decision": "",
    })
```

**`research_assistant/tools_demo.py`** (NEW Module 5.4 — Pattern B demo)
```python
"""Module 5.4 demo — Pattern B: LLM-driven tool use via ToolNode + tools_condition."""
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

**`research_assistant/main.py`** (updated Module 5.4 — run_tools_demo() added)
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

    # Latest snapshot — the final state for this thread
    snapshot = graph.get_state(config)
    print("\n--- final snapshot ---")
    print("  verdict  :", snapshot.values.get("verdict"))
    print("  attempts :", snapshot.values.get("attempts"))
    print("  sources  :", len(snapshot.values.get("sources", [])))
    print("  next     :", snapshot.next)   # () means graph finished cleanly

    # Full checkpoint history, newest first — one row per superstep
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


def run_tools_demo():
    print("\n" + "=" * 60)
    print("v0.6 — Tool: Pattern B (LLM + bind_tools + ToolNode)")
    print("=" * 60)
    from research_assistant import tools_demo
    result = tools_demo.run()
    print("\n--- message thread ---")
    for msg in result["messages"]:
        who = getattr(msg, "name", None) or msg.__class__.__name__
        print(f"  {who:<20} | {_short(msg.content, 80)}")


def main():
    run_tools_demo()
    run_v5_time_travel()
    run_v4()
    run_v3()
    run_v2()
    run_controller_demo()


if __name__ == "__main__":
    main()
```

**Verified v0.3 run:** 2 attempts. Strict pass → 3 docs → `weak — only 3 source(s), wanted 4` → router returns `"retry"` → broadened query (2 keywords → 8) → `Checkpointing Guide` appended → 4 docs → `ok — 4 sources` → `"done"` → END. All three reducer behaviours visible in the two `search` steps: `keywords` overwritten, `sources` appended, `attempts` summed. Module 4.1's hooks fire twice, `hooks.py` unchanged.

**Verified v0.4 run:** `run_v4()` runs with `InMemorySaver` + `thread_id="research-001"`. `get_state(config)` returns the final `StateSnapshot` with `next=()`. `get_state_history(config)` yields one snapshot per superstep (8 rows for a 2-attempt run). `build_graph(checkpointer=None)` is backward-compatible — `run_v3()` still works unchanged.

**Verified v0.5 run:** `run_v5_time_travel()` runs on thread `"travel-demo"`. Original run: 2 attempts, 4 sources, `ok`. Fork point: step 1, `next=("search",)`, keywords `["langgraph", "architecture"]`. `update_state` injects `["checkpointing", "persistence", "state", "channels", "reducers"]` with `as_node="understand"`. Time-travel run: 3 sources, `weak — only 3 source(s), wanted 4` (hits MAX_ATTEMPTS). Comparison shows different source sets from same graph.

**Debts remaining, to be paid in later sub-topics — do not "fix" them early:**
- ~~`evaluate` computes a verdict nobody acts on~~ — **PAID in 5.1.** `route_after_evaluate` + `add_conditional_edges` turned it into the backward edge.
- ~~`sources` carries `operator.add` before anything loops~~ — **PAID in 5.1.** The loop now depends on it; under the default reducer the retry would discard attempt 1 and never converge.
- ~~No checkpointer on `compile()` yet~~ — **PAID in 5.2.** `build_graph(checkpointer=)` + `InMemorySaver` + `thread_id`; `get_state` / `get_state_history` demonstrated.
- ~~No way to inspect or rewind to an earlier step~~ — **PAID in 5.3.** `update_state(fork_snap.config, values, as_node=...)` + re-invoke from returned config; `get_state_history` read with intent to find `snap.next == ("search",)` fork point.
- ~~`corpus.py` is still a keyword-overlap stub standing in for a retriever~~ — **PAID in 5.4.** `tools.py` added with `@tool search_docs`; `agent.py` updated to import from it; `tools_demo.py` shows Pattern B (LLM + bind_tools + ToolNode + tools_condition).
- Nothing pauses for a human to approve citations → **5.5 Interrupts**.
- Nothing is remembered between questions → **5.6 Memory**.

## Concepts Covered (cont.)
- Module 5.1 > Building the Workflow — **the sub-topic that turns the chain into an agent.** Framed as **seven build steps** (StateGraph → nodes → entry → edges/exit → compile → visualise → run), with the call-out that steps 1–5 run **once at import time** and only step 7 is per-request — that build/run split is *why* checkpointing, visualisation and replay are possible. **Step 1 State:** the one dict every node reads/writes (whiteboard analogy retained); three payoffs — centralised context / modularity / **traceability, which is not a nice-to-have because Time Travel depends on state being a plain serialisable value**; three declaration options with a when-to-pick table (`TypedDict` default · `dataclass` for attribute access · Pydantic `BaseModel` when you need real runtime validation, at the cost of a validation pass per transition). **Step 2 Nodes:** function → partial dict. **Three source errors corrected**: (a) `add_node("greet", function=fn)` — the second arg is positional (`action`); `function=` raises `RuntimeError` in langgraph 1.x (verified); (b)+(c) the source's node mutates `state["messages"]` in place *and* returns the whole list — traced the resulting duplication arithmetic (`["a","b","c"] + ["a","b","c"]`) and showed it compounds per loop. Async nodes (`ainvoke`/`astream`); **an LCEL chain registered directly as a node** (Module 2a cashed in) with the plumbing gotcha that the chain receives the *whole state dict*, so prompt vars must be state keys or you wrap it. "Keep nodes single-purpose" reframed as: a node is the unit of **retry, checkpointing, and routing** — a node doing four things is four things you can't retry/resume/route independently. **Steps 3–4 Entry/Exit:** `set_entry_point`/`set_finish_point` = sugar for `add_edge(START/END, …)`; prefer the sentinels. **Source corrected:** you do NOT need a custom `end` node returning `{}` — `END` already exists, and the extra node costs a super-step + checkpoint write for nothing. "Only one entry point" made precise: true of the *helper*, but `START` can fan out to multiple nodes (Pregel parallel super-step). Multiple exit points confirmed, and that's how the router terminates. **Step 5 compile():** real validation (edge targets exist · path from START · no unreachable nodes) + **it's where the rest of Module 5 plugs in** (`checkpointer=` → Persistence/Time Travel, `interrupt_before=` → Interrupts, `store=` → Memory) — landed as "**persistence is an argument to compile(), not something bolted onto nodes**", a direct consequence of the graph being a data structure. **Step 6 Visualise:** `draw_ascii()` **requires the `grandalf` package** (verified ImportError) — `draw_mermaid()` has zero extra deps, which is why the project uses it; value = a diagram generated from the compiled graph, so it cannot drift. **Step 7 Run:** `invoke()` = final state only; `stream()` yields **once per node completion**, chunk shape `{node_name: update_dict}` — and on a *looping* graph `stream()` is the only way to see `search` run twice. `stream_mode` table: `"updates"` (default) · `"values"` · `"messages"` (token streaming for typing-effect UIs) · `"debug"`. Plus `ainvoke`/`astream`/`batch`. When-to-use table from source retained.
  **THE PAYLOAD — conditional edges.** Shape: `add_conditional_edges(source_node, router_fn, path_map)`. Four properties taught: **(1) the router is NOT a node** — no checkpoint, no retry, and **it must not write to state (writes are silently discarded)**; do the work in a node, decide in the edge. **(2) it returns a LABEL, not a node name** — the `path_map` translates, so routers speak domain vocabulary (`retry`/`escalate`/`give_up`) and renames touch one dict. **(3) this is where Module 1's "partial control flow definition" physically lives** — fixed-at-compile-time (which nodes exist, what each label means, that these are the ONLY options) vs decided-at-runtime (which label); the router **cannot invent a third destination or reach an unwired node**, and that containment IS the safety argument. **(4) the router can be an LLM call and still be safe**, for the same reason. **The backward edge**: `"retry": "search"` points at a node that already ran — that one map entry is the cycle, and it's where `sources`' `operator.add` (written in Module 3) finally earns its keep, since under the default overwrite reducer the retry would discard attempt 1's work and the loop would never converge. **The brake, two mechanisms, need both:** `recursion_limit` is a **crash backstop** (default **10007** in langgraph 1.2.11, verified — not 25; settable per-run via `config={"recursion_limit": n}`, raises `GraphRecursionError`), and the *real* brake is an **explicit counter in state** (`attempts: Annotated[int, operator.add]` makes `{"attempts": 1}` an increment). Three loop-safety rules: count in **state** not a module global (nodes are stateless; a global breaks under concurrency and is invisible to the checkpointer) · **the retry must ask a DIFFERENT question** (identical query → identical answer → spins to the limit) · **give up explicitly** (`"give_up"` → partial results + honest verdict, vs. a stack trace).
  **Two more source API errors corrected:** `Send` is imported from **`langgraph.types`**, not `langgraph.graph` (verified), and **there is no `Gather()`** — there's nothing to gather with because **the reducer IS the gather** (parallel writes land at the same Pregel barrier). `Send` explained as *dynamic* fan-out (N unknown until runtime; a router may return a list of `Send` objects). And **`CheckpointAt` does not exist in any released LangGraph** (verified `hasattr` False) — the source's `CheckpointAt(nodes=[...], config={"save":True,"load":True})` is fiction; real API is `compile(checkpointer=InMemorySaver())`, and **you don't nominate nodes: every super-step is checkpointed**, because the barrier is the only moment state is consistent. Also flagged: the source's Bedrock slide **hard-codes a live AWS access key + secret** — not reproduced, rotation advised.
  **Demo A (state management + annotated messages)** — the source's 2-node summarizer/validator rebuilt with three fixes: `Annotated[list, add_messages]` instead of a bare `List` + four lines of hand-rolled append; partial updates instead of `return state`; and **`name=` / `additional_kwargs=` instead of `metadata=`** — the source's `AIMessage(metadata={...})` *does* run, but only because `BaseMessage` is configured `extra="allow"` (verified: `metadata` is NOT in `model_fields`), so nothing in the ecosystem knows to look there; `name` is a declared field that shows in traces. **Demo B (LLM picks the edge)** — the source's Bedrock "dynamic workflow" rebuilt on HF as real on-disk `controller_demo.py`. Key teaching: **the node calls the LLM and writes `decision` to state; the router is `lambda state: state["decision"]`**. Table justifying the split (node = LLM call/state write/checkpointed/retryable; router = none of those, only decides) — putting the LLM call in the router gives you an expensive, un-retryable, un-checkpointed call whose result is thrown away. **Never route on raw model output**: normalise (`next((c for c in CHOICES if c in raw), "end")`) with a safe default, or `"Summarize."` KeyErrors the path_map. Noted honestly that Demo B is a **one-shot router, not a loop** (no edge back to controller) — the project has the real loop. Closed with the source's **five pillars** (Tools · Memory/Persistence · Human-in-Loop · State Customisation · Time Travel) mapped to what in 5.1 unblocks each, landing: all five rest on **the graph being a data structure and the state being a value**.
  **Project → v0.3** (see full source below): `attempts` counter added to state; `search_corpus` gained `require_all` + `exclude_titles` so the retry asks a *different* query; `MIN_SOURCES` 3→4 so the strict first pass deliberately falls short; `MAX_ATTEMPTS=3`; `route_after_evaluate` + `add_conditional_edges` replaced `add_edge("evaluate", END)`; `main.py` streams so the loop is visible. **Verified run: 2 attempts, 3 docs → weak → retry → broadened query → 4 docs → ok.** The two `search` steps show **all three reducer behaviours in one return dict** — `keywords` overwritten (2→8 words), `sources` appended (3+1=4), `attempts` summed (1+1=2) — and Module 4.1's hooks fire twice with zero changes to `hooks.py`, because hooks were wired to a *node*, not to a run.

- Module 4.3 > langgraph-supervisor — paid the "delegate" graph shape debt from Module 3's use-case table. **Supervisor pattern = two-level hierarchy**: supervisor LLM at top routes work to specialized worker agents; workers know only their own tools and prompt, not each other. **Mechanism = tool-based handoff**: supervisor treats each worker agent as a callable tool (same `tool_calls` / `ToolNode` loop as 4.2, one level up); workers run their own full ReAct loop and return results as `ToolMessage`; supervisor reads results and decides next action. **`create_supervisor` API** (separate package, `pip install langgraph-supervisor`): `agents=[...]` (list of named `create_react_agent` graphs), `model`, `prompt` (routing strategy), `.compile()`. **`name` on `create_react_agent`**: how the supervisor identifies the worker — supervisor sees a tool called `transfer_to_<name>`. **`prompt` on workers**: constrains domain scope. **`prompt` on supervisor**: describes routing strategy across workers. **When to use**: tasks decompose into distinct domains, different system prompts or retry logic needed per tool group, >10 tools causing selection degradation, team boundaries around agents. **Nesting**: workers can themselves be supervisors → hierarchical multi-agent at scale. **No project changes for this sub-topic**: Research Assistant is one domain; adding a supervisor would be fake structure. Module 4 fully complete.

- Module 4.2 > langgraph-prebuilt — **opened the `create_react_agent` black box**. Package contains: `create_react_agent` (ReAct tool-calling), `ToolNode`, `tools_condition`, `ValidationNode`, `HumanInterrupt`/`HumanResponse`, plus higher-level packages (`langgraph-supervisor`, Trustcall, LangMem, LangGraph Swarm — named, not drilled). **`create_react_agent` internals = a two-node `StateGraph`**: `agent` node (LLM with `.bind_tools()`, returns `{"messages": [AIMessage]}`) and `tools` node (`ToolNode` — reads `tool_calls` from last AIMessage, runs them in parallel, returns `ToolMessage` list). Router: `tools_condition` (prebuilt conditional edge function: `tool_calls` present → `"tools"`, else → `END`). Loop: `tools → agent`. State: **`MessagesState`** = `TypedDict` with `messages: Annotated[list[AnyMessage], add_messages]` — everything flows as messages (HumanMessage/AIMessage/ToolMessage). Contrast with our `ResearchState` (5 named keys, structured); message-based is better for conversational agents, structured state is better when nodes need precise typed fields. **`ToolNode`**: reads last AIMessage's `tool_calls`, runs each tool by name, parallel if multiple, wraps results in `ToolMessage` with matching `tool_call_id`, returns `{"messages": [...]}`. `handle_tool_errors=True` (default) catches tool exceptions and returns them as ToolMessage content so the LLM can reason about them. **Hook API difference from 4.1**: `create_react_agent` hooks return `None` (no change, cleaner than returning full state) or a partial update dict (LangGraph handles the merge). Contrast with 4.1 wrapper which returned full state because it was composing inside a plain Python function. Hooks fire every ReAct iteration. **`ValidationNode`**: validates tool-call arguments against Pydantic schemas before execution. On fail: injects a `ToolMessage` with the validation error back into history so LLM can self-correct and retry (self-correcting loop). On pass: forwards the original `AIMessage` to `ToolNode`. Declared with `@tool(args_schema=MySchema)` on the tool and `ValidationNode([MySchema])` as a graph node. **`HumanInterrupt`**: previewed — pauses graph at a node boundary, sends structured request to human, waits for `HumanResponse` before resuming; requires checkpointer; covered fully in Module 5 Interrupts. **v0.1 vs v0.2 trade-off**: v0.2 is less code (2 auto-wired nodes vs 4 explicit) but the ReAct loop is sealed — cannot insert approval gates between specific steps, cap per-tool retries, swap models mid-loop, or route on custom state keys without reconstructing the graph; v0.1 IS the loop, every edge is reachable. **vs LangChain's `create_tool_calling_agent`**: LangGraph = graph-based, ReAct loop, embeddable in supervisor, hooks/subgraphs/conditional edges; LangChain = chain-based, direct function call, simpler but not graph-embeddable. Use LangGraph for multi-step reasoning with loops; use LangChain for single-step structured output. **Project v0.2**: new `agent.py` (`create_react_agent` + `search_docs` tool + pre/post hooks using None-return API); `main.py` updated to run both v0.1 and v0.2 side-by-side.

- Module 4.1 > Hooks — paid the anatomy-table debt: the "Hooks & Transitions" row had "edges" for transitions but left hooks unexplained. **Hooks = structured intercept points that fire around a node, in-process, in plain Python.** Two types: `pre_model_hook(state) -> state` runs BEFORE the node (can modify what the node sees); `post_model_hook(state) -> state` runs AFTER (for observation and side effects). Flow: `Input → [pre-hook] → Node (LLM/tool) → [post-hook] → Output`. Two wiring patterns: (1) **manual wrapper** for custom `StateGraph` — `wrap_with_hooks(node_fn, pre, post)` composes pre → node → post inside one registered function; pre-hook return IS used (modifies node input); post-hook return is ignored (observation-only in this pattern — wrapped returns `result`, not post-hook return); (2) **`create_react_agent` kwargs** — pass `pre_model_hook=` and `post_model_hook=` as kwargs, LangGraph calls them around every LLM step; post-hook CAN return a propagating partial update in this mode (covered next sub-topic). Source showed `graph.add_pre_model_hook()` / `builder.add_post_model_hook()` on `StateGraph` — these do not exist; hook kwargs belong to `create_react_agent`; source's actual working code used Pattern 1 correctly. Node.js/Express analogy: same shape (`req → middleware → handler → middleware → res`) but hooks operate on the full typed state dict, not HTTP objects. Project extended with `hooks.py` (`pre_summarize`, `post_summarize`, `wrap_with_hooks`) and `graph.py` updated to wrap the `summarize` node — `nodes.py` unchanged, which is the point. `hooks.py` detail: post-hook receives `{**state, **result}` (merged) so it can see the node's output; wrapper returns `result` so LangGraph reducer logic is unaffected.

- Module 3 > Getting Started with LangGraph: Building Your First AI Workflow Graph — **first code in the course; project named + scaffolded here.** Sub-topic *"What are Agents?"* compressed hard against Module 1 (it restates agent-as-LLM-defined-control-flow). **`Agent = LLM + Tools + Reasoning Loop + State`** — taught as four load-bearing terms, each with what breaks if dropped (LLM→hard-coded script · Tools→chatbot that can only talk · Loop→one-shot answer · State→amnesia). **The exit-condition contrast is the core framing:** a plain LLM call stops when *the text ran out*; an agent stops when *the goal was met* — and that single difference is what multi-step reasoning / error recovery / tool calls / long-running work / autonomous decisions all reduce to (one feature, four angles, not five bullets). Loop beats: `Reason → Act → Observe → Loop`. Flight-booking example ("cheapest flight to Dubai next Friday, notify on WhatsApp") used to make it concrete — **steps "compare prices" and "select an option" are decisions made after seeing data that didn't exist at request time**, which is why they couldn't be pre-written. Six characteristics (Autonomy · Reasoning · Tool Use · Memory · Modularity · Adaptability) taught as a table with a **cost column added** (unpredictability · latency+tokens · real side effects · storage+context pressure · interface to maintain · behaviour drift) and the point landed that *the cost column is the reason LangGraph exists* — every mechanism in the course buys a benefit while capping its cost.
  **The module's real payload = the Anatomy → LangGraph mapping table:** LLM Core → inside a node body (LangGraph never calls a model itself) · Prompt Template → inside a node body, `ChatPromptTemplate`/LCEL · Memory Module → a state channel (short-term) **+** checkpointer/store (long-term) · Toolset → nodes running tools, `@tool` wrapped or prebuilt `ToolNode` · State → the `TypedDict` passed to `StateGraph(...)` · Hooks & Transitions → edges + conditional edges. Three call-outs: the top two rows **are LangChain, not LangGraph** (2a's "LangChain inside a node" cashed in); "Memory Module" secretly covers two different things and that's where the short/long-term conflation starts; **Hooks & Transitions is the only row with no LangChain equivalent — which is why it gets its own module (4)**. "Why agents matter in LangGraph" (4 source claims) collapsed into an *agents-need-X → LangGraph-gives-X* table pointing back at 2b, whose real value is naming the **four node roles: planner / retriever / executor / evaluator** — the skeleton of nearly every agent graph.
  **Source correction made honestly:** the course's Research Assistant diagram (User Query → Planner → Search Tool → Summarizer → Evaluator → Result) has every arrow pointing down — **it is a chain, not an agent**; if the Evaluator finds only 1 source it has nowhere to send the work. Drew the version with the `evaluate → search` backward edge as what we're actually building toward, and framed v0.1 as deliberately shipping the straight line so mechanics land before branching.
  **Hands-on, taught in two steps.** (1) A no-LLM "hello graph" (`CounterState`, `double`/`describe`) to prove the machinery, with the four first-graph gotchas: nodes return **partial updates** (returning full state is the classic beginner habit) · **nodes never call each other, the edge knows** (the indirection that makes nodes swappable) · **`START`/`END` are real nodes** — `add_edge(START, x)` is what sets the entry point, there's no implicit first-node-wins · **`compile()` is a real validation step**, catches unreachable nodes/bad edge targets/no path from START, and is also where checkpointers attach later. LCEL contrast made explicit: **`a | b` fuses "what runs" with "what runs next"; `add_node` + `add_edge` split them** — two lines now buys cycles later. (2) Project v0.1. **Visualisation: `graph.get_graph().draw_mermaid()`** returns Mermaid text with zero extra deps (`draw_mermaid_png()` needs a renderer) — an architecture diagram that *cannot drift from the code because it is the code*; this is the real tool 2b's `networkx` snippet was only miming. Two `invoke` details taught before they bite: **`TypedDict` isn't runtime-enforced** so keys no node has written simply don't exist → use `state.get(k, default)`; and **seed reducer channels explicitly** (`"sources": []`). Five use cases (customer support · enterprise knowledge retrieval · research assistance · workflow automation · education/tutoring) taught via a graph-shape column, landing: **five domains, ~four shapes — route / loop / delegate / remember** — which is why the course teaches the graph, not a catalogue of agents. Closed with a first-graph mistakes table (KeyError → `.get()` · state resetting → full-state return or missing reducer · never terminates → no edge to END · unreachable node → forgot `add_edge(START, ...)` · parallel unreduced writes raise **by design**).

- Module 5.3 > Time Travel — **`checkpoint_id` is an address; passing it back to the graph re-runs forward from that snapshot.** `get_state_history` read with intent: find `snap.next == ("search",)` — the checkpoint after `understand` ran, before first `search` ran. Three operations: (1) replay: `invoke(None, config=snap.config)` — same state, same path, same result, useful for debugging/auditing; (2) time travel: `update_state(snap.config, values, as_node="understand")` writes a new checkpoint on top of the fork point and returns its config → `invoke(None, config=returned_config)` re-runs with modified state; (3) fork model: original checkpoints stay, new branch checkpoints added to the same thread; `checkpoint_id` is the address of the branch point. `update_state` `as_node` parameter: determines which node "made" the update, which sets `next` for the new checkpoint; `as_node="understand"` → `next=("search",)` because `understand → search` is a fixed edge. Source corrected: `get_state_history` returns `StateSnapshot` objects (not tuples/dicts); source's "time travel" demo doesn't re-run nodes because `update_state` was called on a finished thread (`next=()`) — nodes only re-run if you fork from a checkpoint where `next` is non-empty. **Project → v0.5**: `run_v5_time_travel()` added to `main.py` — original run + find fork point + inject `["checkpointing", "persistence", "state", "channels", "reducers"]` via `update_state` + compare: original finds 4 docs (ok), time-travel finds 3 different docs (give_up). No other file changes.

- Module 5.2 > Persistence — **the mechanism that makes every other Module 5 feature possible.** Framed against v0.3's silent problem: a crash between supersteps 4 and 5 throws away attempt 1's three documents. **Checkpointer** writes full state to storage at the end of every superstep (the Pregel barrier — not per-node; per-superstep because that is the only consistent moment). Each saved snapshot is a **`StateSnapshot`**: five fields — `values` (the state dict) · `next` (nodes queued to run next, empty tuple at END) · `config` · `metadata` (step number, source, writes) · `tasks` (what ran, error info if step failed). **Thread** = named container of checkpoints; all invocations with the same `thread_id` share a checkpoint history; one thread = one conversation or task instance. Three checkpointer implementations: `InMemorySaver` (RAM, dev/test) → `SqliteSaver` (SQLite file, single-machine prod) → `PostgresSaver` (distributed prod) — all implement the same interface, switching is one import change. The **two-line mechanic**: `compile(checkpointer=...)` + `config={"configurable": {"thread_id": "..."}}` on every invoke/stream call. **Configuration vs state**: thread_id / user_id / model choice belong in `config["configurable"]`, not in the state TypedDict — they are out-of-band parameters the checkpointer reads automatically; nodes should not plumb them through state. **Reading state**: `graph.get_state(config)` returns the latest `StateSnapshot`; `graph.get_state_history(config)` is a generator yielding all snapshots newest-first — this is the raw data that Time Travel (5.3) navigates. **Single-turn (fault tolerance)** vs **multi-turn (conversational continuity)**: same mechanism, different invocation pattern; for fault tolerance re-invoke same thread_id with `{}` input; for multi-turn re-invoke same thread_id with a partial update. **The merge step on resume**: LangGraph loads checkpoint values as base state, then applies each input key's reducer against the base — default reducer overwrites (question replaces), `operator.add` accumulates (sources stack, attempts sum); keys absent from new input are unchanged. **Ephemeral state pattern**: final node resets per-turn keys before END so the next multi-turn invocation starts clean (not needed for Research Assistant since we use one thread per question). **Source corrected**: course examples show nodes mutating state in place and returning full dict — this was corrected in 5.1 (return partial dicts to avoid duplication under add reducers). Also: `MemorySaver` is a legacy alias for `InMemorySaver`, both work. **Project → v0.4**: `build_graph(checkpointer=None)` — one-line change, backward-compatible; `run_v4()` in main.py demonstrates InMemorySaver + thread_id + get_state + get_state_history.

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

- Module 5.4 > Tool — **Two patterns for calling a tool in a custom `StateGraph`.** Framed against the corpus.py debt carried since Module 3. **What a tool is in LangGraph**: `@tool` gives a function a name, description, and typed schema the LLM can read; the docstring is prompt engineering inside the tool definition; return strings (not dicts) so ToolMessage content is LLM-readable. **Pattern A (direct call)**: node calls `tool.invoke(args)` explicitly — no LLM involved in tool selection. Right choice when the workflow already knows which tool to call and with what. The Research Assistant's `search` node is this. **Pattern B (LLM-driven via ToolNode)**: `llm.bind_tools([...])` → LLM produces `AIMessage` with `tool_calls` → `tools_condition` routes to `ToolNode` → `ToolNode` runs tools in parallel and returns `ToolMessages` → loop back to LLM. Right choice when the LLM needs to decide which tool to use. **`ToolNode` internals**: reads `state["messages"][-1].tool_calls`, dispatches each by name in parallel (one Pregel super-step), wraps results in `ToolMessage` with matching `tool_call_id`, returns `{"messages": [...]}`. Requires `messages: Annotated[list, add_messages]` in state. `handle_tool_errors=True` (default) — exceptions become ToolMessage content so the LLM can recover without graph-level error handling. **`tools_condition`**: prebuilt router from `langgraph.prebuilt`; returns `"tools"` if last message has `.tool_calls`, `"__end__"` otherwise; maps to `{"tools": tool_node, "__end__": END}` in path_map; pure router, never writes to state. **The Pattern B graph built explicitly** — same two-node loop `create_react_agent` builds internally: `START → agent → tools_condition → tools → agent → ...` with `tools → agent` being the backward edge. **Tool invocation lifecycle mapped to LangGraph**: Intent Recognition = `AIMessage` with `tool_calls` populated; Tool Selection = `tool_calls[i]["name"]`; Input Preparation = `tool_calls[i]["args"]`; Execution = `ToolNode` dispatches in parallel; Result Handling = `ToolMessage` appended to `messages`. **Three integration patterns from the source**: Single Tool Node (Pattern A, one tool always called) · Tool Router Node (Pattern B, LLM picks from multiple tools, `tools_condition` is the router) · Tool + Memory Node (tool result also written to long-term `store`, Module 5.6). **Project → v0.6**: new `tools.py` with `@tool search_docs` (moved from `agent.py`, description improved); `agent.py` imports from `tools.py`; new `tools_demo.py` shows Pattern B end-to-end; `main.py` adds `run_tools_demo()`. Main graph (`graph.py`, `nodes.py`) unchanged — Pattern A is correct when there's one tool and the workflow controls which args to pass.

## Next Up
**Module 5.5 — Interrupts** (45m). Pauses the graph at a node boundary for human approval before resuming. Uses `compile(interrupt_before=["node_name"])` + checkpointer + `invoke(None, config=...)` to resume. The Research Assistant will pause before summarizing so the user can approve the sources found.

Remaining Module 5 order after that: Interrupts (45m) · Memory (50m) · LangGraph APIs — Functional and Graph API (10m, mostly a recap of 2b's `@entrypoint`/`@task` note).

Curriculum is locked into `CLAUDE.md` from the syllabus screenshots — 6 modules, numbering final, **10h 20m of teaching content**. Prelude, Exercises, and all Quizzes are **intentionally out of scope** by the learner's choice; that accounts for the gap against the course's stated 15h 30m. Don't flag it, don't ask for more screenshots of it, don't write quiz questions unasked.
