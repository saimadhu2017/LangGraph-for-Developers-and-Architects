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
# THE ANALYST: reads the question, strips filler words, writes keywords to the whiteboard.
# No LLM. Pure string processing — fast and cheap.
def understand(state: ResearchState) -> dict:
    words = (w.strip("?.,\"'").lower() for w in state["question"].split())
    keywords = [w for w in words if w and w not in STOPWORDS and len(w) > 2]
    return {"keywords": keywords}


# ── retriever ────────────────────────────────────────────────────────────
# THE LIBRARIAN: reads keywords from the whiteboard, searches the filing cabinet,
# writes the best matching documents to `sources`.
#
# v0.3 — this node now runs MORE THAN ONCE, so it has to know which pass it is on.
# It finds that out from state (`attempts`), not from a variable it kept in memory:
# nodes are stateless functions, the graph carries the state.
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
# THE WRITER: reads the question and sources, calls the LLM, writes a cited answer.
# Uses InferenceClient directly (provider="novita") to bypass router compatibility
# issues in the current langchain-huggingface version. Same logic as an LCEL chain.
#
# v0.3 — this runs again after every retry, on the LARGER source set.
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
# THE EDITOR: counts the sources and writes a verdict — "ok" or "weak".
#
# v0.3 — the verdict is finally READ by something. `route_after_evaluate` in graph.py
# turns this string into the next node. The node still does not know where the work
# goes next; it only reports what it found. Deciding is the edge's job.
def evaluate(state: ResearchState) -> dict:
    count = len(state["sources"])
    if count >= MIN_SOURCES:
        return {"verdict": f"ok — {count} sources"}
    return {"verdict": f"weak — only {count} source(s), wanted {MIN_SOURCES}"}
