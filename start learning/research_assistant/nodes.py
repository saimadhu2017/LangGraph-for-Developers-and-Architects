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
# THE ANALYST: reads the question, strips filler words, writes keywords to the whiteboard.
# No LLM. Pure string processing — fast and cheap.
def understand(state: ResearchState) -> dict:
    words = (w.strip("?.,\"'").lower() for w in state["question"].split())
    keywords = [w for w in words if w and w not in STOPWORDS and len(w) > 2]
    return {"keywords": keywords}


# ── retriever ────────────────────────────────────────────────────────────
# THE LIBRARIAN: reads keywords from the whiteboard, searches the filing cabinet,
# writes the best matching documents to `sources`.
def search(state: ResearchState) -> dict:
    found = search_corpus(state["keywords"])
    # `sources` has an `operator.add` reducer, so this APPENDS.
    return {"sources": found}


# ── executor ─────────────────────────────────────────────────────────────
# THE WRITER: reads the question and sources, calls the LLM, writes a cited answer.
# Uses InferenceClient directly (provider="together") to bypass router compatibility
# issues in the current langchain-huggingface version. Same logic as an LCEL chain.
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
# THE EDITOR: counts the sources. Writes a verdict — "ok" or "weak".
# Right now nobody acts on the verdict. In Module 5, a "weak" verdict will send
# the Librarian back to search again. That one backward edge turns this into an agent.
def evaluate(state: ResearchState) -> dict:
    count = len(state["sources"])
    if count >= MIN_SOURCES:
        return {"verdict": f"ok — {count} sources"}
    return {"verdict": f"weak — only {count} source(s), wanted {MIN_SOURCES}"}
