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
