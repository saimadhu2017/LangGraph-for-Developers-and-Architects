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
