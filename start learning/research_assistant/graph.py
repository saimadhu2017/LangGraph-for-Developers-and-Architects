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
