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


def build_graph(checkpointer=None, interrupt_before=None):
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
    # v0.7 — optional interrupt_before. None = no interrupts (all prior versions unaffected).
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
