from langgraph.checkpoint.memory import InMemorySaver

from research_assistant import agent as v2
from research_assistant import controller_demo
from research_assistant.graph import build_graph

QUESTION = "Explain LangGraph architecture using 3 reliable sources."


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


def main():
    run_v4()
    run_v3()
    run_v2()
    run_controller_demo()


if __name__ == "__main__":
    main()
