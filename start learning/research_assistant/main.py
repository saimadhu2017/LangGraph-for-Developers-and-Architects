from research_assistant import agent as v2
from research_assistant.graph import build_graph

QUESTION = "Explain LangGraph architecture using 3 reliable sources."


def run_v1():
    print("=" * 60)
    print("v0.1 — Custom StateGraph (4 explicit nodes + hooks)")
    print("=" * 60)
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())
    final = graph.invoke({"question": QUESTION, "sources": []})
    print("\nKEYWORDS:", final["keywords"])
    print("\nSOURCES:")
    for s in final["sources"]:
        print("  -", s["title"])
    print("\nSUMMARY:\n", final["summary"])
    print("\nVERDICT:", final["verdict"])


def run_v2():
    print("\n" + "=" * 60)
    print("v0.2 — Prebuilt create_react_agent (2-node ReAct loop + hooks)")
    print("=" * 60)
    answer = v2.run(QUESTION)
    print("\nANSWER:\n", answer)


def main():
    run_v1()
    run_v2()


if __name__ == "__main__":
    main()
