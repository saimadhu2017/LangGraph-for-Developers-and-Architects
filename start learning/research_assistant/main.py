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
