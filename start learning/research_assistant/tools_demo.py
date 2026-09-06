"""Module 5.4 demo — Pattern B: LLM-driven tool use via ToolNode + tools_condition.

Builds the same loop that create_react_agent builds automatically,
but explicitly, so every seam is visible.
"""
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
