import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import create_react_agent

from .tools import search_docs

load_dotenv()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _get_model():
    endpoint = HuggingFaceEndpoint(
        repo_id=MODEL,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        max_new_tokens=512,
        temperature=0.3,
    )
    return ChatHuggingFace(llm=endpoint)


def _pre_hook(state: dict) -> dict | None:
    messages = state.get("messages", [])
    print(f"\n[PRE-HOOK]  agent step — {len(messages)} message(s) in state")
    return None


def _post_hook(state: dict) -> dict | None:
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "")
        print(f"[POST-HOOK] {type(last).__name__} — {len(str(content))} chars")
    return None


def build_agent():
    return create_react_agent(
        model=_get_model(),
        tools=[search_docs],
        pre_model_hook=_pre_hook,
        post_model_hook=_post_hook,
    )


def run(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Research the following question and provide a detailed answer with citations:\n\n"
            + question
        ))]
    })
    return result["messages"][-1].content
