"""Module 5.1 demo — the router's answer comes from the LLM, not from an `if`.

In graph.py the router is deterministic Python: it reads `verdict` and picks a label.
Here the router reads a label the LLM wrote into state. Same machinery, different
author of the decision — this is Module 1's "partial control flow definition" at its
most literal: we still declare the three destinations, the LLM only picks among them.

Rebuilt from the course's AWS Bedrock example on the HuggingFace client this project
already uses. Three things were changed on purpose:
  1. Nodes return PARTIAL UPDATES instead of mutating and returning the whole state.
  2. `messages` uses the `add_messages` reducer instead of a hand-rolled list append.
  3. The LLM's free-text answer is normalised before it is allowed to route.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .nodes import MODEL, get_client


class ControllerState(TypedDict):
    user_input: str
    summary: str
    messages: Annotated[list, add_messages]   # appends; never replaces history
    decision: str                             # "summarize" | "validate" | "end"


def _ask(prompt: str, max_tokens: int = 256) -> str:
    response = get_client().chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


CHOICES = ("summarize", "validate", "end")


# ── Node 1: the LLM decides what should happen next ──────────────────────
def controller_node(state: ControllerState) -> dict:
    prompt = f"""You are an AI workflow controller.
User said: {state["user_input"]}
Already summarised: {"yes" if state.get("summary") else "no"}

Decide the next step in the workflow.
Options:
1. summarize - if the user input is long text and has not been summarised
2. validate  - if there is already a summary
3. end       - if no action is needed

Answer with ONLY one word: summarize, validate, or end."""

    raw = _ask(prompt, max_tokens=5).lower()

    # NEVER route on raw model output. An LLM that replies "Summarize!" or
    # "I would summarize this" must not be able to crash the graph with a
    # KeyError on the path_map. Unrecognised answers fall back to a safe exit.
    decision = next((c for c in CHOICES if c in raw), "end")

    return {
        "decision": decision,
        "messages": [AIMessage(content=f"Decision: {decision}", name="controller")],
    }


# ── Node 2: summarise ────────────────────────────────────────────────────
def summarizer_node(state: ControllerState) -> dict:
    summary = _ask(f"Summarize this text briefly:\n\n{state['user_input']}")
    return {
        "summary": summary,
        "messages": [
            AIMessage(
                content=f"Summary: {summary}",
                name="summarizer",
                additional_kwargs={"node": "summarizer", "type": "summary"},
            )
        ],
    }


# ── Node 3: validate ─────────────────────────────────────────────────────
def validator_node(state: ControllerState) -> dict:
    summary = state.get("summary", "")
    verdict = _ask(
        f"Is the following summary under 80 words? Reply only with Yes or No.\n\n{summary}",
        max_tokens=5,
    )
    return {
        "messages": [
            AIMessage(
                content=f"Validation result: {verdict}",
                name="validator",
                additional_kwargs={"node": "validator", "type": "validation"},
            )
        ]
    }


def build_controller_graph():
    builder = StateGraph(ControllerState)

    builder.add_node("controller", controller_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("validator", validator_node)

    builder.add_edge(START, "controller")

    # The router here is a one-liner because controller_node already did the
    # thinking and parked its answer in state. Routers should stay this thin:
    # do the work in a node, decide in the edge.
    builder.add_conditional_edges(
        "controller",
        lambda state: state["decision"],
        {"summarize": "summarizer", "validate": "validator", "end": END},
    )

    builder.add_edge("summarizer", "validator")
    builder.add_edge("validator", END)

    return builder.compile()


TEXT = (
    "LangGraph is a Python framework that helps developers build complex, "
    "stateful, multi-step AI agents using a directed graph of nodes."
)


def run(text: str = TEXT) -> dict:
    app = build_controller_graph()
    return app.invoke({
        "user_input": text,
        "summary": "",
        "messages": [HumanMessage(content=text)],
        "decision": "",
    })
