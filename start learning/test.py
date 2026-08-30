from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. The state schema. This is the contract every node reads from and writes to.
class CounterState(TypedDict):
    value: int
    log: str


# 2. A node is a plain function: state in, PARTIAL update out.
def double(state: CounterState) -> dict:
    return {"value": state["value"] * 2}


def describe(state: CounterState) -> dict:
    return {"log": f"final value is {state['value']}"}


# 3. Declare the structure.
builder = StateGraph(CounterState)
builder.add_node("double", double)
builder.add_node("describe", describe)

builder.add_edge(START, "double")
builder.add_edge("double", "describe")
builder.add_edge("describe", END)

# 4. Compile — this validates the graph and produces something runnable.
graph = builder.compile()

# 5. Run it.
print(graph.invoke({"value": 21, "log": ""}))
# {'value': 42, 'log': 'final value is 42'}
