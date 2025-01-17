from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import random

class State(TypedDict):
    graph_state: str

def node1(state: State):
    print("Node 1")
    return {"graph_state" : state["graph_state"] + " I am"}

def node2(state: State):
    print("Node 2")
    return {"graph_state" : state["graph_state"] + " happy"}

def node3(state: State):
    print("Node 3")
    return {"graph_state" : state["graph_state"] + " sad"}

def decide_node(state: State) -> Literal["node2",  "node3"]:
    if rnd := random.random() < 0.5:
        return "node2"
    return "node3"


builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)
# builder.add_node("decide_node", decide_node)

builder.add_conditional_edges("node1", decide_node, {"node3": "node3", "node2": "node2"})

builder.set_entry_point("node1")
builder.set_finish_point("node2")
builder.set_finish_point("node3")
app = builder.compile()

initial_state = {
    "graph_state" : "Hi, I am Svetlio and "
}

result = app.invoke(initial_state) # returns the final graph state
print(result)

