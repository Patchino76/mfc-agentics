from typing import TypedDict, Annotated
from operator import add

class PizzaState(TypedDict):
    toppings: Annotated[list[str], add]
    quantity: int


def add_cheese(state:PizzaState):
    quantity = state['quantity'] + 1
    return {'toppings' : ["cheese"], 'quantity' : quantity}

def add_meat(state:PizzaState):
    quantity = state['quantity'] + 1
    return {'toppings' : ["meat"], 'quantity' : quantity}

from langgraph.graph import StateGraph

graph = StateGraph(PizzaState)
graph.add_node("cheese", add_cheese)
graph.add_node("meat", add_meat)
graph.add_edge("cheese", "meat")
graph.set_entry_point("cheese")
graph.set_finish_point("meat")

app = graph.compile()

initial_state = {
    'toppings' : [],
    'quantity' : 0
}

res = app.invoke(initial_state)

print(res)