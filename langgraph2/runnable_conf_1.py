from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import operator

# Step 1: Define the state schema
class WorkflowState(TypedDict):
    messages: Annotated[list[str], operator.add]

# Step 2: Define the tools (as graph nodes)
def greet_tool(state):
    return {"messages": ["Hello, world!"]}

def append_tool(state):
    last_message = state["messages"][-1]
    return {"messages": [f"{last_message} This is a LangGraph example."]}

# Step 3: Create the graph
graph = StateGraph(WorkflowState)
graph.add_node("greet", greet_tool)  # First tool
graph.add_node("append", append_tool)  # Second tool

# Link the two tools
graph.add_edge("greet", "append")

# Set the entry point and compile the graph
graph.set_entry_point("greet")
workflow_app = graph.compile()

# Step 4: Define a custom RunnableConfig
class CustomRunnableConfig:
    def __init__(self):
        self.parent_run_id = None  # Required for internal usage

    def on_run(self, node_name, result):
        print(f"Node '{node_name}' output: {result}")

runnable_config = CustomRunnableConfig()

# Step 5: Run the workflow
inputs = {"messages": []}  # Initial empty state
final_output = workflow_app.invoke(inputs, config={"callbacks": runnable_config.on_run})

# Print the final output
print("Final Output:", final_output)
