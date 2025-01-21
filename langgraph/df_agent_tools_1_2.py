from langgraph.prebuilt import ToolNode, ToolExecutor
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage, SystemMessage, FunctionMessage
from typing_extensions import TypedDict, List, Literal,  Annotated, Sequence
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import pandas as pd
import operator
from rich import print
import re
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64
import os
import ast
import json

from synthetic_df import gen_synthetic_df


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    dataframe: str
    query: str


def create_system_message(state: AgentState):
    query = state["query"]  # Extract query from the state
    df_json = state["dataframe"]  # JSON string of the DataFrame
    print("DataFrame JSON:", df_json)  # Debugging step to inspect JSON data

    # Directly read the JSON string
    sample_df = pd.read_json(df_json, orient="split")
    sample_df_head = sample_df.head().to_string()
    print(sample_df_head)
    system_prompt = f"""
        You are an expert Python developer and data analyst. 
        Based on the user's query and the provided DataFrame sample, 
        generate Python function code to perform the requested analysis.

        User Query: {query}
        Sample DataFrame used only to infer the structure of the DataFrame:
        {sample_df_head}

        **IMPORTANT**:
        - Always call the appropriate tool by returning a `function_call`:
            - Use `execute_code_tool` for Python code that processes data.
            - Use `execute_plot_tool` for Python code that creates visualizations or plots.
        - Include the generated Python code in the `arguments` of the `function_call`.

        Return only the `function_call` and ensure it is correctly structured for the tool.
        """
    return {"messages": [SystemMessage(content=system_prompt)]}

def call_model(state):
    messages = state["messages"]  # Ensure system message is included
    response = llm_tools.invoke(messages)
    print("Agent Response:", response)
    return {"messages": messages + [response]}

# def extract_function_body(code: str, function_name: str) -> str:
#     # Parse the code into an Abstract Syntax Tree (AST)
#     tree = ast.parse(code)
    
#     # Find the function definition node
#     for node in ast.walk(tree):
#         if isinstance(node, ast.FunctionDef) and node.name == function_name:
#             # Extract the function body
#             function_body = ast.get_source_segment(code, node)
#             return function_body
    
#     return None

def extract_function_body(code: str, function_name: str) -> str:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(code, node)
    raise ValueError("Function not found.")


@tool
def execute_code_tool(dataframe_json: str, generated_code: str) -> str:
    """
    Executes dynamically generated Python code on a provided dataframe in JSON format.

    Args:
        dataframe_json (str): JSON string representing the input dataframe.
        generated_code (str): The Python code, returned as a string, defining the function to be executed.

    Returns:
        str: The result of executing the function, which must be a string, number, or list.
    """
    # Deserialize the JSON string to a DataFrame
    dataframe = pd.read_json(dataframe_json)

    # Extract and execute the function
    function_match = re.search(r"(def\s+\w+\(.*?\):\n(?:\s+.*\n)*)", generated_code, re.DOTALL)
    if not function_match:
        raise ValueError("Error: Could not extract a valid function definition from the generated code.")
    
    function_body = function_match.group(1)
    namespace = {}
    exec(function_body, namespace)
    
    function_name = re.search(r"def\s+(\w+)\(", function_body).group(1)
    result = namespace[function_name](dataframe)
    
    if not isinstance(result, (str, int, float, list)):
        raise ValueError("The result must be a string, number, or list.")
    
    return json.dumps(result)  # Ensure output is JSON-compatible


@tool
def execute_plot_tool(dataframe_json: str, generated_code: str) -> str:
    """
    Executes dynamically generated Python code to create a plot from a provided dataframe in JSON format.

    Args:
        dataframe_json (str): JSON string representing the input dataframe.
        generated_code (str): The Python code, returned as a string, defining the function that generates the plot.

    Returns:
        str: A base64-encoded string representing the generated plot image.
    """
    # Deserialize the JSON string to a DataFrame
    dataframe = pd.read_json(dataframe_json)

    # Extract and execute the function
    function_match = re.search(r"(def\s+\w+\(.*?\):\n(?:\s+.*\n)*)", generated_code, re.DOTALL)
    if not function_match:
        raise ValueError("Error: Could not extract a valid function definition from the generated code.")
    
    function_body = function_match.group(1)
    namespace = {}
    exec(function_body, namespace)
    
    function_name = re.search(r"def\s+(\w+)\(", function_body).group(1)
    img_base64 = namespace[function_name](dataframe)
    
    return img_base64  # Ensure output is a base64 string



tools = [execute_code_tool, execute_plot_tool]
tool_node = ToolNode(tools)

llm = ChatOllama(model="granite3.1-dense:8b", temperature=0)
llm_tools = llm.bind_tools(tools)


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Add nodes to the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("generate_prompt", create_system_message)
graph.add_node("query_model", call_model)
graph.add_node("tools", tool_node)

#
graph.set_entry_point("generate_prompt")  # First node in the graph
graph.add_edge("generate_prompt", "query_model")
graph.add_conditional_edges(
    "query_model",  # Start node
    should_continue,  # Function to decide which node to go to next
    # {
    #     "execute_code": "execute_code_tool",  # Route to code execution
    #     "execute_plot": "execute_plot_tool",  # Route to plot execution
    # },
    ["tools", END]
)
graph.add_edge("tools", "query_model")


# Compile the graph
app = graph.compile()

example_df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Score": [90, 85, 88]
})
df = gen_synthetic_df()
df_json = df.to_json(orient="split")

# user_query = "Calculate the average score for users older than 28."
user_query = "Plot all streams with their durations for the planned downtime category."
initial_state = {
    "messages": [],
    "dataframe": df_json,  # Pass DataFrame as JSON string
    "query": user_query
}

result = app.invoke(initial_state)

print("Generated Python Function:")
print(result["messages"][-1].content)
