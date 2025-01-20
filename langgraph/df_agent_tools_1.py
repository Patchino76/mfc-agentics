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
from io import BytesIO
import base64
import os
import ast

from synthetic_df import gen_synthetic_df


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    dataframe: pd.DataFrame
    query: str

llm = ChatOllama(model="granite3.1-dense:8b", temperature=0)


def create_system_message(state):
    query = state["query"]  # Extract query from the state
    sample_df = state["dataframe"].head().to_string()  # Use the head of the DataFrame for structure
    system_prompt = f"""
    You are an expert Python developer and data analyst. 
    Based on the user's query and the provided DataFrame sample, 
    generate Python function code to perform the requested analysis. 

    User Query: {query}
    Sample DataFrame used only to infer the structure of the DataFrame:
    {sample_df}

    Provide the Python function as a single string that can be executed using the exec function. 
    The function should accept a pd.DataFrame object with the same structure as {sample_df} as a parameter. 
    Return only the Python function as a string and do not try to execute the code. 
    Do not add sample dataframes, function descriptions and do not add calls to the function.
    Return a pure function with no additional code or descriptions outside the function definition.
            
    If you create a plot function, do not use plt.show(), instead return the image in base64 format using the base64 and BytesIO libraries.
    If returning a base64 string do not add 'data:image/png;base64' to it.
    Ensure that matplotlib.pyplot, base64 and BytesIO are imported within the dynamically generated function definition.
    Finaly decode the image with utf-8 like this:
        img = BytesIO() 
        plt.savefig(img, format='png') 
        img.seek(0) 
        # Encode the image to base64 
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        return img_base64
        DO NOT ADD ANY SYMBOLS AFTER THE RETURN STATEMENT !!!
    """
    return {"messages": [SystemMessage(content=system_prompt)]}

def call_model(state):
    messages = state["messages"]  # Ensure system message is included
    response = llm.invoke(messages)
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
def execute_code_tool(dataframe: pd.DataFrame, generated_code: str) -> str:
    """
    Executes dynamically generated Python code on a provided dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe on which the Python function will operate.
        generated_code (str): The Python code, returned as a string, defining the function to be executed.

    Returns:
        str: The result of executing the function, which must be a string, number, or list.

    Raises:
        ValueError: If the generated code does not contain a valid function definition or the result 
                    of the function execution is not a valid type.
    """
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
    
    return str(result)


@tool
def execute_plot_tool(dataframe: pd.DataFrame, generated_code: str) -> str:
    """
    Executes dynamically generated Python code to create a plot from a provided dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe on which the Python function will operate.
        generated_code (str): The Python code, returned as a string, defining the function that generates the plot.

    Returns:
        str: A base64-encoded string representing the generated plot image.

    Raises:
        ValueError: If the generated code does not contain a valid function definition or the function execution fails.
    """
    # Extract and execute the function
    function_match = re.search(r"(def\s+\w+\(.*?\):\n(?:\s+.*\n)*)", generated_code, re.DOTALL)
    if not function_match:
        raise ValueError("Error: Could not extract a valid function definition from the generated code.")
    
    function_body = function_match.group(1)
    namespace = {}
    exec(function_body, namespace)
    
    function_name = re.search(r"def\s+(\w+)\(", function_body).group(1)
    img_base64 = namespace[function_name](dataframe)
    
    return img_base64

def decide_action(state):
    query = state["query"].lower()
    if "plot" in query or "chart" in query or "graph" in query:
        return "execute_plot"
    else:
        return "execute_code"



tools = [execute_code_tool, execute_plot_tool]
tool_node = ToolNode(tools)

# Add nodes to the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("generate_prompt", create_system_message)
graph.add_node("query_model", call_model)

# Replace ToolExecutor with ToolNode
graph.add_node("execute_code", ToolNode(execute_code_tool))
graph.add_node("execute_plot", ToolNode(execute_plot_tool))

# Add conditional edges for deciding which tool to execute
graph.add_conditional_edges(
    "query_model",  # Start node
    decide_action,  # Function to decide which node to go to next
    {
        "execute_code": "execute_code",  # Route to code execution
        "execute_plot": "execute_plot",  # Route to plot execution
    },
)

# Set entry and finish points
graph.set_entry_point("generate_prompt")  # First node in the graph
graph.set_finish_point("query_model")     # Use the final logical node as the finish point

# Compile the graph
app = graph.compile()

example_df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Score": [90, 85, 88]
})
df = gen_synthetic_df()

user_query = "Calculate the average score for users older than 28."
# user_query = "Plot all streams with their durations for the planned downtime category."
initial_state = {
    "messages": [],
    "dataframe": example_df,
    "query": user_query
}

result = app.invoke(initial_state)

print("Generated Python Function:")
print(result["messages"][-1].content)
