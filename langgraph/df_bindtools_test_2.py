from langgraph.prebuilt import ToolNode, ToolExecutor
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage, SystemMessage, FunctionMessage, AIMessage
from typing_extensions import TypedDict, List, Literal,  Annotated, Sequence
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
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
import os
from dotenv import load_dotenv

from synthetic_df import gen_synthetic_df

env = load_dotenv(override=True)
print(env)
print(os.getenv("GROQ_API_KEY"))


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    dataframe_json: str
    query: str
    generated_code: str



def create_system_message(state: AgentState):
    query = state["query"]  # Extract query from the state
    df_json = state["dataframe_json"]  # JSON string of the DataFrame
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
        """
    return {"messages": [SystemMessage(content=system_prompt)]}


def call_model(state):
    messages = state["messages"]  # Ensure system message is included
    response = llm_tools.invoke(messages)
    print("Agent Response:", response)
    return {"messages": messages + [response]}



@tool
def execute_code_tool(state: AgentState) -> str:
    """
    Executes dynamically generated Python code on a provided dataframe in JSON format.

    Args:
        dataframe_json (str): JSON string representing the input dataframe.
        generated_code (str): The Python code, returned as a string, defining the function to be executed.

    Returns:
        str: The result of executing the function, which must be a string, number, or list.
    """
    # Deserialize the JSON string to a DataFrame
    dataframe = pd.read_json(state["dataframe_json"])
    generated_code = state["generated_code"]

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


tools = [execute_code_tool]
tool_node = ToolNode(tools)


llm = ChatGroq(model="llama-3.3-70b-versatile", api_key = "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")
# llm = ChatOllama(model="granite3.1-dense:8b", temperature=0) #llama3.1:latest granite3.1-dense:8b qwen2.5-coder:14b  jacob-ebey/phi4-tools deepseek-r1:14b 
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
user_query = "Average downtime by category."
initial_state = {
    "messages": [],
    "dataframe_json": df_json,  # Pass DataFrame as JSON string
    "query": user_query,
    "generated_code": ""
}

result = app.invoke(initial_state)

print("Generated Python Function:")
print(result["messages"][-1].content)
