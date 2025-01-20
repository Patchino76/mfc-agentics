from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage, SystemMessage, FunctionMessage
import requests
# from typing import List, Literal,  Annotated, Sequence
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

# class AngetAtate(TypedDict):
#     dataframe: pd.DataFrame
#     query: str
#     generated_code: str
#     result: str
#     logs: Annotated[list[str], add]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    dataframe: pd.DataFrame
    query: str

llm = ChatOllama(model="granite3.1-dense:8b", temperature=0)

#@tool
# def execute_generated_code(state):
#     messages = state["messages"]
#     dataframe = state["dataframe"]
    
#     # Extract the generated code from the last message
#     generated_code = messages[-1].content.strip()
#     print("Generated Python Function:")
#     print(generated_code)

#     # Define a namespace for executing the code
#     namespace = {}
#     try:
#         # Execute the generated code
#         exec(generated_code, namespace)
#         # Assume the function is named `generated_function` as per the prompt
#         result = namespace["generated_function"](dataframe)
#         print("Result:")
#         print(result)
        
#         # Ensure the result is of an allowed type
#         if not isinstance(result, (str, int, float, list)):
#             raise ValueError("The result must be a string, number, or list.")
        
#         # Add the result as a new FunctionMessage
#         return {"messages": messages + [FunctionMessage(content=str(result), name="execute_code")]}
#     except Exception as e:
#         # Handle any execution errors
#         error_message = f"Error during code execution: {str(e)}"
#         return {"messages": messages + [FunctionMessage(content=error_message, name="execute_code")]}


# tools = [execute_generated_code]
# tool_node = ToolNode(tools)


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
            
    If you create a plot function, do not use plt.show(), instead return the image in base64 format using the base64 and BytesIO libraries.
    If returning a base64 string do not add 'data:image/png;base64' to it.
    """
    return {"messages": [SystemMessage(content=system_prompt)]}

def call_model(state):
    messages = state["messages"]  # Ensure system message is included
    response = llm.invoke(messages)
    return {"messages": messages + [response]}


def execute_generated_code(state):
    messages = state["messages"]
    dataframe = state["dataframe"]
    
    # Extract the generated code from the last message
    generated_code = messages[-1].content.strip()

    # Extract the full function definition using a regex
    function_match = re.search(r"(def\s+\w+\(.*?\):\n(?:\s+.*\n)*)", generated_code, re.DOTALL)
    if not function_match:
        error_message = "Error: Could not extract a valid function definition from the generated code."
        return {"messages": messages + [FunctionMessage(content=error_message, name="execute_code")]}
    
    # Get the full function body
    function_body = function_match.group(1)
    print("Extracted Function Body:")
    print(function_body)

    # Define a namespace for executing the code
    namespace = {}
    try:
        # Execute the extracted function
        exec(function_body, namespace)
        
        # Extract the function name from the function body
        function_name_match = re.search(r"def\s+(\w+)\(", function_body)
        if not function_name_match:
            raise ValueError("Could not determine the function name.")
        function_name = function_name_match.group(1)
        print("Extracted Function Name:")
        print(function_name)
        
        # Dynamically call the function
        result = namespace[function_name](dataframe)
        
        # Ensure the result is of an allowed type
        if not isinstance(result, (str, int, float, list)):
            raise ValueError("The result must be a string, number, or list.")
        
        # Add the result as a new FunctionMessage
        return {"messages": messages + [FunctionMessage(content=str(result), name="execute_code")]}
    except Exception as e:
        # Handle any execution errors
        error_message = f"Error during code execution: {str(e)}"
        return {"messages": messages + [FunctionMessage(content=error_message, name="execute_code")]}

def execute_generated_plot(state):
    messages = state["messages"]
    dataframe = state["dataframe"]
    
    # Extract the generated code from the last message
    generated_code = messages[-1].content.strip()

    # Refine the regex to capture only the function body up to the last 'return'
    function_match = re.search(
        r"(def\s+\w+\(.*?\):\n(?:\s+.*?return\s+.*?\n?)*)",  # Capture function until the last `return`
        generated_code,
        re.DOTALL
    )
    if not function_match:
        error_message = "Error: Could not extract a valid function definition from the generated code."
        return {"messages": messages + [FunctionMessage(content=error_message, name="execute_plot")]}
    
    
    # Get the full function body
    function_body = function_match.group(1)
    print("Extracted Function Body:")
    print(function_body)

    # Define a namespace for executing the code
    namespace = {}
    try:
        # Execute the extracted function
        exec(function_body, namespace)
        
        # Extract the function name from the function body
        function_name_match = re.search(r"def\s+(\w+)\(", function_body)
        if not function_name_match:
            raise ValueError("Could not determine the function name.")
        function_name = function_name_match.group(1)
        print("Extracted Function Name:")
        print(function_name)
        
        # Dynamically call the function
        plt.figure()  # Create a new figure to avoid overlap
        namespace[function_name](dataframe)  # Assume the function generates the plot directly
        
        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()  # Close the plot to free memory

        output_file_path = os.path.join("outputs", "output.txt")
        with open(output_file_path, "w") as file:
            file.write(plot_base64)

        print(f"Base64 string saved to {output_file_path}")


        # Add the result as a new FunctionMessage
        return {"messages": messages + [FunctionMessage(content=plot_base64, name="execute_plot")]}
    except Exception as e:
        # Handle any execution errors
        error_message = f"Error during plot execution: {str(e)}"
        return {"messages": messages + [FunctionMessage(content=error_message, name="execute_plot")]}

# Add nodes to the graph
graph = StateGraph(AgentState)
# Add nodes to the graph
graph.add_node("generate_prompt", create_system_message)
graph.add_node("query_model", call_model)
# graph.add_node("execute_code", execute_generated_code)
graph.add_node("execute_plot", execute_generated_plot)

# Define edges
graph.add_edge("generate_prompt", "query_model")  # Prompt generation -> LLM call
# graph.add_edge("query_model", "execute_code")  # LLM call -> Code execution
graph.add_edge("query_model", "execute_plot")
graph.set_entry_point("generate_prompt")  # Start from prompt generation
graph.set_finish_point("execute_plot")  # End after the code execution

# Compile the graph
app = graph.compile()

example_df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Score": [90, 85, 88]
})

# user_query = "Calculate the average score for users older than 28."
user_query = "Plot the Age and Score columns."
initial_state = {
    "messages": [],
    "dataframe": example_df,
    "query": user_query
}

result = app.invoke(initial_state)

print("Generated Python Function:")
print(result["messages"][-1].content)


