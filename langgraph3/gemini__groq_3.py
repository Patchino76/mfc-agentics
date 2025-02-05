import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import seaborn as sns
import pandas as pd
from langchain_core.tools.base import InjectedToolCallId
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, ToolExecutor
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage, SystemMessage, FunctionMessage, AIMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict, List, Literal, Annotated, Sequence, Any, Dict, Optional
from langgraph.types import Command
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.prebuilt import InjectedState
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
import google.generativeai as genai
from synthetic_df import gen_synthetic_df

load_dotenv(override=True)

# Configure the Gemini API
genai.configure(api_key="AIzaSyD-S0ajn_qCyVolBLg0mQ83j0ENoqznMX0")
llm_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash-thinking-exp-01-21")
llm_groq = ChatGroq(model="llama-3.3-70b-versatile", api_key = "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")
llm_ollama = ChatOllama(model="granite3.1-dense:8b", temperature=0) #llama3.1:latest granite3.1-dense:8b qwen2.5-coder:14b  jacob-ebey/phi4-tools deepseek-r1:14b
full_df = gen_synthetic_df()

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    query: str
    generated_code: str
    exec_result: Optional[Any] = None
    review_comments: Optional[str] = None


def generate_python_function(state : AgentState):
    """
    Generate Python function code based on a natural language query about a DataFrame.
    """
    sample_df = full_df.head().to_string()
    query = state["query"]
    # messages = state["messages"]
    # last_message = messages[-1]
    # query = last_message.content

    # Prepare the prompt for Gemini
    func_prompt = f"""You are an expert Python developer and data analyst. Based on the user's query and the provided DataFrame sample,
    generate Python function code to perform the requested analysis.

    User Query: {query}
    Sample DataFrame used only to infer the structure of the DataFrame:
    {sample_df}

    Provide the Python function as a single string that can be executed using the exec function.
    The function should accept a pd.DataFrame object with the same structure as the sample DataFrame as a parameter.
    Return only the Python function as a string and do not try to execute the code.
    Do not add sample dataframes, function descriptions and do not add calls to the function.

    If you need to return a pd.Series, please convert it to a pd.DataFrame. 
    Place the index in the first column and the other in the second one before returning it.
    Try to convert the column names of the dataframe into bulgarian language.

    Do not prefer seaborn than matplotlib for better and more complicated plots.
    If you create a plot function, do not use plt.show(), instead return the image in base64 format using the base64 and BytesIO libraries.
    If returning a base64 string do not add 'data:image/png;base64' to it."""

    response = llm_gemini.generate_content(func_prompt)
    print("1 - Generated Python Function Response:")
    # print(response.text)
    extracted_function = extract_function_code(response.text)
    state["generated_code"] = extracted_function
    return state # Return the updated state


def extract_function_code(generated_code: str) -> str:

    # Extract and execute the function
    function_match = re.search(r"(def\s+\w+\(.*?\):\n(?:\s+.*\n)*)", generated_code, re.DOTALL)
    if not function_match:
        raise ValueError("Error: Could not extract a valid function definition from the generated code.")

    function_body = function_match.group(1)
    print("2 - Extracted Function Body:")
    # print(function_body)
    return function_body


@tool
def review_python_code(state: dict) -> dict:
    """Review the generated Python code for errors and improvements."""
    if "generated_code" not in state:
        return state

    code = state["generated_code"]
    
    review_prompt = """You are a Python code reviewer specialized in data analysis and visualization. 
    Review the following code and provide corrections if needed. Focus on:

    1. Syntax errors and logical bugs
    2. DataFrame operations efficiency
    3. Proper error handling for data operations
    4. Memory management for large datasets
    5. Proper cleanup of matplotlib/seaborn resources
    6. Type hints and documentation
    7. Bulgarian language column names (if missing)
    8. Proper handling of base64 encoding for plots
    9. Proper handling of empty or invalid data
    10. Seaborn usage for better visualization (when appropriate)

    If you find issues, provide the complete corrected code between ```python``` tags.
    If the code is good, respond with "PASS".

    Code to review:
    ```python
    {code}
    ```
    """

    llm_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash-thinking-exp-01-21")

    response = llm_gemini.generate_content(review_prompt.format(code=code))
    review_result = response.text.strip()
    
    if review_result != "PASS":
        # Extract the corrected code if provided
        code_match = re.search(r"```python\n(.*?)```", review_result, re.DOTALL)
        if code_match:
            corrected_code = code_match.group(1).strip()
            state["generated_code"] = corrected_code
            state["review_comments"] = review_result.replace(code_match.group(0), "").strip()
            print("4 - Code Review Comments:")
            print(state["review_comments"])
            print("\n5 - Corrected Code:")
            print(corrected_code)
        else:
            state["review_comments"] = review_result
            print("4 - Code Review Comments (no corrections needed):")
            print(review_result)
    else:
        print("4 - Code Review: PASS")

    return state


@tool
def execute_python_function(state: dict) -> dict:
    """Execute the generated and reviewed Python function."""
    if "generated_code" not in state:
        return state
        
    function_body = state["generated_code"]
    print("\n6 - Executing Final Function:")
    print(function_body)
    
    try:
        namespace = {}
        # Import all necessary libraries
        exec("""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import base64
import numpy as np
import io
        """.strip(), namespace)
        
        # Execute the function definition
        exec(function_body, namespace)

        # Find and execute the function
        function_name = re.search(r"def\s+(\w+)\(", function_body).group(1)
        result = namespace[function_name](full_df)
        
        print("\n7 - Execution Result Type:", type(result))
        
        state["exec_result"] = result
        return state
        
    except Exception as e:
        error_message = f"Error executing function: {str(e)}"
        print("\nExecution Error:", error_message)
        state["error"] = error_message
        return state


tools = [execute_python_function]
tool_node = ToolNode(tools)
llm_tools = llm_groq.bind_tools(tools)


def call_model(state:AgentState):
    messages = state["messages"]  # Ensure system message is included
    response = llm_tools.invoke(messages)
    print("Agent Response:", response)
    return {"messages": messages + [response]}

    return END
def router(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    return END

def run_graph(query: str):
    graph = StateGraph(AgentState)
    graph.add_node("generate_python_function", generate_python_function)
    graph.add_node("review_python_code", review_python_code)
    graph.add_node("execute_python_function", execute_python_function)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "generate_python_function")
    graph.add_edge("generate_python_function", "review_python_code")
    graph.add_edge("review_python_code", "execute_python_function")
    graph.add_edge("execute_python_function", "call_model")
    graph.add_conditional_edges("call_model", router, {"tools": "tools", END: END})

    app = graph.compile()

    initial_state = AgentState(
        messages=[(SystemMessage(content=""" You have been provided with Python code in the 'generated_code' part of the state.
            Your ONLY task is to use the 'execute_code_tool' to execute this provided code."""))],
        query=query,
        generated_code="",
        exec_result=None
    )

    result = app.invoke(initial_state)
    exec_result = result["exec_result"]
    print("exec_result: ", exec_result)
    print("Type of exec_result: ", type(exec_result))
    if "review_comments" in result:
        print("Code Review Comments:", result["review_comments"])
    return exec_result

user_query = "Справка за престоите на поток 1 и поток 2 по категории.."
exec_result = run_graph(user_query)
print("final result: ", exec_result)
print("Type of exec_result:", type(exec_result))

# user_query = "Справка за престоите на поток 1 и поток 2 по категории. Do not plot anything."
# Кои са причините за най-дълг престой на поток 1 и поток 2? Колко са те?
# Начертай диаграма на разсейване с регресионна права на престоите по категории на поток 5 и 7.
# result = run_graph(user_query)
# print("final result: ", result)
