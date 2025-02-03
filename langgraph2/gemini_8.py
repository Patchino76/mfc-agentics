import pandas as pd
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, ToolExecutor
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage, SystemMessage, FunctionMessage, AIMessage, HumanMessage, ToolMessage
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
import google.generativeai as genai
from synthetic_df import gen_synthetic_df

load_dotenv(override=True)

# Configure the Gemini API
genai.configure(api_key="AIzaSyD-S0ajn_qCyVolBLg0mQ83j0ENoqznMX0")
llm_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash-thinking-exp-01-21")
llm_groq = ChatGroq(model="llama-3.3-70b-versatile", api_key = "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")
llm_ollama = ChatOllama(model="MFDoom/deepseek-r1-tool-calling:14b", temperature=0) #llama3.1:latest granite3.1-dense:8b qwen2.5-coder:14b  jacob-ebey/phi4-tools deepseek-r1:14b
full_df = gen_synthetic_df()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # query: str
    generated_code: str | None
    execution_result: str | None


def generate_python_function(state : AgentState): # OLLAMA LLM FUNCTION
    """
    Generate Python function code based on a natural language query about a DataFrame.
    """
    sample_df = full_df.head().to_string()
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content

    system_prompt = """You are an expert Python developer and data analyst. Your task is to generate Python function code based on user queries about DataFrame analysis.
    You must provide ONLY the Python function code without any explanations, descriptions, or example calls.
    If creating plots, do not use plt.show(), instead return the image in base64 format using base64 and BytesIO libraries.
    If returning a base64 string do not add 'data:image/png;base64' to it.
    The function should accept a pd.DataFrame parameter with the same structure as the sample DataFrame provided."""

    user_prompt = f"""Generate a Python function for the following query:
    
    Query: {query}
    
    Sample DataFrame (used only to infer structure):
    {sample_df}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm_ollama.invoke(messages)
    print("1 - Generated Python Function Response:")
    print(response.content)
    extracted_function = extract_function_code(response.content)
    state["generated_code"] = extracted_function
    return state
# def generate_python_function(state : AgentState):
#     """
#     Generate Python function code based on a natural language query about a DataFrame.
#     """
#     sample_df = full_df.head().to_string()
#     # query = state["query"]
#     messages = state["messages"]
#     last_message = messages[-1]
#     query = last_message.content

#     # Prepare the prompt for Gemini
#     func_prompt = f"""You are an expert Python developer and data analyst. Based on the user's query and the provided DataFrame sample,
#     generate Python function code to perform the requested analysis.

#     User Query: {query}
#     Sample DataFrame used only to infer the structure of the DataFrame:
#     {sample_df}

#     Provide the Python function as a single string that can be executed using the exec function.
#     The function should accept a pd.DataFrame object with the same structure as the sample DataFrame as a parameter.
#     Return only the Python function as a string and do not try to execute the code.
#     Do not add sample dataframes, function descriptions and do not add calls to the function.

#     If you create a plot function, do not use plt.show(), instead return the image in base64 format using the base64 and BytesIO libraries.
#     If returning a base64 string do not add 'data:image/png;base64' to it."""

#     response = llm_gemini.generate_content(func_prompt)
#     print("1 - Generated Python Function Response:")
#     print(response.text)
#     extracted_function = extract_function_code(response.text)
#     state["generated_code"] = extracted_function
#     return state # Return the updated state

def extract_function_code(generated_code: str) -> str:

    # Extract and execute the function
    function_match = re.search(r"(def\s+\w+\(.*?\):\n(?:\s+.*\n)*)", generated_code, re.DOTALL)
    if not function_match:
        raise ValueError("Error: Could not extract a valid function definition from the generated code.")

    function_body = function_match.group(1)
    print("2 - Extracted Function Body:")
    print(function_body)
    return function_body

@tool
def execute_code_tool(generated_code: str) -> str:
    """
    Executes dynamically generated Python code on a provided dataframe.

    Args:
        generated_code (str): The Python code, returned as a string, defining the function to be executed.

    Returns:
        str: The result of executing the function, which must be a string, number, or list.
    """
    # Deserialize the JSON string to a DataFrame

    function_body = generated_code
    print("3 - Executing Function Body:")
    print(function_body)
    namespace = {}
    exec("import pandas as pd\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nimport base64\nimport numpy as np\n", namespace)  # Import necessary modules
    exec(function_body, namespace)

    function_name = re.search(r"def\s+(\w+)\(", function_body).group(1)
    result = namespace[function_name](full_df)

    # if not isinstance(result, (str, int, float, list)):
    #     raise ValueError("The result must be a string, number, or list.")

    print("Result of code execution:", result)
    return json.dumps(result)


tools = [execute_code_tool]
tool_node = ToolNode(tools)
llm_tools = llm_ollama.bind_tools(tools)


def call_model(state:AgentState):
    messages = state["messages"]  # Ensure system message is included
    response = llm_tools.invoke(messages)
    print("Agent Response:", response)
    return {"messages": messages + [response]}

# def tools_routing(state: AgentState):
#     messages = state["messages"]
#     last_message = messages[-1]
#     if last_message.tool_calls:
#         return "tools"
    # return END
def tools_routing(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    test_message = AIMessage(
        content = "",
        tool_calls = [
            {
                "name" : "execute_code_tool",
                "args" : {
                    "generated_code" : state["generated_code"]
                },
                "id" : "1",
                "type" : "tool_call"
            }
        ],
    )
    if last_message.tool_calls:
        # return "tools"
        tool_node.invoke({"messages": [test_message]})
    return END

graph = StateGraph(AgentState)
graph.add_node("call_model", call_model)
graph.add_node("generate_python_function", generate_python_function)
graph.add_node("tools", tool_node)

graph.add_edge(START, "generate_python_function")
# graph.add_edge("call_model", "generate_python_function")
graph.add_edge("generate_python_function", "call_model")
graph.add_conditional_edges("call_model", tools_routing, {"tools": "tools", END: END}) # Use dictionary for clarity

# graph.set_finish_point(END) # Finish at the end of the graph or after tools
app = graph.compile()

user_query = "Справка за престоите на поток 1 и поток 2 по категории. Do not plot anything."

initial_state = AgentState(
    messages=[
        SystemMessage(content=""" You have been provided with Python code in the 'generated_code' part of the state.
        Your ONLY task is to use the 'execute_code_tool' to execute this provided code.
        Do NOT generate new code. The code to execute is already available.
        You must call the 'execute_code_tool' and pass the current state as an argument to it.
        The 'execute_code_tool' will handle the execution of the code.
        Your response should be a Tool Call to the 'execute_code_tool' with the current state."""),
       HumanMessage(content=user_query)
    ],
    # query=user_query,
    generated_code=None,
    execution_result=None
)


result = app.invoke(initial_state)

print("Final result after tool calls")
print(result["messages"][-1].content)