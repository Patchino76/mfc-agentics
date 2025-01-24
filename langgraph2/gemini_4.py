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
llm_ollama = ChatOllama(model="granite3.1-dense:8b", temperature=0) #llama3.1:latest granite3.1-dense:8b qwen2.5-coder:14b  jacob-ebey/phi4-tools deepseek-r1:14b
full_df = gen_synthetic_df()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # query: str
    generated_code: str | None
    execution_result: str | None

@tool
def generate_python_function(state : AgentState):
    """
    Generate Python function code based on a natural language query about a DataFrame.
    """
    sample_df = full_df.head().to_string()
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content

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

    If you create a plot function, do not use plt.show(), instead return the image in base64 format using the base64 and BytesIO libraries.
    If returning a base64 string do not add 'data:image/png;base64' to it."""

    response = llm_gemini.generate_content(func_prompt)
    print("1 - Generated Python Function:")
    print(response.text)
    extracted_function = extract_function_code(response.text)
    return state.update({"generated_code": extracted_function})

def extract_function_code(generated_code: str) -> str:
    function_match = re.search(r"(def\s+\w+\(.*?\):\n(?:\s+.*\n)*)", generated_code, re.DOTALL)
    if not function_match:
        raise ValueError("Error: Could not extract a valid function definition from the generated code.")

    function_body = function_match.group(1)
    print("2 - Extracted Function Body:")
    print(function_body)
    return function_body

@tool
def execute_code_tool(state: AgentState) -> str:
    """
    Executes dynamically generated Python code on a provided dataframe.
    """

    function_body = state["generated_code"]
    if function_body is None:
        raise ValueError("No generated code to execute. Ensure generate_python_function tool is called first.")

    print("3 - Executing Function Body:")
    print(function_body)
    namespace = {}
    exec("import pandas as pd\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nimport base64\nimport numpy as np\n", namespace)  # Import necessary modules
    exec(function_body, namespace)

    function_name = re.search(r"def\s+(\w+)\(", function_body).group(1)
    result = namespace[function_name](full_df)

    if not isinstance(result, (str, int, float, list, pd.Series, pd.DataFrame)):
        if hasattr(result, 'to_json'):
            result = result.to_json()
        else:
            raise ValueError(f"The result must be a string, number, list, pandas Series or DataFrame. Got type: {type(result)}")

    print("Result of code execution:", result)
    return json.dumps(result)


generate_function_tool_node = ToolNode([generate_python_function])
execute_code_tool_node = ToolNode([execute_code_tool])
llm_tools = llm_groq.bind_tools([generate_python_function, execute_code_tool])


def call_model(state:AgentState):
    messages = state["messages"]
    response = llm_tools.invoke(messages)
    print("Agent Response:", response)
    return {"messages": messages + [response]}

def tools_routing(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        tool_calls = last_message.tool_calls
        tool_names = [tool_call.name for tool_call in tool_calls]
        if "generate_python_function" in tool_names:
            return "generate_function_tool"
        elif "execute_code_tool" in tool_names:
            return "execute_code_tool"
        else:
            return END
    return END


graph = StateGraph(AgentState)
graph.add_node("call_model", call_model)
graph.add_node("generate_function_tool", generate_function_tool_node)
graph.add_node("execute_code_tool", execute_code_tool_node)


graph.add_edge(START, "call_model")
graph.add_conditional_edges(
    "call_model",
    tools_routing,
    {
        "generate_function_tool": "generate_function_tool",
        "execute_code_tool": "execute_code_tool",
        END: END
    }
)
graph.add_edge("generate_function_tool", "call_model") # Go back to model after generating function
graph.add_edge("execute_code_tool", END) # End after executing code


app = graph.compile()

user_query = "Calculate the average downtime by category. Do not plot anything. Execute the code and provide the result."

initial_state = AgentState(
    messages=[
        SystemMessage(content="""You are a helpful AI assistant that helps users analyze data using Python.
        When given a query about data analysis, you will help generate and execute Python code to answer the query.
        Always use the provided tools to accomplish the task."""),
        AIMessage(content="I understand you want to analyze data. I'll help you generate and execute Python code to answer your query."),
        HumanMessage(content=user_query)
    ],
    generated_code=None,
    execution_result=None
)


result = app.invoke(initial_state)

print("Final result after tool calls")
print(result["messages"][-1].content)