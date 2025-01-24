import pandas as pd
from dotenv import load_dotenv
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
import google.generativeai as genai
from synthetic_df import gen_synthetic_df

load_dotenv(override=True)

# Configure the Gemini API
genai.configure(api_key="AIzaSyD-S0ajn_qCyVolBLg0mQ83j0ENoqznMX0")
llm_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash-thinking-exp-01-21")  
llm_groq = ChatGroq(model="llama-3.3-70b-versatile", api_key = "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")
llm_ollama = ChatOllama(model="granite3.1-dense:8b", temperature=0) #llama3.1:latest granite3.1-dense:8b qwen2.5-coder:14b  jacob-ebey/phi4-tools deepseek-r1:14b 
full_df = gen_synthetic_df()


@tool
def generate_python_function(query: str = 'Каква е продължителността на престоите по категории?'):
    """
    Generate Python function code based on a natural language query about a DataFrame.

    This function uses a generative AI model to create Python code that analyzes pandas DataFrames
    based on natural language queries. It handles both English and non-English queries by translating
    them appropriately.

    Args:
        query (str): Natural language query describing the desired analysis. Can be in any language,
            as it will be translated to English if needed.

    Returns:
        str: A string containing the generated Python function code. The generated function will:
            - Accept a pandas DataFrame as its input parameter
            - Perform the requested analysis based on the query
            - Return appropriate results (numerical values, strings, or base64-encoded plots)
            - Include proper error handling and edge cases
            - Be ready for execution using Python's exec() function

    Example:
        >>> query = "What is the average duration by category?"
        >>> function_code = generate_python_function(query)
        >>> # The generated code can then be executed using exec()

    Notes:
        - The function handles non-English queries by detecting the language and translating
        - Generated plotting functions return base64-encoded strings instead of displaying plots
        - The generated code follows PEP 8 style guidelines
        - Includes proper error handling for missing data or invalid inputs
    """
    sample_df = full_df.head().to_string()

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
    # print(response.text)
    return extract_function_code(response.text)

def extract_function_code(generated_code: str) -> str:

    # Extract and execute the function
    function_match = re.search(r"(def\s+\w+\(.*?\):\n(?:\s+.*\n)*)", generated_code, re.DOTALL)
    if not function_match:
        raise ValueError("Error: Could not extract a valid function definition from the generated code.")
    
    function_body = function_match.group(1)
    print("Extracted Function Body:")
    print(function_body)
    return function_body

@tool
def execute_code_tool(function_body: str) -> str:
    """
    Executes dynamically generated Python code on a provided dataframe.

    Args:
        function_body (str): The Python code, returned as a string, defining the function to be executed.

    Returns:
        str: The result of executing the function, which must be a string, number, or list.
    """
    # Deserialize the JSON string to a DataFrame

    
    namespace = {}
    exec("import pandas as pd\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nimport base64\nimport numpy as np\n", namespace)  # Import necessary modules
    exec(function_body, namespace)
    
    function_name = re.search(r"def\s+(\w+)\(", function_body).group(1)
    result = namespace[function_name](full_df)
    
    if not isinstance(result, (str, int, float, list)):
        raise ValueError("The result must be a string, number, or list.")
    
    print("Result of code execution:", result)
    return json.dumps(result) 


tools = [generate_python_function, execute_code_tool]
tool_node = ToolNode(tools)
llm_tools = llm_groq.bind_tools(tools)

# test_message = AIMessage(
#     content = "",
#     tool_calls = [
#         {
#             "name" : "generate_python_function",
#             "args" : {
#                 "query" : "What is the average duration by category?"
#             },
#             "id" : "1",
#             "type" : "tool_call"
#         }
#     ],
# )
# tool_node.invoke({"messages": [test_message]})

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    # generated_code: str

def call_model(state:AgentState):
    messages = state["messages"]  # Ensure system message is included
    response = llm_tools.invoke(messages)
    print("Agent Response:", response)
    return {"messages": messages + [response]}

def tools_routing(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("call_model", call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", tools_routing, ["tools", END])

graph.set_finish_point("tools")
app = graph.compile()

user_query = "Calculate the average downtime by category. Do not plot anything. Execute the code and provide the result."
initial_state = {
    "messages": [
        SystemMessage(content="""You are a helpful AI assistant that helps users analyze data using Python. 
        When given a query about data analysis, you will help generate and execute Python code to answer the query.
        Always use the provided tools to accomplish the task."""),
        AIMessage(content=user_query)
    ],
    "query": user_query,
}

result = app.invoke(initial_state)

print("Generated Python Function:")
print(result["messages"][-1].content)
