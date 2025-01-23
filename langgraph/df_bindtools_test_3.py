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
print(os.getenv("GEMINI_API_KEY"))


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    generated_code: str


def create_system_message(state: AgentState):
    query = state["query"]  # Extract query from the state
    sample_df = gen_synthetic_df().head().to_string()
    print(sample_df)

    system_prompt = f"""
    You are a highly skilled Python developer and data analyst. 
    Your task is to generate Python function code based on the user's query and the provided DataFrame structure.

    **Specifications:**
    1. Use the user query: {query}.
    2. Refer to the provided sample DataFrame to infer the structure: {sample_df}.
    3. The function must:
    
    - Be a self-contained function with **no external dependencies** or references to code outside the function.
    - Accept a `pd.DataFrame` with the same structure as {sample_df} as an input parameter.

    4. The LLM should determine if the function involves plotting or not. If the query {query} contains plot,  
    then the function should:
    - Do not use `plt.show()`. Instead, return the plot as a Base64-encoded string.
    - Follow this procedure to encode the plot:
        ```python
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        return img_base64
        ```
    - Ensure no additional symbols or text follow the return statement.

    5. **Output**:
    - Return only the pure Python function definition as a single string.
    - Do not include any imports inside or outside of the function.
    - Do not include any extra code, descriptions, function calls, or comments outside the function.
    - Do not call the function.

    Your sole output should be the requested function code as a single string that can be executed using the `exec` function.

    After the function is generated, the LLM should call the appropriate tool for execution.
    """


    return {"messages": [SystemMessage(content=system_prompt)]}


def call_model(state):
    messages = state["messages"]  # Ensure system message is included
    response = llm_tools.invoke(messages)
    print("Agent Response:", response)
    return {"messages": messages + [response]}



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
    dataframe = gen_synthetic_df()
    generated_code = generated_code
    print("Generated Python Function:", generated_code)

    # Extract and execute the function
    function_match = re.search(r"(def\s+\w+\(.*?\):\n(?:\s+.*\n)*)", generated_code, re.DOTALL)
    if not function_match:
        raise ValueError("Error: Could not extract a valid function definition from the generated code.")
    
    function_body = function_match.group(1)
    print("Extracted Function Body:")
    print(function_body)
    namespace = {}
    exec("import pandas as pd\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nimport base64\nimport numpy as np", namespace)  # Import necessary modules
    exec(function_body, namespace)
    
    function_name = re.search(r"def\s+(\w+)\(", function_body).group(1)
    result = namespace[function_name](dataframe)
    
    if not isinstance(result, (str, int, float, list)):
        raise ValueError("The result must be a string, number, or list.")
    
    print("Result of code execution:", result)
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
# graph.add_edge("tools", "query_model")
graph.set_finish_point("tools")


# Compile the graph
app = graph.compile()

example_df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Score": [90, 85, 88]
})
df = gen_synthetic_df().head(5)
df_json = df.to_json(orient="split")

# user_query = "Calculate the average score for users older than 28."
user_query = "Calculate the average downtime by category. Do not plot anything. "
initial_state = {
    "messages": [],
    "dataframe_json": df_json,  # Pass DataFrame as JSON string
    "query": user_query,
    "generated_code": ""
}

result = app.invoke(initial_state)

print("Generated Python Function:")
print(result["messages"][-1].content)
