from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage, SystemMessage
import requests
# from typing import List, Literal,  Annotated, Sequence
from typing_extensions import TypedDict, List, Literal,  Annotated, Sequence
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import pandas as pd
import operator
from rich import print

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

@tool
def execute_generated_code(state: AgentState):
    """
        A tool for executing python code over a pandas dataframe to provide necessary
        information and results. 
    """
    dataframe = state['dataframe']
    generated_code = state['generated_code']
    logs = state.get('logs', [])

    try:
        # Prepare a local namespace to execute the code
        local_namespace = {}
        exec(generated_code, globals(), local_namespace)
        
        # Assume the function name is known or can be dynamically extracted
        func_name = next(iter(local_namespace))  # Extract the first defined function
        generated_function = local_namespace[func_name]

        # Call the function with the DataFrame as an argument
        result = generated_function(dataframe)
        print("dataframe:", dataframe.head())
        logs.append(f"Execution result: {result}")
    except Exception as e:
        logs.append(f"Error executing generated code: {str(e)}")
        result = None

    # Update the state with the execution result
    return {'result': result, 'logs': logs}

tools = [execute_generated_code]
tool_node = ToolNode(tools)


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


# Add nodes to the graph
graph = StateGraph(AgentState)
graph.add_node("generate_prompt", create_system_message)
graph.add_node("query_model", call_model)

# Define edges
graph.add_edge("generate_prompt", "query_model")  # Prompt generation -> LLM call
graph.set_entry_point("generate_prompt")  # Start from prompt generation
graph.set_finish_point("query_model")  # End after the model's response

# Compile the graph
app = graph.compile()

example_df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Score": [90, 85, 88]
})

user_query = "Calculate the average score for users older than 28."
initial_state = {
    "messages": [],
    "dataframe": example_df,
    "query": user_query
}

result = app.invoke(initial_state)

print("Generated Python Function:")
print(result["messages"][-1].content)


