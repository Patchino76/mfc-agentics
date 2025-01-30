import pandas as pd
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, ToolExecutor, InjectedState
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage, SystemMessage, FunctionMessage, AIMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict, List, Literal,  Annotated, Sequence
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.tools import tool, InjectedToolArg
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
import numpy as np

load_dotenv(override=True)

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_gemini = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config={
        "temperature": 0,
        "max_output_tokens": 1024,
        "top_p": 1,
        "top_k": 1
    }
)

# Generate sample data
full_df = gen_synthetic_df()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    generated_code: str | None
    execution_result: str | None

def analyze_correlations(df: pd.DataFrame) -> str:
    """Default correlation analysis function."""
    try:
        # Calculate correlations between stream durations
        stream_durations = df.groupby('stream_name')['duration_minutes'].agg(['mean', 'count'])
        stream_durations = stream_durations[stream_durations['count'] > 1]
        
        # Get total downtime by stream
        total_by_stream = df.groupby('stream_name')['duration_minutes'].sum().sort_values(ascending=False)
        
        result = []
        result.append("\nTotal Downtime by Stream:")
        result.append(total_by_stream.to_string())
        
        result.append("\nAverage Downtime by Stream:")
        result.append(stream_durations['mean'].sort_values(ascending=False).to_string())
        
        if len(stream_durations) > 1:
            # Pivot the data to get stream correlations
            stream_matrix = pd.pivot_table(
                df, 
                values='duration_minutes',
                index='start_time',
                columns='stream_name',
                aggfunc='sum'
            ).fillna(0)
            
            if stream_matrix.shape[1] > 1:
                correlations = stream_matrix.corr()
                result.append("\nStream Correlations:")
                result.append(correlations.to_string())
        
        return "\n".join(result)
    except Exception as e:
        return f"Error in analysis: {str(e)}"

def generate_python_function(state: AgentState):
    """Generate Python function code based on a natural language query."""
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content

    # Create the analysis function
    state["generated_code"] = f"""def analyze_df(df):
    return analyze_correlations(df)
"""
    return state

def execute_code_tool(state: AgentState) -> str:
    """Execute the generated code and return results."""
    try:
        # Create namespace with required functions
        namespace = {
            'pd': pd,
            'analyze_correlations': analyze_correlations,
            'df': full_df
        }
        
        # Execute the function definition and call it
        exec(state["generated_code"], namespace)
        result = namespace['analyze_df'](full_df)
        
        return result
        
    except Exception as e:
        return f"Error executing code: {str(e)}"

def call_model(state: AgentState):
    """Handle execution flow."""
    if state["generated_code"] and not state["execution_result"]:
        try:
            result = execute_code_tool(state)
            state["execution_result"] = result
            state["messages"].append(AIMessage(content=f"Here's the correlation analysis:\n{result}"))
        except Exception as e:
            state["messages"].append(AIMessage(content=f"Error in analysis: {str(e)}"))
    return state

def tools_routing(state: AgentState):
    """Route to tools or end based on state."""
    if state["execution_result"] is None and state["generated_code"]:
        return "tools"
    return END

# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("call_model", call_model)
graph.add_node("generate_python_function", generate_python_function)
graph.add_node("tools", ToolNode([execute_code_tool]))

# Define edges
graph.add_edge(START, "generate_python_function")
graph.add_edge("generate_python_function", "call_model")
graph.add_edge("tools", "call_model")
graph.add_conditional_edges(
    "call_model",
    tools_routing,
    {
        "tools": "tools",
        END: END
    }
)

app = graph.compile()

user_query = "Можем ли да открием корелации между престоите в отделните потоци? Do not plot anything."

initial_state = AgentState(
    messages=[
        SystemMessage(content="I will analyze correlations in your DataFrame."),
        HumanMessage(content=user_query)
    ],
    generated_code=None,
    execution_result=None
)

result = app.invoke(initial_state)

print("\nCorrelation Analysis Results:")
if result["execution_result"]:
    print(result["execution_result"])
else:
    print("No results were generated")