import pandas as pd
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, ToolExecutor
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage, SystemMessage, FunctionMessage, AIMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict, List, Literal, Annotated, Sequence
from langchain_core.tools import tool
import operator
from rich import print
import re
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64
import os
import ast
import json
import google.generativeai as genai
from synthetic_df import gen_synthetic_df

load_dotenv(override=True)

# Configure the models
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_gemini = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config={"temperature": 0}
)

# Generate sample data
full_df = gen_synthetic_df()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    generated_code: str | None
    execution_result: str | None

def generate_python_function(state: AgentState):
    """Generate Python function code based on a natural language query."""
    sample_df = full_df.head().to_string()
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content

    func_prompt = f"""You are an expert Python developer and data analyst. Generate a Python function for DataFrame analysis.
    
    User Query: {query}
    Sample DataFrame structure:
    {sample_df}

    Requirements:
    1. Function MUST be named 'analyze_df' and accept a single 'df' parameter
    2. Return analysis results (numbers, strings, or base64 plots)
    3. For plots, use BytesIO and base64 encoding (no plt.show())
    4. Include error handling for missing data
    5. DO NOT include any code outside the function
    6. DO NOT create test data or execute code
    7. DO NOT add docstrings or comments

    Example of expected format:
    def analyze_df(df):
        return df.groupby('Category')['Value'].mean()
    """

    response = llm_gemini.generate_content(func_prompt)
    print("Generated Function:")
    print(response.text)
    
    extracted_function = extract_function_code(response.text)
    if not extracted_function.startswith("def analyze_df(df):"):
        extracted_function = "def analyze_df(df):\n" + "\n".join(
            "    " + line for line in extracted_function.split("\n") if not line.startswith("def")
        )
    
    state["generated_code"] = extracted_function
    state["messages"].append(AIMessage(content="I have generated the analysis function."))
    return state

def extract_function_code(generated_code: str) -> str:
    """Extract only the function code, removing any surrounding text."""
    lines = generated_code.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
        if in_function:
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else generated_code

@tool
def execute_code_tool(state: AgentState) -> str:
    """Execute the generated Python function on the DataFrame."""
    if not state["generated_code"]:
        return "No code has been generated yet."
    
    try:
        # Create a new namespace to avoid polluting globals
        namespace = {'pd': pd, 'plt': plt, 'BytesIO': BytesIO, 'base64': base64}
        
        # Execute the function definition
        exec(state["generated_code"], namespace)
        
        # Execute the function with our DataFrame
        result = namespace['analyze_df'](full_df)
        
        # Convert result to string if needed
        if isinstance(result, (pd.Series, pd.DataFrame)):
            result = result.to_string()
        elif not isinstance(result, str):
            result = str(result)
            
        return result
        
    except Exception as e:
        return f"Error executing code: {str(e)}"

def call_model(state: AgentState):
    """Handle LLM responses with strict control over code execution."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If we have generated code and no execution result, execute it
    if state["generated_code"] and not state["execution_result"]:
        try:
            result = execute_code_tool(state)
            state["execution_result"] = result
            response = llm_gemini.generate_content(
                f"""Here is the result of the DataFrame analysis:
                {result}
                
                Please provide a clear explanation of these results."""
            )
            state["messages"].append(AIMessage(content=f"Analysis Results:\n{result}\n\nExplanation:\n{response.text}"))
            state["next"] = END
            return state
        except Exception as e:
            state["messages"].append(AIMessage(content=f"Error executing code: {str(e)}"))
            state["next"] = END
            return state
    
    if isinstance(last_message, HumanMessage):
        state["next"] = "generate_python_function"
        return state
    
    state["next"] = END
    return state

def tools_routing(state: AgentState):
    """Route to tools only when explicitly needed."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    return END

# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("call_model", call_model)
graph.add_node("generate_python_function", generate_python_function)
graph.add_node("tools", ToolNode([execute_code_tool]))

# Define edges
graph.add_edge(START, "call_model")
graph.add_edge("generate_python_function", "call_model")
graph.add_edge("tools", "call_model")

# Add conditional routing based on next state
graph.add_conditional_edges(
    "call_model",
    lambda state: state["next"],
    {
        "generate_python_function": "generate_python_function",
        "tools": "tools",
        END: END
    }
)

app = graph.compile()

# Test the workflow
if __name__ == "__main__":
    user_query = "Calculate the average downtime by category. Do not plot anything."
    
    initial_state = AgentState(
        messages=[
            SystemMessage(content="""You are a DataFrame analysis assistant. Follow these steps strictly:
            1. When you receive a user query, wait for the Python function to be generated
            2. Once the function is generated, it will be automatically executed
            3. The results will be shown to the user with an explanation
            4. DO NOT try to execute code yourself or generate new code"""),
            HumanMessage(content=user_query)
        ],
        generated_code=None,
        execution_result=None
    )

    result = app.invoke(initial_state)
    print("\nFinal Results:")
    if result["execution_result"]:
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and "Analysis Results:" in msg.content:
                print(msg.content)
                break
    else:
        print("No results were generated")
