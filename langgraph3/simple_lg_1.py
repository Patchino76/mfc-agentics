#%%
import pandas as pd
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, ToolExecutor
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

load_dotenv(override=True)

#%%genai.configure(api_key="AIzaSyD-S0ajn_qCyVolBLg0mQ83j0ENoqznMX0")
llm_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash-thinking-exp-01-21")
llm_groq = ChatGroq(model="llama-3.3-70b-versatile", api_key = "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")
llm_ollama = ChatOllama(model="granite3.1-dense:8b", temperature=0) #llama3.1:latest granite3.1-dense:8b qwen2.5-coder:14b  jacob-ebey/phi4-tools deepseek-r1:14b
# %%
class InputState(TypedDict):
    messages: List[BaseMessage]
    name: str
    generated_code: str | None

def gen_code(state: InputState) -> InputState:
    name = state["name"]
    generated_code = f"print(f'Hello {name}')"
    print("Generated code:")
    print(generated_code)
    state["generated_code"] = generated_code
    state["messages"].append(SystemMessage(content="generating code..."))
    return state

@tool
def execute_code(code: str) -> str:
    """Execute the given code"""
    exec(code)
    return "Code executed successfully"

def call_model(state: InputState) -> InputState:
    messages = state["messages"]
    messages.append(SystemMessage(content=""" You have been provided with Python code in the 'generated_code' part of the state.
        Your ONLY task is to use the 'execute_code_tool' to execute this provided code."""))
    response = llm_tools.invoke(messages)
    state["messages"] = messages + [response]
    return state

tools = [execute_code]
tool_node = ToolNode(tools)
llm_tools = llm_groq.bind_tools(tools)

def router(state: InputState) -> Literal["execute", "end"]:
    print("we are in the router...")
    print(state)
    if state["generated_code"]:
        return "execute"
    return "end"

def execute(state: InputState) -> InputState:
    if state["generated_code"]:
        execute_code(state["generated_code"])
    return state

# %%
graph = StateGraph(InputState)

# Add nodes
graph.add_node("gen_code", gen_code)
graph.add_node("execute", execute)
graph.add_node("call_model", call_model)

# Add edges
graph.add_edge(START, "gen_code")
graph.add_edge("gen_code", "call_model")
graph.add_conditional_edges(
    "call_model",
    router,
    {
        "execute": "execute",
        "end": END
    }
)
graph.add_edge("execute", END)

app = graph.compile()
app
#%%

# Test the workflow
initial_state = InputState(
    messages=[SystemMessage(content="Starting code generation...")],
    generated_code=None,
    name="Svetlio"
)

result = app.invoke(initial_state)
# %%
