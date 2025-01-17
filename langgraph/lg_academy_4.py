from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rich import print
from langchain_ollama import ChatOllama


messages = [SystemMessage(content="You are a helpful assistant that summarizes text.", name="system")]
messages.extend([HumanMessage(content="Hello, how are you?", name="human")])
messages.extend([AIMessage(content="What's your favorite color?", name="ai")])

for m in messages:
    print(m)

llm = ChatOllama(model="llama3.1", temperature=0)
response = llm.invoke(messages)
print(response)
