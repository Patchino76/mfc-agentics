#%%
import pandas as pd
from langchain_ollama import ChatOllama
from pydantic import BaseModel 
from langchain.prompts import PromptTemplate
from typing import List
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage, HumanMessage

from database import DataBase
# %%
# %%
db = DataBase("D:/DataSets/MFC/SQLITE3/db.sqlite3")
query = 'SELECT * FROM MA_FULL_01 LIMIT 10000'
df = db.get_dataframe(query)
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
df.set_index('TimeStamp', inplace=True)
df.head()
# %%
llm = ChatOllama(model="granite3.1-dense:8b")
# llm = ChatOllama(model="llama3.1:8b" )
# llm = ChatGroq(model="llama-3.3-70b-versatile", api_key = "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")

class ColumnInfo(BaseModel):
    name: str
    dtype: str

class DataFrameColumns(BaseModel):
    columns: List[ColumnInfo]

# %%

# Define the function to extract columns
def extract_columns(df: pd.DataFrame) -> DataFrameColumns:
    columns_info = [ColumnInfo(name=col, dtype=str(df[col].dtype)) for col in df.columns]
    return DataFrameColumns(columns=columns_info)

# Create LangGraph agent
builder = MessageGraph() 
builder.set_entry_point(extract_columns)
builder.add_node(extract_columns, END)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
# Query the agent 
result = graph.run(df) 
print(result.json(indent=2))
# %%
