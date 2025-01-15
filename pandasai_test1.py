#%%
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from pydantic import BaseModel 
from langchain.prompts import PromptTemplate
from typing import List
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage, HumanMessage
from pandasai import SmartDataframe
import os
from dotenv import load_dotenv
env = load_dotenv()
print(env)

from database2 import DataBase

# %%
llm = ChatOllama(model="granite3.1-dense:8b")
# llm = ChatOllama(model="llama3.1:8b" )
# llm = ChatGroq(model="llama-3.3-70b-versatile", api_key = "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")

db = DataBase("D:/DataSets/MFC/SQLITE3/db.sqlite3")
query = 'SELECT * FROM MA_FULL_01 LIMIT 1000'
df_pd = db.get_dataframe(query)
# df_pd['TimeStamp'] = pd.to_datetime(df_pd['TimeStamp'])
# df_pd.set_index('TimeStamp', inplace=True)
# df_pd.head()
df_pd = df_pd.drop(columns=["TimeStamp"])
df_pd = df_pd.reset_index(drop=True)
df_pd.head()
# %%
df_ai = SmartDataframe(df_pd, config={"llm": llm, "verbose":True})

# %%
result = df_ai.chat("a seaborn joint plot of ore and watermill?")   
result
# %%
