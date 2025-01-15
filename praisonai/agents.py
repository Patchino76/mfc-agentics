from praisonaiagents import Agent, Task, PraisonAIAgents
import pandas as pd
# import subporcess 
import os
from praisonaiagents.tools import read_csv, read_excel, write_csv, write_excel, filter_data, get_summary, group_by, pivot_table

from dotenv import load_dotenv
print("Loading env vars...")
print(load_dotenv("C:\\Users\\Svetlio\\OneDrive\\Projects\\mfc-agentics\\praisonai\\.env",  override=True))
for key, value in os.environ.items(): 
    print(f"{key}: {value}")

print(os.getenv("OPENAI_API_KEY"))

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

df_analyst = Agent(
    name = "Dataframe Analyst",
    role = "Senior Dataframe Analyst and Data Scientist",
    goal = "Answer questions about the dataframe",
    backstory = """
        You are a Dataframe Analyst and Data Scientist with a strong background in data analysis and machine learning.
        You have expertise in Python, Pandas, NumPy, and Matplotlib for data manipulation and visualization.
        You are skilled in statistical analysis, identifying trends and analysing complex datasets.
        """,
    verbose = True,
    llm = "ollama/granite3.1-dense:8b",
    markdown = True,
    min_reflect=1,
    max_reflect=1,
    tools=[read_csv, read_excel, write_csv, write_excel, filter_data, get_summary, group_by, pivot_table],
    self_reflect = False
)
df_analyst.tools.append(df)
analysis_task = Task(
    name = "data_analysis",
    description = """
        Calculate the correlation matrix between the dataframe columns.
        """,
    expected_output="Statistical summary and key insights.",
    agent = df_analyst
)

agents = PraisonAIAgents( agents=[df_analyst], tasks=[analysis_task], process="sequential" ) 
agents.start()