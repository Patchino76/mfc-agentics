from crewai import Agent, Task, Crew
from crewai import LLM
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
print("Loading env vars...")
print(load_dotenv(override=True))
# for key, value in os.environ.items():
#     print(f"{key}: {value}")


# Initialize Ollama
llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434", 
)
# llm = LLM(
#     model="groq/llama-3.2-90b-text-preview",
#     temperature=0.1
# )
# Custom tool wrapper
class DataTool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description

# Custom tools for data analysis
analyze_dataframe_tool = DataTool(
    name="Analyze Dataframe",
    func=lambda df: df.corr().to_string(),
    description="Analyze the dataframe and return correlation matrix"
)

get_summary_stats_tool = DataTool(
    name="Get Summary Stats",
    func=lambda df: df.describe().to_string(),
    description="Get summary statistics for the dataframe"
)

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

# Create the Data Analyst agent
data_analyst = Agent(
    role='Senior Data Analyst',
    goal='Analyze the dataframe and provide statistical insights',
    backstory="""You are a Senior Data Analyst with extensive experience in data analysis 
        and statistical modeling. You excel at identifying patterns and deriving insights 
        from complex datasets.""",
    verbose=True,
    llm=llm,
    tools=[analyze_dataframe_tool, get_summary_stats_tool]
)

# Create the analysis task
analysis_task = Task(
    description="""
    Analyze the provided dataset by:
    1. Calculating the correlation matrix between columns
    2. Providing summary statistics
    3. Identifying key patterns and relationships
    
    Work with the dataframe stored in the 'df' variable.
    """,
    expected_output="""
    The task should output a detailed analysis report including:
    - A correlation matrix of the dataset
    - Summary statistics of the dataset
    - Key patterns and relationships identified in the data
    """,
    agent=data_analyst
)

# Create and run the crew
crew = Crew(
    agents=[data_analyst],
    tasks=[analysis_task],
    verbose=True
)

# Execute the analysis
result = crew.kickoff()

# Print results
print("\nAnalysis Results:")
print(result)