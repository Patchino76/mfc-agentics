from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import CodeInterpreterTool
from crewai import LLM
from typing import Dict, Optional
import pandas as pd
from dotenv import load_dotenv
import os
from database import DataBase
from synthetic_df import gen_synthetic_df


# Load environment variables
print("Loading env vars...")
print(load_dotenv(override=True))

# llm = LLM(
#     model="groq/llama-3.3-70b-versatile",
#     temperature=0.1
# )

llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434", 
)

class AnalyzeDataFrameTool(BaseTool):
    name: str = "analyze_dataframe"
    description: str = "Analyze a pandas DataFrame and return basic statistics"
    
    def __init__(self):
        super().__init__()

    def _run(self) -> str:
        """Analyze a pandas DataFrame and return basic statistics."""
        try:
            # Load the DataFrame using gen_synthetic_df
            df = gen_synthetic_df()
            if df is not None and isinstance(df, pd.DataFrame):
                analysis = f"""
                DataFrame Analysis:
                Shape: {df.shape}
                Columns: {list(df.columns)}
                Data Types: {df.dtypes.to_dict()}
                Summary Statistics:
                {df.describe().to_string()}
                """
                return analysis
            return "No valid DataFrame found"
        except Exception as e:
            return f"Error analyzing DataFrame: {str(e)}"

    def _arun(self):
        pass  # Placeholder for async version

# class ExecutePandasCodeTool(BaseTool):
#     name: str = "execute_pandas_code"
#     description: str = "Execute pandas code and return the result as a string"
    
#     def __init__(self):
#         super().__init__()

#     def _run(self, code: str) -> str:
#         """Execute pandas code and return the result as a string."""
#         try:
#             # Create a local environment with pandas
#             local_env = {'pd': pd}
#             # Execute the code and capture the output
#             exec(code, local_env)
#             # Return the last defined variable
#             return str(local_env.get('result', 'Code executed successfully'))
#         except Exception as e:
#             return f"Error executing code: {str(e)}"

#     async def _arun(self, code: str) -> str:
#         """Async version of the tool"""
#         return self._run(code)

# Create tool instances
analyze_tool = AnalyzeDataFrameTool()
# execute_tool = ExecutePandasCodeTool()
code_interpreter = CodeInterpreterTool()

# Create the agents
# Enhance the data_analyst agent to handle time series data

data_analyst = Agent(
    role='Data Analyst',
    goal='Analyze and plot complex time series data',
    backstory="""You are an experienced data analyst skilled in pandas and Python. 
    Your job is to analyze various pandas dataframes and provide meaningful insights, sttatistics, including visualizations.""",
    tools=[analyze_tool],
    verbose=True,
    llm=llm
)

code_executor = Agent(
    role='Code Execution Agent',
    goal='Execute python code safely and return results',
    backstory="""You are a code execution specialist who ensures pandas code runs
    correctly and safely. You validate and execute code while handling errors.""",
    tools=[code_interpreter],
    verbose=True,
    llm=llm,
    allow_delegation=False
)

# Example usage
def run_data_analysis():
    # Load the DataFrame from gen_synthetic_df
    df = gen_synthetic_df()

    # Create tasks with expected outputs
    analysis_task = Task(
        description="Analyze this DataFrame and provide insights:",
        expected_output="""A comprehensive analysis of the DataFrame including:
        - the python code to calculate the total duration for each stream in the dataframe
        """,
        agent=data_analyst
    )

    execution_task = Task(
        description="Execute this code:",
        expected_output="""The result of the code execution.""",
        agent=code_executor
    )

    # Create and run the crew
    crew = Crew(
        agents=[data_analyst, code_executor],
        tasks=[analysis_task, execution_task],
        process=Process.sequential
    )

    result = crew.kickoff()
    return result

if __name__ == "__main__":
    result = run_data_analysis()
    print("\nCrew Analysis Results:")
    print(result)
