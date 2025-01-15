from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai import LLM
from typing import Dict, Optional
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
print("Loading env vars...")
print(load_dotenv())

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.1
)

class AnalyzeDataFrameTool(BaseTool):
    name: str = "analyze_dataframe"
    description: str = "Analyze a pandas DataFrame and return basic statistics"
    
    def __init__(self):
        super().__init__()

    def _run(self, df_code: str) -> str:
        """Analyze a pandas DataFrame and return basic statistics."""
        try:
            # Create a local environment with pandas
            local_env = {'pd': pd}
            # Execute the code to create the DataFrame
            exec(df_code, local_env)
            # Get the DataFrame from the local environment
            df = local_env.get('df')
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
            return "No valid DataFrame found in the code"
        except Exception as e:
            return f"Error analyzing DataFrame: {str(e)}"

    async def _arun(self, df_code: str) -> str:
        """Async version of the tool"""
        return self._run(df_code)

class ExecutePandasCodeTool(BaseTool):
    name: str = "execute_pandas_code"
    description: str = "Execute pandas code and return the result as a string"
    
    def __init__(self):
        super().__init__()

    def _run(self, code: str) -> str:
        """Execute pandas code and return the result as a string."""
        try:
            # Create a local environment with pandas
            local_env = {'pd': pd}
            # Execute the code and capture the output
            exec(code, local_env)
            # Return the last defined variable
            return str(local_env.get('result', 'Code executed successfully'))
        except Exception as e:
            return f"Error executing code: {str(e)}"

    async def _arun(self, code: str) -> str:
        """Async version of the tool"""
        return self._run(code)

# Create tool instances
analyze_tool = AnalyzeDataFrameTool()
execute_tool = ExecutePandasCodeTool()

# Create the agents
data_analyst = Agent(
    role='Data Analyst',
    goal='Analyze pandas DataFrames and provide insights',
    backstory="""You are an experienced data analyst skilled in pandas and Python. 
    Your job is to analyze DataFrames and provide meaningful insights.""",
    tools=[analyze_tool],
    verbose=True,
    llm=llm
)

code_executor = Agent(
    role='Code Executor',
    goal='Execute pandas code safely and return results',
    backstory="""You are a code execution specialist who ensures pandas code runs
    correctly and safely. You validate and execute code while handling errors.""",
    tools=[execute_tool],
    verbose=True,
    llm=llm
)

# Example usage
def run_data_analysis():
    # Create sample DataFrame code
    sample_df_code = """
    df = pd.DataFrame({
        'name': ['John', 'Jane', 'Bob', 'Alice'],
        'age': [25, 30, 35, 28],
        'salary': [50000, 60000, 75000, 65000]
    })
    """

    # Create tasks with expected outputs
    analysis_task = Task(
        description=f"""Analyze this DataFrame and provide insights:
        {sample_df_code}""",
        expected_output="""A comprehensive analysis of the DataFrame including:
        - Basic statistics (mean, std, min, max)
        - Data types of columns
        - Shape of the DataFrame
        - Any notable patterns or insights""",
        agent=data_analyst
    )

    execution_task = Task(
        description=f"""Execute this code and calculate the average salary:
        {sample_df_code}
        result = df['salary'].mean()""",
        expected_output="""The average salary calculated from the DataFrame,
        returned as a numerical value.""",
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
