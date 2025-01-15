import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Any
from pydantic_core import core_schema
from crewai_tools import CodeInterpreterTool
from crewai import LLM
from dotenv import load_dotenv
import json
from rich import print
from synthetic_df import gen_synthetic_df

print("Loading env vars...")
print(load_dotenv(override=True))

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.1
)

llm2 = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434", 
)


class CustomDataFrameTool(BaseTool):
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    name: str = "custom_dataframe_tool"
    description: str = "Execute generated code on a full DataFrame"
    df: pd.DataFrame = Field(default=None, description="The pandas DataFrame to be processed.")

    def __init__(self, df: pd.DataFrame = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.df = df

    def _run(self, code: str) -> str:
        """Execute the given code on the DataFrame and return the result."""
        print("------------------ CUSTOM TOOL RUN ------------------")
        try:
            local_env = {'pd': pd, 'df': self.df}
            exec(code, local_env)
            return str(local_env.get('result', 'Code executed successfully'))
        except Exception as e:
            return f"Error executing code: {str(e)}"

    async def _arun(self, code: str):
        raise NotImplementedError("Async version not implemented.")


full_dataframe = gen_synthetic_df()
df_tool = CustomDataFrameTool(df=full_dataframe)
df_tool.name = "df_tool"

def format_output(output):
    try:
        output_str = str(output)
        if output_str.strip():
            data = json.loads(output_str)
            formatted_output = json.dumps(data, indent=4)
            print("------------------FORMATED OUTPUT ------------------")
            print(formatted_output)
        else:
            print("Output is empty or not valid JSON")
    except (json.JSONDecodeError, TypeError) as e:
        print("Error decoding JSON:", e)

# Define global variable for the full dataframe
  # Generate the full DataFrame here

# Define agents
analytical_agent = Agent(
    role="Data Analyst",
    goal="Analyze queries and dataframe descriptions to generate Python code.",
    backstory=(
        "You are an expert Python data analyst. "
        "You will receive a query and a serialized dataframe in JSON format. "
        "Use the provided JSON to determine the structure of the DataFrame and generate Python code "
        "to fulfill the query. Do not make any assumptions about the DataFrame structure."
    ),
    verbose=True,
    llm=llm2,
    memory=True, 
    allow_delegation=True,
    max_iter=2,
    # step_callback=format_output
)


executor_agent = Agent(
    role='Code Execution Agent',
    goal='Execute generated code on DataFrame',
    backstory="""You are a code execution specialist who ensures generated code runs
    correctly and safely on the DataFrame. You validate and execute code while handling errors.""",
    tools=[df_tool],
    allow_delegation=False,
    allow_code_execution=True,
    memory=True, 
    llm=llm2,
    max_iter=2,
)

# Define tasks
generate_code_task = Task(
    name = "generate code task",
    description=(
        "You will receive a query and a small sample of the dataframe {dataframe_sample} in JSON format. "
        "Examine the {dataframe_sample} to determine the exact structure of the DataFrame."
        "Generate python code to fulfill the {query} based on the sample."
        "Do not make any assumptions about the DataFrame structure; "
        "Use only the provided sample to infer the structure of the dataframe."
        "Do not put the dataframe sample in the code. Let the Code Execution Agent handle that."
    ),
    expected_output="Python code string that processes the {query}.",
    agent=analytical_agent,
)

execution_task = Task(
    name = "execute code task",
    description="Execute generated code on the dataframe that is provided in the df_tool.",
    expected_output="The result of the code execution.",
    agent=executor_agent
)

# Create the crew
crew = Crew(
    agents=[analytical_agent, executor_agent],
    tasks=[generate_code_task, execution_task],
    process=Process.sequential
)

# Mock inputs
print(full_dataframe.head())
inputs = {
    "query": "This dataframe contains downtime data for different streams and machines. "
    "List the unique categories of downtime that are in the dataframe?",
    "dataframe_sample": full_dataframe.head(3).to_json(),
}

# Kickoff the crew
result = crew.kickoff(inputs)
print(result)
