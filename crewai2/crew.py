import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool
from crewai import LLM
import pandas as pd
from dotenv import load_dotenv
import json
from rich import print

print("Loading env vars...")
print(load_dotenv(override=True))

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.1
)

# llm = LLM(
#     model="ollama/llama3.1",
#     base_url="http://localhost:11434", 
# )

tool = CodeInterpreterTool()

def format_output(output):
    data = json.loads(str(output))
    formatted_output = json.dumps(data, indent=4)
    print(formatted_output)


# Define agents
analytical_agent = Agent(
    role="Data Analyst",
    goal="Analyze queries and dataframe descriptions to generate Python code.",
    backstory=(
        "You are an expert Python data analyst. "
        "You will receive a query and a serialized dataframe in JSON format. "
        "Use the provided JSON to determine the structure of the DataFrame and generate Python code "
        "to fulfill the query. Do not make any assumptions about the DataFrame structure. "
    ),
    verbose=True,
    llm=llm,
    memory=True, 
    allow_delegation= True,
    max_iter=2,
    step_callback=format_output
)

code_interpreter_agent = Agent(
    role="Code Executor",
    goal="Execute Python code on provided dataframes and return the results.",
    backstory=(
        "Your expertise lies in evaluating and executing Python scripts to deliver accurate outputs."
    ),
    verbose=True,
    tools=[tool],
    alow_delegation=False,
    allow_code_execution=True,
    memory=True, 
    llm=llm,
    max_iter=2,
)

# Define tasks
generate_code_task = Task(
    description=(
        "You will receive a query and a serialized dataframe {dataframe_json} in JSON format. "
        "Examine the {dataframe_json} to determine the exact structure of the DataFrame."
        "Generate python code to fulfill the {query} based of the dataframe."
        "Do not make any assumptions about the DataFrame structure; "
        "Use only the provided JSON to infer the structure of the dataframe."
    ),
    expected_output="Python code string that processes the {query}.",
    agent=analytical_agent,
    name="generate_code_task",
)

execute_code_task = Task(
    description=(
        "Receive Python code from the generate_code_task and a serialized dataframe object {dataframe_json}, "
        "deserialize the dataframe_json if necessary, execute the code, and return the result."
    ),
    expected_output="Result of the executed code.",
    agent=code_interpreter_agent,
    Context = [generate_code_task]
)

# Create the crew
crew = Crew(
    agents=[analytical_agent, code_interpreter_agent],
    tasks=[generate_code_task, execute_code_task],
    process=Process.sequential
)

# Mock inputs
dataframe = pd.DataFrame({"sales": [100, 200, 300], "region": ["North", "South", "West"]})
print(dataframe)
inputs = {
    "query": "Calculate the average of the 'sales' column.",
    "dataframe_json": dataframe.to_json()
}

# Kickoff the crew
result = crew.kickoff(inputs)
print(result)
