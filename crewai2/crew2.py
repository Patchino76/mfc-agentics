import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool
from crewai import LLM
import pandas as pd
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
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434", 
)

tool = CodeInterpreterTool()

# def format_output(output):
#     data = json.loads(str(output))
#     formatted_output = json.dumps(data, indent=4)
#     print(formatted_output)
def format_output(output):
    try:
        # Convert the output to a string if it's not already
        output_str = str(output)
        if output_str.strip():  # Check if the string is not empty
            # Attempt to load the output as JSON
            data = json.loads(output_str)
            # Pretty-print the JSON data
            formatted_output = json.dumps(data, indent=4)
            print(formatted_output)
        else:
            print("Output is empty or not valid JSON")
    except (json.JSONDecodeError, TypeError) as e:
        print("Error decoding JSON:", e)


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
    allow_delegation=False,
    allow_code_execution=True,
    memory=True, 
    llm=llm2,
    max_iter=2,
)

# Define tasks
generate_code_task = Task(
    description=(
        "You will receive a query and a small sample of the dataframe {dataframe_sample} in JSON format. "
        "Examine the {dataframe_sample} to determine the exact structure of the DataFrame."
        "Generate python code to fulfill the {query} based on the sample."
        "Do not make any assumptions about the DataFrame structure; "
        "Use only the provided sample to infer the structure of the dataframe."
    ),
    expected_output="Python code string that processes the {query}.",
    agent=analytical_agent,
    name="generate_code_task",
)

execute_code_task = Task(
    description=(
        "Receive Python code from the generate_code_task and execute it on the full dataframe object {dataframe_full}, "
        "execute the code, and return the result."
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
# dataframe = pd.DataFrame({"sales": [100, 200, 300], "region": ["North", "South", "West"]})
dataframe = gen_synthetic_df()
print(dataframe.head())
inputs = {
    "query": "This dataframe contains downtime data for different streams and machines. How many unique categories of downtime do we have?",
    "dataframe_sample": dataframe.head(2).to_json(),
    "dataframe_full": dataframe.to_json()
}

# Kickoff the crew
result = crew.kickoff(inputs)
print(result)
