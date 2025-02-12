import pandas as pd
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph
import openai

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Define the state object
class AnalysisState(TypedDict):
    dataframe: pd.DataFrame
    query: str
    generated_code: str
    result: str
    logs: Annotated[list[str], add]

# Define the agent node to generate Python code
def generate_code(state: AnalysisState):
    query = state['query']
    sample_df = state['dataframe'].head(3)  # Take a small sample of the DataFrame
    logs = state.get('logs', [])

    # Prepare the prompt for the LLM
    prompt = f"""
You are an expert Python developer. Based on the user's query and the provided DataFrame sample, generate Python code to perform the requested analysis. 
The generated code should work with pandas and should not modify the DataFrame. 

User Query: {query}
Sample DataFrame:
{sample_df}

Provide only the Python code without any explanation or additional text.
"""
    try:
        # Call the OpenAI API to generate Python code
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use an appropriate engine
            prompt=prompt,
            max_tokens=150,
            temperature=0,
        )
        code = response['choices'][0]['text'].strip()
        logs.append(f"Generated code: {code}")
    except Exception as e:
        code = "result = 'Error generating code.'"
        logs.append(f"Error generating code: {str(e)}")

    return {'generated_code': code, 'logs': logs}

# Define the execution node to run the generated Python code
def execute_code(state: AnalysisState):
    df = state['dataframe']
    code = state['generated_code']
    logs = state.get('logs', [])
    result = None

    try:
        # Execute the generated code in a local namespace
        exec(code, globals(), locals())
        result = locals().get('result', 'No result generated.')
        logs.append(f"Execution successful. Result: {result}")
    except Exception as e:
        logs.append(f"Error during execution: {str(e)}")
        result = 'Execution error.'

    return {'result': result, 'logs': logs}

# Create the graph
graph = StateGraph(AnalysisState)
graph.add_node("generate_code", generate_code)
graph.add_node("execute_code", execute_code)
graph.add_edge("generate_code", "execute_code")
graph.set_entry_point("generate_code")
graph.set_finish_point("execute_code")

# Compile the graph
app = graph.compile()

# Example usage
df = pd.DataFrame({
    'A': [10, 20, 30, 40],
    'B': [1, 2, 3, 4],
    'C': [100, 200, 300, 400]
})

initial_state = {
    'dataframe': df,
    'query': "What is the average of column A?",  # User query
    'generated_code': '',
    'result': '',
    'logs': []
}

# Run the graph
result = app.invoke(initial_state)

# Display results
print("Generated Code:")
print(result['generated_code'])
print("\nResult of Execution:")
print(result['result'])
print("\nExecution Logs:")
print(result['logs'])
