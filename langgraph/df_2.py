import pandas as pd
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph
import ollama


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

    # Prepare the prompt for Ollama
    prompt = [
                {
                    "role": "user",
                    "content": f"""
        You are an expert Python developer and data analyst. Based on the user's query and the provided DataFrame sample, 
        generate Python function code to perform the requested analysis. 

        User Query: {query}
        Sample DataFrame used only to infer the structure of the DataFrame:
        {sample_df}

        Provide only the  Python function and nothing else. The function should accepd a pd.Dataframe object 
        with the same structure as {sample_df} as a parameter. 
        Return only the python function and do not try to execute the code.
     """
                }
            ]
    try:
        # Call the Ollama API to generate Python code
        response = ollama.chat("llama3.1", messages=prompt)
        print("API Response:", response)  # Debugging step to check response structure

        # Extract the Python code from the response
        if 'message' in response and 'content' in response['message']:
            raw_code = response['message']['content']
        else:
            raise ValueError("Unexpected response format from Ollama.")

        logs.append(f"Generated code: {raw_code}")
    except Exception as e:
        logs.append(f"Error generating code: {str(e)}")

    return {'generated_code': raw_code, 'logs': logs}


# Create the graph
graph = StateGraph(AnalysisState)
graph.add_node("generate_code", generate_code)
graph.set_entry_point("generate_code")
graph.set_finish_point("generate_code")

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
    'query': "Calculate the correlation between columns A and B",  # User query
    'generated_code': '',
    'result': '',
    'logs': []
}

# Run the graph
result = app.invoke(initial_state)

# Display results
print("Generated Code:")
print(result['generated_code'])

