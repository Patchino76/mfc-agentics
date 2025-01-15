import pandas as pd
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph
import ollama

from synthetic_df import gen_synthetic_df


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
    sample_df = state['dataframe'].head(5)  # Take a small sample of the DataFrame
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

        Provide the Python function as a single string that can be executed using the exec function. 
        The function should accept a pd.DataFrame object with the same structure as {sample_df} as a parameter. 
        Return only the Python function as a string and do not try to execute the code. 
        Do not add sample dataframes, function descriptions and do not add calls to the function.
        """
            }
        ]
    try:
        # Call the Ollama API to generate Python code
        response = ollama.chat("llama3.1", messages=prompt)
        # print("API Response:", response)  # Debugging step to check response structure
        

        # Extract the Python code from the response
        if 'message' in response and 'content' in response['message']:
            raw_code = response['message']['content']
            raw_code = raw_code.strip().strip("```").replace("python\n", "").replace("python\r\n", "")
            print("refined Code:", raw_code)
        else:
            raise ValueError("Unexpected response format from Ollama.")

        logs.append(f"Generated code: {raw_code}")
    except Exception as e:
        logs.append(f"Error generating code: {str(e)}")

    return {'generated_code': raw_code, 'logs': logs}

def execute_generated_code(state: AnalysisState):
    dataframe = state['dataframe']
    generated_code = state['generated_code']
    logs = state.get('logs', [])

    try:
        # Prepare a local namespace to execute the code
        local_namespace = {}
        exec(generated_code, globals(), local_namespace)
        
        # Assume the function name is known or can be dynamically extracted
        func_name = next(iter(local_namespace))  # Extract the first defined function
        generated_function = local_namespace[func_name]

        # Call the function with the DataFrame as an argument
        result = generated_function(dataframe)
        print("dataframe:", dataframe.head())
        logs.append(f"Execution result: {result}")
    except Exception as e:
        logs.append(f"Error executing generated code: {str(e)}")
        result = None

    # Update the state with the execution result
    return {'result': result, 'logs': logs}


# Create the graph
graph = StateGraph(AnalysisState)
graph.add_node("generate_code", generate_code)
graph.add_node("execute_generated_code", execute_generated_code)
graph.add_edge("generate_code", "execute_generated_code")
graph.set_entry_point("generate_code")
graph.set_finish_point("execute_generated_code")

# Compile the graph
app = graph.compile()

# Example usage
df = pd.DataFrame({
    'A': [10, 30, 60, 100],
    'B': [1, 2, 3, 4],
    'C': [100, 200, 300, 400]
})
df = gen_synthetic_df()

initial_state = {
    'dataframe': df,
    'query': "Calc the total downtimes for stream 1 ",  # User query
    'generated_code': '',
    'result': '',
    'logs': []
}

# Run the graph
result = app.invoke(initial_state)

# Display results
print("Generated Code:")
print(result['generated_code'])

print("Execution Result:")
print(result['result'])

print("Logs:")
print("\n".join(result['logs']))

