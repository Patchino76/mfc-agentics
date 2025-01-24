#%%
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from synthetic_df import gen_synthetic_df
import os

load_dotenv(override=True)

# Configure the Gemini API
genai.configure(api_key="AIzaSyD-S0ajn_qCyVolBLg0mQ83j0ENoqznMX0")

# Initialize the model
MODEL = "gemini-2.0-flash-thinking-exp-01-21"
llm = genai.GenerativeModel(model_name=MODEL)  # or use your specific model name

# %%

# Test the LLM
# response = llm.generate_content("Tell me a short joke about programming")
# print(response.text)
# %%

example_df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Score": [90, 85, 88]
})
query = 'Каква е продължителността на престоите по категории?'
# sample_df = example_df.to_string()
sample_df = gen_synthetic_df().head().to_string()

# Prepare the prompt for Gemini
prompt = f"""You are an expert Python developer and data analyst. Based on the user's query and the provided DataFrame sample, 
generate Python function code to perform the requested analysis. 

User Query: {query}
Sample DataFrame used only to infer the structure of the DataFrame:
{sample_df}

Provide the Python function as a single string that can be executed using the exec function. 
The function should accept a pd.DataFrame object with the same structure as the sample DataFrame as a parameter. 
Return only the Python function as a string and do not try to execute the code. 
Do not add sample dataframes, function descriptions and do not add calls to the function.

If you create a plot function, do not use plt.show(), instead return the image in base64 format using the base64 and BytesIO libraries.
If returning a base64 string do not add 'data:image/png;base64' to it."""

# Generate response using Gemini
response = llm.generate_content(prompt)
print(response.text)
# %%
