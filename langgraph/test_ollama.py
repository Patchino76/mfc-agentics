#%%
import ollama

# %%
try:
    # Replace "llama3.1:latest" with the actual model name
    test_response = ollama.chat(model = "granite3.1-dense:8b", 
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],)
    print("Connection successful:", test_response)
except Exception as e:
    print("Connection error:", str(e))
# %%
