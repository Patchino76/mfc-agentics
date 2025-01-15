#%%
import ollama

# %%
try:
    # Replace "llama3.1:latest" with the actual model name
    test_response = ollama.chat("llama3.1:latest", "Hello, test prompt.")
    print("Connection successful:", test_response)
except Exception as e:
    print("Connection error:", str(e))
# %%
