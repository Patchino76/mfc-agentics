#%%
import os
import base64
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
import ollama
from langchain_ollama import ChatOllama

print(load_dotenv())
#%%
llm_chat = ChatOllama(
    model="todorov/bggpt",
)

def analyze_image_ollama(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    prompt = "Analyse the graph image? Explain in details trends, correlations and patterns that you can see like a data analyst."

    # Call the Ollama chat function
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [base64_image]  # Pass the base64 encoded image here
            }
        ]
    )
    
    return response['message']['content']

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def analyze_image_groq(image_path):
    # Convert the image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Prepare the message for the model
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyse the graph image? Explain in details trends, correlations and patterns that you can see like a data analyst."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
    )
    
    return response.choices[0].message.content

def translate_text(text):
    """Translate the given text from English to Bulgarian."""
    # Prepare the messages for the model
    messages = [
        ("system", "You are a helpful translator. Translate the user text to Bulgarian."),
        ("human", text)
    ]
    
    # Invoke the model with the prepared messages
    response = llm_chat.invoke(messages)
    
    return response.content

if __name__ == "__main__":
    # Path to your PNG file
    png_file_path = "images/sp2.png"  # Replace with your actual file path
    
    result = analyze_image_groq(png_file_path)
    print("Analysis Result:", result)
    result_bg = translate_text(result)
    print("Translated Result:", result_bg)
