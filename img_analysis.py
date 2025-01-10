#%%
import os
import base64
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import ollama

# print(load_dotenv())
#%%

# llm = Groq(api_key=os.getenv("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"))
# llm = ChatGroq(model="llama-3.2-11b-vision-preview", api_key = "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")
# llm = Groq(api_key=os.getenv("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"))
# llm = ChatOllama(model="llama3.2-vision" )
def analyze_image(image_path):
    """Analyze the image using Llama 3.2-11B Vision model via Groq API."""
    
    # Convert the image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Prepare the message for the model
    response = llm.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
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

def analyze_image2(image_path):
    
    # Convert the image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    message = f"Analyze this image: data:image/png;base64,{base64_image}"
    # Call the LLM to get the analysis result
    result = llm.invoke(message)
    
    return result

def analyze_image3(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    prompt = "Analyze this image and describe what you see, including any objects, colors, and any text you can detect."

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


if __name__ == "__main__":
    # Path to your PNG file
    png_file_path = "images/mttr1.jpg"  # Replace with your actual file path
    
    try:
        result = analyze_image3(png_file_path)
        print("Analysis Result:", result)
    except Exception as e:
        print("An error occurred:", e)
