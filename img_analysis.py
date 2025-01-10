#%%
import os
import base64
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Load environment variables from .env file
print(load_dotenv())
#%%

# llm = ChatGroq(model="llama-3.2-11b-vision-preview", api_key = "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")
# llm = Groq(api_key=os.getenv("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"))
llm = ChatOllama(model="llama3.2-vision" )
def analyze_image(image_path):
    """Analyze the image using Llama 3.2-11B Vision model via Groq API."""
    # Load the image
    # load_image(image_path)
    
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
    
    # Prepare the prompt for the model
    prompt = f"What is in this image? Here is the image data: data:image/png;base64,{base64_image}"
    
    # Call the LLM to get the analysis result
    result = llm.invoke(prompt)
    
    return result


if __name__ == "__main__":
    # Path to your PNG file
    png_file_path = "images/mttr1.jpg"  # Replace with your actual file path
    
    try:
        result = analyze_image(png_file_path)
        print("Analysis Result:", result)
    except Exception as e:
        print("An error occurred:", e)
