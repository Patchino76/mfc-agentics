#%%
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from database import DataBase
import seaborn as sns
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

# %%
db = DataBase("D:/DataSets/MFC/SQLITE3/db.sqlite3")
query = 'SELECT * FROM MA_FULL_01 LIMIT 1000'
df = db.get_dataframe(query)
df.head()
# %%
llm = ChatOllama(
    model="granite3.1-dense:8b",
)

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type="openai-tools",
    allow_dangerous_code=True,
)

# %%

# Define the query_data function
def query_data(question: str) -> str:
    """
    Query the DataFrame using natural language through the LangChain agent
    
    Args:
        question (str): Natural language question about the data
        
    Returns:
        str: Agent's response
    """
    try:
        response = agent.run(question)
        return response
    except Exception as e:
        return f"Error querying data: {str(e)}"


# result = query_data("What are the column names in the dataset?")
# print(result)

# %%


# Define the query_data function
def query_plot(question: str) -> str:
    """
    Query the DataFrame using natural language through the LangChain agent
    
    Args:
        question (str): Natural language question about the data
        
    Returns:
        str: Agent's response or image in base64 format
    """
    try:
        if "plot" in question.lower() or "histogram" in question.lower():
            # Execute the code to create a plot or histogram
            column_name = question.split()[-1]  # Assuming the column name is the last word in the question
            plt.figure(figsize=(10, 6))
            df[column_name].hist(bins=30)
            plt.title(f'Histogram of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            
            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert the bytes buffer to a base64 string
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return f"data:image/png;base64,{img_str}"
        else:
            response = agent.run(question)
            return response
    except Exception as e:
        return f"Error querying data: {str(e)}"

# Example usage
result = query_plot("Create a histogram of the Ore")
print(result)

# %%
# Function to display the image from base64 string
def display_image(base64_str: str):
    """
    Display the image from a base64 string
    
    Args:
        base64_str (str): Base64 encoded image string
    """
    # Decode the base64 string
    img_data = base64.b64decode(base64_str.split(',')[1])
    
    # Create an image from the decoded bytes
    img = Image.open(io.BytesIO(img_data))
    
    # Display the image
    img.show()


display_image(result)

# %%

# Define the query_plot function using LangChain's parser

def query_plot_langchain(query: str) -> None:
    """
    Generate and display a plot based on a natural language query using LangChain's parser.

    Args:
        query (str): Natural language query describing the plot to generate.
    """
    try:
        # Define a prompt template for parsing
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a data analyst. Extract the type of graph and columns from the following query:
            Query: {query}
            """
        )

        # Initialize the LLM
        llm = Ollama(model="llama2")

        # Parse the query
        response = llm(prompt_template.format(query=query))
        parsed_data = response['text']  # Assuming response contains a 'text' field with the parsed result

        # Example parsing logic (this can be expanded with NLP models)
        if "heatmap" in parsed_data.lower():
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            plt.title("Heatmap of DataFrame")
            plt.show()
        elif "line" in parsed_data.lower():
            columns = parsed_data.lower().split("line trends of ")[1].split(",")
            df[columns].plot(figsize=(10, 6))
            plt.title("Line Trends")
            plt.xlabel("Index")
            plt.ylabel("Values")
            plt.show()
        else:
            print("Query not recognized. Please specify a valid plot type.")
    except Exception as e:
        print(f"Error generating plot: {str(e)}")

# Example usage
# query_plot_langchain("generate a heatmap of the dataframe")
# query_plot_langchain("generate line trends of column1, column2")
