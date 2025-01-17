from typing import TypedDict, Annotated, Sequence
import operator
import json
from langchain_core.messages import BaseMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools.render import format_tool_to_openai_function

# Define state for the LangGraph
class DataFrameToolState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Collect messages for tool interactions
    dataframe: str  # Placeholder for the dataframe (in a serialized format)

# Define Tools
class CodeGeneratorTool:
    """Tool to generate Python code for dataframe analysis."""
    def invoke(self, query: str) -> str:
        # Placeholder: Replace with an actual code generation implementation.
        return f"# Python code generated for: {query}\nprint(df.head())"

class CodeExecutionTool:
    """Tool to execute generated code on a dataframe."""
    def invoke(self, code: str, dataframe: str) -> str:
        # Placeholder: Mock execution, replace with actual execution logic.
        try:
            exec_env = {'df': dataframe}  # Replace `dataframe` with an actual DataFrame object
            exec(code, exec_env)
            return "Execution successful."
        except Exception as e:
            return f"Execution error: {e}"

# Define Tools and Executor
tools = [CodeGeneratorTool(), CodeExecutionTool()]
tool_executor = ToolExecutor(tools)

# Model
model = ChatOllama(model="granite3.1-dense:8b", temperature=0)
functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)

# Nodes
def generate_code(state):
    """Generate Python code for a query."""
    query = state['messages'][-1].content  # Last message contains the user query
    response = tools[0].invoke(query)  # CodeGeneratorTool
    return {"messages": [FunctionMessage(content=response, name="code_generator")]}

def execute_code(state):
    """Execute the generated Python code."""
    last_message = state['messages'][-1]
    code = last_message.content
    dataframe = state.get('dataframe', None)
    response = tools[1].invoke(code, dataframe)  # CodeExecutionTool
    return {"messages": [FunctionMessage(content=response, name="code_executor")]}

def should_continue(state):
    """Determine next step based on execution success or error."""
    last_message = state['messages'][-1]
    if "Execution successful" in last_message.content:
        return "end"
    else:
        return "generate_code"

# Graph Definition
workflow = StateGraph(DataFrameToolState)
workflow.add_node("generate_code", generate_code)
workflow.add_node("execute_code", execute_code)
workflow.set_entry_point("generate_code")

# Add conditional edges
workflow.add_conditional_edges(
    "execute_code",
    should_continue,
    {"generate_code": "generate_code", "end": END}
)

workflow.add_edge("generate_code", "execute_code")

# Compile Graph
app = workflow.compile()

# Example Input
inputs = {
    "messages": [BaseMessage(content="Summarize the data in the dataframe.")],
    "dataframe": "pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})"
}

# Invoke Workflow
result = app.invoke(inputs)
print("Final Result:", result)
