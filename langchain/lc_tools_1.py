#%%
from langchain_community.tools import BearlyInterpreterTool
from langchain.agents import AgentType, initialize_agent
from langchain_ollama import ChatOllama
from synthetic_df import gen_synthetic_df

#%%
df = gen_synthetic_df()
llm = ChatOllama(model="granite3.1-dense:8b", temperature=0)
bearly_tool = BearlyInterpreterTool(api_key="...")
# %%
bearly_tool.add_file(
    source_path="../docs/agents.pdf", target_path="agents.pdf", description=""
)
tools = [bearly_tool.as_tool()]
tools[0].name
# %%
print(tools[0].description)
# %%
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)
# %%
agent.invoke("What is the dataframe {df}?")
# %%
