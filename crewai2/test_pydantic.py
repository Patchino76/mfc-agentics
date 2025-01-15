#%%
from typing import Optional, Any

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from crewai.tools import BaseTool
from synthetic_df import gen_synthetic_df
# %%
class DataFrameWrapper(BaseModel):
    """
    A Pydantic model to hold a Pandas DataFrame. The DataFrame is mandatory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Key change here

    df: pd.DataFrame = Field(description="The Pandas DataFrame")

    def print_head(self, n: int = 5):
        """Prints the head of the DataFrame."""
        print(self.df.head(n))
# %%
class DataFrameWrapper(BaseTool):
    """
    A Pydantic model to hold a Pandas DataFrame. The DataFrame is mandatory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Key change here

    df: pd.DataFrame = Field(description="The Pandas DataFrame")

    def print_head(self, n: int = 5):
        """Prints the head of the DataFrame."""
        print(self.df.head(n))
# %%
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import pandas as pd
from langchain.tools import BaseTool

class CustomDataFrameTool(BaseTool):
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    name: str = "custom_dataframe_tool"
    description: str = "Execute generated code on a full DataFrame"
    df: pd.DataFrame = Field(default=None, description="The pandas DataFrame to be processed.")

    def __init__(self, df: pd.DataFrame = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.df = df

    def _run(self, code: str) -> str:
        """Execute the given code on the DataFrame and return the result."""
        try:
            local_env = {'pd': pd, 'df': self.df}
            exec(code, local_env)
            return str(local_env.get('result', 'Code executed successfully'))
        except Exception as e:
            return f"Error executing code: {str(e)}"

    async def _arun(self, code: str):
        raise NotImplementedError("Async version not implemented.")

# Example usage:
full_dataframe = gen_synthetic_df()
df = CustomDataFrameTool(df=full_dataframe)

try:
    tool = CustomDataFrameTool(df=df)
    print("Tool created successfully:")
    print(tool.df)
    result = tool._run("result = df.describe()")
    print(result)

    tool_no_df = CustomDataFrameTool()  # This will NOT raise an error
    print("Tool created without df (using default).")
    if tool_no_df.df is None:
        print("df is None as expected.")
except Exception as e:
    print(f"Unexpected Error: {e}")
# %%
