#%%
import pandas as pd

dataframe = pd.DataFrame({"sales": [100, 200, 300], "region": ["North", "South", "West"]})
dataframe.to_json()
# %%
