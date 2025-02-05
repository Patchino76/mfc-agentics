#%%
import pandas as pd
import io


# %%
data = {
    "category": ["Поток 1", "Поток 2"],
    "stream_name": ["Поток 1", "Поток 2"],
    "електро": [285, 366],
    "механо": [264, 394],
    "ппр": [511, 470],
    "системни": [386, 433],
    "технологични": [426, 459],
}
df = pd.DataFrame(data)
# %%
df.attrs["column_descriptions"] = {
    "category": "Категория на потока",
    "stream_name": "Име на потока",
    "електро": "Електро описание",
    "механо": "Механични данни",
    "ппр": "Информация за ППР",
    "системни": "Системни данни",
    "технологични": "Технологични данни"
}
# %%
df_head_str = df.head().to_string()
print(df_head_str)
# %%
# Capture df.info() output
buffer = io.StringIO()
df.info(buf=buffer)
df_info_str = buffer.getvalue()
print(df_info_str)
# %%
prompt = (
    "Below is information about the DataFrame:\n\n"
    "DataFrame Head:\n"
    f"{df_head_str}\n\n"
    "DataFrame Info:\n"
    f"{df_info_str}\n\n"
    "DataFrame Attributes:\n"
    f"{df.attrs}\n"
)

# Now you can pass `prompt` to the LLM.
print(prompt)
# %%
