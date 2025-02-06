#%%
from data.dispatchers_data import create_data_prompt, load_dispatchers_data
import pandas as pd
# %%

df = load_dispatchers_data()
print(df.head(10))
# %%
def solve_user_query(df):
    # Group by date and find minimum production for each day
    df['date'] = df.index.date
    daily_min = df.groupby('date')['CopperConcentrate'].min().reset_index()
    
    # Sort by copper concentrate to get the lowest production days
    daily_min_sorted = daily_min.sort_values(by='CopperConcentrate', ascending=True).head(5)
    
    # Format the results
    result_df = daily_min_sorted.copy()
    result_df.columns = ['Дата', 'Добит меден концентрат']
    result_df['Добит меден концентрат'] = result_df['Добит меден концентрат'].apply(lambda x: round(x, 2))
    
    return result_df

# rez = solve_user_query(df)
# print(rez)
# %%
def analyze_data(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io
    import base64

    g = sns.jointplot(data=df, x='ProcessedOreMFC', y='CopperConcentrate', kind='reg')
    g.ax_joint.set_xlabel('Преработена руда в цех МФЦ')
    g.ax_joint.set_ylabel('Добит меден концентрат')
    g.fig.suptitle('Взаимна хистограма на преработка и добит меден концентрат', y=1.02)

    buf = io.BytesIO()
    g.fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

image_base64 = analyze_data(df)
print(image_base64)
# %%
