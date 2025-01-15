#%%
import matplotlib.pyplot as plt
from typing import List, Any
from pydantic import BaseModel
#%%
class PlotData(BaseModel):
    x: List[Any]
    y: List[Any]
    label: str
    plot_type: str  # e.g., 'line', 'bar', 'scatter'

class PlotConfig(BaseModel):
    title: str
    x_label: str
    y_label: str
    legend: bool

class PlotResult(BaseModel):
    data: List[PlotData]
    config: PlotConfig
#%%
def ai_plot(plot_result: PlotResult):
    plt.figure(figsize=(10, 6))
    
    for plot_data in plot_result.data:
        if plot_data.plot_type == 'line':
            plt.plot(plot_data.x, plot_data.y, label=plot_data.label)
        elif plot_data.plot_type == 'bar':
            plt.bar(plot_data.x, plot_data.y, label=plot_data.label)
        elif plot_data.plot_type == 'scatter':
            plt.scatter(plot_data.x, plot_data.y, label=plot_data.label)
    
    plt.title(plot_result.config.title)
    plt.xlabel(plot_result.config.x_label)
    plt.ylabel(plot_result.config.y_label)
    
    if plot_result.config.legend:
        plt.legend()
    
    plt.show()
#%%
# Example usage
example_plot = PlotResult(
    data=[
        PlotData(
            x=[1, 2, 3, 4, 5],
            y=[10, 20, 30, 40, 50],
            label="Sample Line Plot",
            plot_type="line"
        ),
        PlotData(
            x=[1, 2, 3, 4, 5],
            y=[15, 25, 35, 45, 55],
            label="Sample Bar Plot",
            plot_type="bar"
        )
    ],
    config=PlotConfig(
        title="Sample Plot",
        x_label="X Axis",
        y_label="Y Axis",
        legend=True
    )
)

ai_plot(example_plot)

# %%
