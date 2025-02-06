#%%
import pandas as pd
import io
from rich import print
import os

#%%
def load_dispatchers_data(file_path = "c:/projects/mfc-agentics/langgraph4/data/dispatchers_en_22.csv") -> pd.DataFrame:

    df = pd.read_csv(file_path, index_col=False)
    
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Create a timestamp index with 8-hour frequency
    start_time = pd.Timestamp('2022-01-01 00:00:00')
    df.index = pd.date_range(start=start_time, periods=len(df), freq='8h')
    
    # Add shift column based on time of day
    # Shift 1: 00:00-07:59, Shift 2: 08:00-15:59, Shift 3: 16:00-23:59
    df['Shift'] = pd.cut(df.index.hour, 
                        bins=[-1, 7, 15, 23], 
                        labels=[1, 2, 3],
                        include_lowest=True)
    
    # Move Shift column to the front
    cols = ['Shift'] + [col for col in df.columns if col != 'Shift']
    df = df[cols]
    
    return df

df = load_dispatchers_data()
print(df.head(10))


# %%
def create_data_prompt():
    df = load_dispatchers_data()

    column_descriptions = {
        'Shift' : 'Смяна',
        # Ore input and statuses:
        'DailyOreInput': 'Подадена руда от МГТЛ за денонощието',
        'Stock2Status': 'Състояние на склад №2',
        'CrushedOreSST': 'Натрошена руда от Цех ССТ',
        # Classification:
        'Class15': 'Класа 15',
        'Class12': 'Класа 12',
        # Transportation and processing:
        'TransportedOre': 'Превозена руда до междинни бункери',
        'IntermediateBunkerStatus': 'Състояние на междинни бункери',
        'ProcessedOreMFC': 'Преработена руда в цех МФЦ',
        # Ore properties:
        'OreMoisture': 'Влага на преработената руда',
        'DryProcessedOre': 'Суха преработена руда',
        # Rock types:
        'Granite': 'Грано',
        'Dikes': 'Дайки',
        'Shale': 'Шисти',
        # Grinding classes:
        'GrindingClassPlus0_20mm': 'Смилане класа + 0,20мм',
        'GrindingClassMinus0_08mm': 'Смилане класа -0,08мм',
        # Pulp density:
        'PulpDensity': 'Плътност на пулпа',
        # Copper content measurements:
        'CopperContentOre': 'Съдържание на мед в рудите по Куриер',
        'CopperContentWaste': 'Съдържание на мед в отпадъка по Куриер',
        'CopperContentConcentrate': 'Съдържание на мед в медния к-т Куриер',
        # Extraction methods:
        'TechExtraction': 'Технологично извличане по Куриер',
        'LoadExtraction': 'Товарно извличане',
        # Final concentrate properties:
        'CopperConcentrate': 'Добит меден концентрат',
        'ConcentrateMoisture': 'Влага на медния концентрат',
        'CopperContent': 'Съдържание на мед в медния к-т',
        'MetalCopper': 'Метал мед в медния концентрат',
        'ThickenerWeight': 'Литрово тегло в сгъстителя',
        }


    df.attrs["column_descriptions"] = column_descriptions
    df_head_str = df.head().to_string()
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_str = buffer.getvalue()

    column_descriptions = df.attrs.get("column_descriptions", {})
    prompt = (
        "Below is the structure and a sample of a DataFrame along with metadata for its columns:\n\n"
        "1. Data Sample (first few rows):\n"
        f"{df_head_str}\n\n"
        "2. DataFrame Structure and Info:\n"
        f"{df_info_str}\n\n"
        "3. Column Descriptions (metadata):\n"
        f"{column_descriptions}\n\n"
        "Please analyze this DataFrame structure and metadata."
    )
    return prompt
# %%
prompt = create_data_prompt()
print(prompt)
# %%
