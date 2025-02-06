#%%
import pandas as pd
import io
from rich import print

#%%
def load_dispatchers_data(file_path) -> pd.DataFrame:
    # Load the CSV file without using the first column as index
    df = pd.read_csv(file_path, index_col=False)
    
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Create a timestamp index with 8-hour frequency
    start_time = pd.Timestamp.now()
    df.index = pd.date_range(start=start_time, periods=len(df), freq='8h')
    
    return df

df = load_dispatchers_data('dispatchers_en_22.csv')
# print(df.head(2))
print(df.columns)

# %%
column_descriptions = {
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
    'ThickenerWeight': 'Литрово тегло в сгъстителя'
}

# df.attrs = column_descriptions
# print(df.attrs)

# %%
df.attrs["column_descriptions"] = column_descriptions
# print(df.attrs)
# %%
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
print(prompt)
# %%
