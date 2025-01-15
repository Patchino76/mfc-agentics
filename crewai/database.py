#%%
import pandas as pd
import sqlite3

class DataBase:
    def __init__(self, db_path: str):
        """
        Initialize the DataBase class with a path to the SQLite database.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path

    def get_dataframe(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.

        Args:
            query (str): SQL query to execute.

        Returns:
            pd.DataFrame: DataFrame containing the query results.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
