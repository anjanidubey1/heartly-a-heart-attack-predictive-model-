import pandas as pd
from typing import Tuple

def load_and_clean_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Loads the medical dataset from a specified path, cleans it, renames columns,
    and separates features (X) from the target (y).

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: A tuple containing the
        cleaned dataframe, the features (X), and the target (y).
    """
    try:
        
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None, None, None

    
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')

  
    if 'result' in df.columns:
        df = df.rename(columns={'result': 'target'})
    else:
        print("Error: 'Result' column not found in the dataframe.")
        return None, None, None

  
    df['target'] = df['target'].map({'positive': 1, 'negative': 0})

    
    df.dropna(inplace=True)
    df['target'] = df['target'].astype(int)

    
    df = df.drop_duplicates()
    
   
    X = df.drop('target', axis=1)
    
    
    y = df['target']
    
    print("Data loaded and cleaned successfully.")
    return df, X, y

if __name__ == '__main__':
    
    df, X, y = load_and_clean_data('Medicaldataset.csv')
    if df is not None:
        print("--- First 5 rows of the DataFrame ---")
        print(df.head())
        print("\n--- Info about the DataFrame ---")
        df.info()
        print("\n--- Shape of features (X) ---")
        print(X.shape)
        print("\n--- Shape of target (y) ---")
        print(y.shape)
