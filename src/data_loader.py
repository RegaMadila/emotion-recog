import pandas as pd
import glob
import os

def load_data(data_path):
    """
    Loads and concatenates all CSV files from a given directory.
    """
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} does not exist.")
        return pd.DataFrame()

    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {data_path}")
        return pd.DataFrame()

    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def clean_initial_data(df):
    """
    Performs initial cleaning on the dataframe.
    """
    if df.empty:
        return df
        
    emotion_columns = ['anger', 'confusion', 'disgust', 'fear', 'joy', 'love', 'sadness', 'surprise']
    
    # Ensure relevant columns exist
    available_cols = [col for col in emotion_columns if col in df.columns]
    if not available_cols:
        return df
    
    # Filter rows to ensure at least one target emotion is present.
    
    df_clean = df.copy()
    

    if 'text' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['text'])
        df_clean = df_clean[df_clean['text'].str.strip() != '']
    
    return df_clean
