import pandas as pd

def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.select_dtypes(include='number').mean())
    return df
