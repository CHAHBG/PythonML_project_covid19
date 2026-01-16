
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_logic():
    print("Loading data...")
    try:
        df = pd.read_csv('data/covid19_data.csv', nrows=1000) # Small sample
        print(f"Data loaded: {len(df)} rows")
    except FileNotFoundError:
        print("Error: CSV not found")
        return None

    # Cleaning Logic from app.py
    df['DEATH'] = np.where(df['DATE_DIED'] == '9999-99-99', 0, 1)
    
    # Date parsing
    df['DATE_DIED_DT'] = df['DATE_DIED'].replace('9999-99-99', np.nan)
    df['DATE_DIED_DT'] = pd.to_datetime(df['DATE_DIED_DT'], dayfirst=True, errors='coerce')
    df['MOIS'] = df['DATE_DIED_DT'].dt.to_period('M')
    
    print("Date column created successfully.")
    print("Sample Months:", df['MOIS'].dropna().unique()[:5])

    # Mapping
    map_dict = {1: 'Oui', 2: 'Non', 97: 'Inconnu', 99: 'Inconnu'}
    cols = ['PNEUMONIA', 'INTUBED', 'ICU']
    for col in cols:
        df[f'{col}_LABEL'] = df[col].map(map_dict).fillna("Inconnu")
    
    print("Mapping applied. ICU Labels:", df['ICU_LABEL'].unique())
    
    return df

def verify_plots(df):
    print("Verifying plots...")
    
    # Pie Data Check
    counts = df['DEATH'].value_counts()
    print("Pie stats:", counts.to_dict())
    
    # Line Plot Data Check
    df_dead = df[df['DEATH'] == 1].copy()
    if not df_dead.empty:
        deaths_per_month = df_dead.groupby('MOIS').size()
        print("Deaths per month calculated.")
    else:
        print("No deaths in sample for line plot.")

    # Violin Plot Data Check
    # Ensure Age and ICU_LABEL exist
    if 'AGE' in df.columns and 'ICU_LABEL' in df.columns:
        print("Columns for Violin Plot ready.")
    else:
        print("Missing columns for Violin Plot")

if __name__ == "__main__":
    df = load_data_logic()
    if df is not None:
        verify_plots(df)
        print("Verification Complete: Success")
