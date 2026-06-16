import os
import pandas as pd
import requests

DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def download_dataset():
    """Downloads the telecom churn dataset if it does not already exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at: {DATA_DIR}")

    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        try:
            response = requests.get(DATA_URL, timeout=30)
            response.raise_for_status()
            with open(DATA_FILE, "wb") as f:
                f.write(response.content)
            print(f"Dataset downloaded successfully and saved to: {DATA_FILE}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    else:
        print(f"Dataset already exists at: {DATA_FILE}")

def load_data():
    """Downloads and loads the dataset into a pandas DataFrame."""
    download_dataset()
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except Exception as e:
        print(f"Error loading dataset into pandas: {e}")
        raise

def display_info(df):
    """Displays key information about the dataset."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("Data Types & Non-Null Counts:")
    print("-" * 30)
    print(df.info())
    print()
    
    print("Missing Values Per Column:")
    print("-" * 30)
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values detected in original DataFrame.")
    print()
    
    print("Sample Data (First 5 Rows):")
    print("-" * 30)
    print(df.head())
    print()
    
    print("Summary Statistics (Numerical Columns):")
    print("-" * 30)
    print(df.describe())
    print("=" * 60)

if __name__ == "__main__":
    # Test execution
    df = load_data()
    display_info(df)
