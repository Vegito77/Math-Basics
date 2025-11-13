from sklearn.datasets import fetch_openml
import pandas as pd
import os

def load_boston_data(save_to_csv=True):
    """
    Fetches the Boston Housing dataset from OpenML.
    Optionally saves it to data/raw/boston.csv.
    Returns the pandas DataFrame.
    """
    print("📥 Fetching Boston Housing dataset from OpenML...")
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    df = boston.frame

    if save_to_csv:
        os.makedirs("data/raw", exist_ok=True)
        csv_path = "data/raw/boston.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved dataset to {csv_path}")

    return df


if __name__ == "__main__":
    df = load_boston_data()
    print(df.head())