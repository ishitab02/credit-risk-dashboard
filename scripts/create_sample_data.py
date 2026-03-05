"""
Creates a sampled dataset for deployment.
Reduces data size while preserving all functionality.
"""

import pandas as pd
import os

def create_sample():
    print("Creating sampled dataset for deployment...")

    # Create output directory
    os.makedirs("data/sample", exist_ok=True)

    # Sample applications (50k rows, stratified by TARGET)
    print("Sampling applications...")
    df = pd.read_csv("data/raw/application_train.csv")

    # Stratified sample to preserve default rate
    df_default = df[df['TARGET'] == 1].sample(n=4000, random_state=42)
    df_repaid = df[df['TARGET'] == 0].sample(n=46000, random_state=42)
    df_sample = pd.concat([df_default, df_repaid]).sample(frac=1, random_state=42)

    df_sample.to_csv("data/sample/application_train.csv", index=False)
    print(f"  Applications: {len(df_sample):,} rows (was {len(df):,})")

    # Get sampled IDs
    sample_ids = set(df_sample['SK_ID_CURR'])

    # Sample bureau (only matching IDs)
    print("Sampling bureau...")
    bureau = pd.read_csv("data/raw/bureau.csv")
    bureau_sample = bureau[bureau['SK_ID_CURR'].isin(sample_ids)]
    bureau_sample.to_csv("data/sample/bureau.csv", index=False)
    print(f"  Bureau: {len(bureau_sample):,} rows (was {len(bureau):,})")

    # Sample previous_application (only matching IDs)
    print("Sampling previous_application...")
    prev = pd.read_csv("data/raw/previous_application.csv")
    prev_sample = prev[prev['SK_ID_CURR'].isin(sample_ids)]
    prev_sample.to_csv("data/sample/previous_application.csv", index=False)
    print(f"  Previous apps: {len(prev_sample):,} rows (was {len(prev):,})")

    # Copy description file
    print("Copying description file...")
    desc = pd.read_csv("data/raw/HomeCredit_columns_description.csv", encoding='latin-1')
    desc.to_csv("data/sample/HomeCredit_columns_description.csv", index=False)

    # Check sizes
    print("\nSample sizes:")
    for f in os.listdir("data/sample"):
        size = os.path.getsize(f"data/sample/{f}") / (1024*1024)
        print(f"  {f}: {size:.1f} MB")

    print("\nDone! Update db_setup.py to use 'data/sample/' for deployment.")

if __name__ == "__main__":
    create_sample()
