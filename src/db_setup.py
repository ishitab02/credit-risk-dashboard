import os
import pandas as pd
from sqlalchemy import create_engine


def setup_database(use_sample=None):
    """Creates SQLite database from CSV files. use_sample=None auto-detects data source."""
    db_path = "database/credit_risk.db"

    if os.path.exists(db_path):
        print(f"Database already exists at {db_path}")
        return

    os.makedirs("database", exist_ok=True)

    if use_sample is None:
        if os.path.exists("data/sample/application_train.csv"):
            data_dir = "data/sample"
            print("Using SAMPLE data (deployment mode)")
        elif os.path.exists("data/raw/application_train.csv"):
            data_dir = "data/raw"
            print("Using RAW data (full dataset)")
        else:
            raise FileNotFoundError("No data found in data/sample/ or data/raw/")
    elif use_sample:
        data_dir = "data/sample"
    else:
        data_dir = "data/raw"

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False}
    )

    print("Loading applications... ", end="", flush=True)
    app_chunks = pd.read_csv(
        f"{data_dir}/application_train.csv",
        chunksize=10000
    )
    first_chunk = True
    total_rows = 0
    for chunk in app_chunks:
        chunk.to_sql(
            "applications",
            engine,
            if_exists="replace" if first_chunk else "append",
            index=False,
            chunksize=1000
        )
        first_chunk = False
        total_rows += len(chunk)
    print(f"done ({total_rows} rows)")

    print("Loading bureau... ", end="", flush=True)
    bureau_chunks = pd.read_csv(
        f"{data_dir}/bureau.csv",
        chunksize=10000
    )
    first_chunk = True
    total_rows = 0
    for chunk in bureau_chunks:
        chunk.to_sql(
            "bureau",
            engine,
            if_exists="replace" if first_chunk else "append",
            index=False,
            chunksize=1000
        )
        first_chunk = False
        total_rows += len(chunk)
    print(f"done ({total_rows} rows)")

    print("Loading previous_application... ", end="", flush=True)
    prev_chunks = pd.read_csv(
        f"{data_dir}/previous_application.csv",
        chunksize=10000
    )
    first_chunk = True
    total_rows = 0
    for chunk in prev_chunks:
        chunk.to_sql(
            "previous_application",
            engine,
            if_exists="replace" if first_chunk else "append",
            index=False,
            chunksize=1000
        )
        first_chunk = False
        total_rows += len(chunk)
    print(f"done ({total_rows} rows)")

    print(f"\nDatabase created successfully at {db_path}")

    test_df = pd.read_sql("SELECT COUNT(*) as cnt FROM applications", engine)
    print(f"Verification: applications table has {test_df['cnt'].iloc[0]} rows")


if __name__ == "__main__":
    setup_database()
