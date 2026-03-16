import argparse
from pathlib import Path

import pandas as pd

root = Path(__file__).parent.parent


REQUIRED_RAW_COLUMNS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]

EXPECTED_OUTPUT_COLUMNS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFraud",
    "errorbalanceOrg", "errorbalanceDest", "HourOfDay",
]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Pure transformation: takes a raw DataFrame, returns the processed one.
    No file I/O — safe to call in unit tests.
    """
    # --- input schema guard ---
    missing = set(REQUIRED_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    df = df.copy()
    df["type"] = df["type"].astype("category")
    df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    df_new = df.loc[(df.type == "TRANSFER") | (df.type == "CASH_OUT")].copy()
    df_new["errorbalanceOrg"] = df_new.newbalanceOrig + df_new.amount - df_new.oldbalanceOrg
    df_new["errorbalanceDest"] = df_new.oldbalanceDest + df_new.amount - df_new.newbalanceDest
    df_new["HourOfDay"] = df_new.step % 24

    # --- output schema guard ---
    missing_out = set(EXPECTED_OUTPUT_COLUMNS) - set(df_new.columns)
    if missing_out:
        raise RuntimeError(f"transform() produced unexpected output schema, missing: {missing_out}")

    return df_new


def preprocess(data_file, processed_file_dir):
    """I/O wrapper: reads raw CSV, runs transform(), writes result."""
    processed_file_dir = root / processed_file_dir
    processed_file_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    df_new = transform(df)
    df_new.to_csv(processed_file_dir / "data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--data-file", type=str, required=True, help="Path to raw data")
    parser.add_argument("--processed-file-dir", type=str, required=True, help="Path to processed data")
    args = parser.parse_args()
    preprocess(args.data_file, args.processed_file_dir)
