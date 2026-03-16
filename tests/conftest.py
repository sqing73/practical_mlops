"""
tests/conftest.py
-----------------
Shared pytest fixtures used across all test modules.
"""
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Raw data fixture (mimics the Kaggle CSV schema exactly)
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]


def _raw_row(step, type_, amount, old_org, new_org, old_dest, new_dest, is_fraud):
    return {
        "step": step,
        "type": type_,
        "amount": amount,
        "nameOrig": "C111",
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "nameDest": "M999",
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": 0,
    }


@pytest.fixture
def raw_df():
    """10-row synthetic raw DataFrame with the full Kaggle schema."""
    rows = [
        _raw_row(1,  "TRANSFER",  500, 1000, 500,   0,  500, 0),
        _raw_row(2,  "CASH_OUT",  200, 300,  100,  50,  250, 1),
        _raw_row(3,  "PAYMENT",   100, 500,  400,   0,    0, 0),  # filtered out
        _raw_row(4,  "TRANSFER", 1000, 2000, 1000,  0, 1000, 0),
        _raw_row(25, "CASH_OUT",  750, 750,    0, 100,  850, 1),  # step=25 → HourOfDay=1
        _raw_row(48, "TRANSFER",  300, 300,    0,   0,  300, 0),  # step=48 → HourOfDay=0
        _raw_row(5,  "DEBIT",      50, 200,  150,   0,    0, 0),  # filtered out
        _raw_row(6,  "TRANSFER",  400, 400,    0,   0,  400, 0),
        _raw_row(7,  "CASH_OUT",  600, 600,    0, 200,  800, 1),
        _raw_row(8,  "CASH_IN",   900, 100, 1000,   0,    0, 0),  # filtered out
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


@pytest.fixture
def processed_df(raw_df):
    """Processed DataFrame (output of transform) built from raw_df."""
    from src.preprocess import transform
    return transform(raw_df)
