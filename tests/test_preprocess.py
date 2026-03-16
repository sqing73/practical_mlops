"""
tests/test_preprocess.py
------------------------
Unit tests for src/preprocess.py — specifically the pure `transform()` function.
No file I/O, no Kaggle credentials needed.
"""
import pandas as pd
import pytest

from src.preprocess import EXPECTED_OUTPUT_COLUMNS, transform

# ---------------------------------------------------------------------------
# Column-level tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_drops_nameOrig(self, processed_df):
        assert "nameOrig" not in processed_df.columns

    def test_drops_nameDest(self, processed_df):
        assert "nameDest" not in processed_df.columns

    def test_drops_isFlaggedFraud(self, processed_df):
        assert "isFlaggedFraud" not in processed_df.columns

    def test_new_columns_present(self, processed_df):
        assert "errorbalanceOrg" in processed_df.columns
        assert "errorbalanceDest" in processed_df.columns
        assert "HourOfDay" in processed_df.columns

    def test_all_expected_columns_present(self, processed_df):
        for col in EXPECTED_OUTPUT_COLUMNS:
            assert col in processed_df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Row-filtering tests
# ---------------------------------------------------------------------------

class TestRowFiltering:
    def test_only_transfer_and_cash_out_remain(self, processed_df):
        """After transform, only TRANSFER and CASH_OUT rows should remain."""
        allowed = {"TRANSFER", "CASH_OUT"}
        remaining_types = set(processed_df["type"].astype(str).unique())
        assert remaining_types <= allowed

    def test_payment_rows_filtered(self, raw_df, processed_df):
        n_payment = (raw_df["type"] == "PAYMENT").sum()
        assert n_payment > 0, "fixture has no PAYMENT rows — test is vacuous"
        # none should appear in output
        assert "PAYMENT" not in processed_df["type"].astype(str).values

    def test_row_count_matches_transfer_plus_cash_out(self, raw_df, processed_df):
        expected = raw_df[raw_df["type"].isin(["TRANSFER", "CASH_OUT"])].shape[0]
        assert len(processed_df) == expected


# ---------------------------------------------------------------------------
# Derived-column accuracy tests
# ---------------------------------------------------------------------------

class TestDerivedColumns:
    def test_errorbalanceOrg_formula(self, raw_df, processed_df):
        """errorbalanceOrg = newbalanceOrig + amount - oldbalanceOrg"""
        # Re-compute manually from the raw filtered slice
        raw_filtered = raw_df[raw_df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
        expected = (raw_filtered["newbalanceOrig"]
                    + raw_filtered["amount"]
                    - raw_filtered["oldbalanceOrg"])
        actual = processed_df["errorbalanceOrg"].reset_index(drop=True)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_errorbalanceDest_formula(self, raw_df, processed_df):
        """errorbalanceDest = oldbalanceDest + amount - newbalanceDest"""
        raw_filtered = raw_df[raw_df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
        expected = (raw_filtered["oldbalanceDest"]
                    + raw_filtered["amount"]
                    - raw_filtered["newbalanceDest"])
        actual = processed_df["errorbalanceDest"].reset_index(drop=True)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_HourOfDay_is_step_mod_24(self, raw_df, processed_df):
        raw_filtered = raw_df[raw_df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
        expected = raw_filtered["step"] % 24
        actual = processed_df["HourOfDay"].reset_index(drop=True)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_HourOfDay_range(self, processed_df):
        assert processed_df["HourOfDay"].between(0, 23).all()


# ---------------------------------------------------------------------------
# Schema-guard tests (input validation & output contract)
# ---------------------------------------------------------------------------

class TestSchemaGuards:
    def test_raises_on_missing_input_column(self, raw_df):
        """transform() should raise ValueError if a required raw column is missing."""
        bad_df = raw_df.drop(columns=["isFlaggedFraud"])
        with pytest.raises(ValueError, match="missing required columns"):
            transform(bad_df)

    def test_raises_on_multiple_missing_columns(self, raw_df):
        bad_df = raw_df.drop(columns=["nameOrig", "nameDest"])
        with pytest.raises(ValueError, match="missing required columns"):
            transform(bad_df)

    def test_expected_output_columns_matches_actual_output(self, processed_df):
        """EXPECTED_OUTPUT_COLUMNS should be a subset of the actual output."""
        from src.preprocess import EXPECTED_OUTPUT_COLUMNS
        for col in EXPECTED_OUTPUT_COLUMNS:
            assert col in processed_df.columns, f"Missing expected output column: {col}"


# ---------------------------------------------------------------------------
# Idempotency / side-effect tests
# ---------------------------------------------------------------------------

class TestSideEffects:
    def test_transform_does_not_mutate_input(self, raw_df):
        original_cols = list(raw_df.columns)
        original_len = len(raw_df)
        _ = transform(raw_df)
        assert list(raw_df.columns) == original_cols
        assert len(raw_df) == original_len

    def test_empty_dataframe_returns_empty(self):
        """An all-PAYMENT DataFrame should produce an empty output."""
        empty_raw = pd.DataFrame(columns=[
            "step", "type", "amount",
            "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest",
            "isFraud", "isFlaggedFraud",
        ])
        result = transform(empty_raw)
        assert len(result) == 0
