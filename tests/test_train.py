"""
tests/test_train.py
-------------------
Unit tests for src/train.py — specifically the pure helper functions
`build_features()` and `evaluate_model()`.
No MLflow server, no Optuna, no file I/O needed.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.train import EXPECTED_FEATURE_COLUMNS, build_features, evaluate_model

# ---------------------------------------------------------------------------
# build_features() tests
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_returns_four_splits(self, processed_df):
        result = build_features(processed_df)
        assert len(result) == 4, "Expected (X_train, X_test, y_train, y_test)"

    def test_raises_on_missing_input_column(self, processed_df):
        """build_features() should raise ValueError if a required column is missing."""
        bad_df = processed_df.drop(columns=["HourOfDay"])
        with pytest.raises(ValueError, match="missing required columns"):
            build_features(bad_df)

    def test_no_target_in_features(self, processed_df):
        X_train, X_test, y_train, y_test = build_features(processed_df)
        assert "isFraud" not in X_train.columns
        assert "isFraud" not in X_test.columns

    def test_target_is_binary(self, processed_df):
        _, _, y_train, y_test = build_features(processed_df)
        all_labels = pd.concat([y_train, y_test])
        assert set(all_labels.unique()).issubset({0, 1})

    def test_train_test_split_ratio(self, processed_df):
        """Default test_size=0.2 — train should be ~80% of total rows."""
        X_train, X_test, _, _ = build_features(processed_df)
        total = len(X_train) + len(X_test)
        assert len(X_test) == pytest.approx(total * 0.2, abs=1)

    def test_feature_columns_are_numeric(self, processed_df):
        X_train, _, _, _ = build_features(processed_df)
        assert all(np.issubdtype(dt, np.number) for dt in X_train.dtypes)

    def test_type_column_is_encoded(self, processed_df):
        """'type' should be label-encoded to integers, not string categories."""
        X_train, X_test, _, _ = build_features(processed_df)
        combined = pd.concat([X_train, X_test])
        assert np.issubdtype(combined["type"].dtype, np.number)

    def test_no_nan_after_encoding(self, processed_df):
        X_train, X_test, y_train, y_test = build_features(processed_df)
        for arr in [X_train, X_test]:
            assert not arr.isnull().any().any(), "NaNs found in feature matrix"

    def test_does_not_mutate_input(self, processed_df):
        original_cols = list(processed_df.columns)
        original_len = len(processed_df)
        build_features(processed_df)
        assert list(processed_df.columns) == original_cols
        assert len(processed_df) == original_len


# ---------------------------------------------------------------------------
# evaluate_model() tests
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def _fit_model(self, processed_df):
        X_train, X_test, y_train, y_test = build_features(processed_df)
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def test_returns_float(self, processed_df):
        model, X_test, y_test = self._fit_model(processed_df)
        acc = evaluate_model(model, X_test, y_test)
        assert isinstance(acc, float)

    def test_accuracy_in_valid_range(self, processed_df):
        model, X_test, y_test = self._fit_model(processed_df)
        acc = evaluate_model(model, X_test, y_test)
        assert 0.0 <= acc <= 1.0

    def test_perfect_predictions_give_1(self):
        """If predictions exactly match labels, accuracy must be 1.0."""
        import numpy as np
        from sklearn.dummy import DummyClassifier

        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([0, 0, 1, 1])
        # Manually construct a model whose predict() we can control
        model = DecisionTreeClassifier(random_state=0)
        model.fit(X, y)
        # Force evaluate_model against the training data (trivially perfect)
        acc = evaluate_model(model, X, y)
        assert acc == 1.0

    def test_all_wrong_predictions_give_0(self):
        """If every prediction is wrong, accuracy should be 0."""
        import numpy as np

        class AlwaysWrong:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        y_test = pd.Series([1, 1, 1, 1])
        X_test = pd.DataFrame({"a": [1, 2, 3, 4]})
        acc = evaluate_model(AlwaysWrong(), X_test, y_test)
        assert acc == 0.0
