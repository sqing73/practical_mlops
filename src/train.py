import argparse

import mlflow
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

EXPECTED_FEATURE_COLUMNS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorbalanceOrg", "errorbalanceDest", "HourOfDay",
]


def build_features(df: pd.DataFrame):
    """Pure feature engineering: encode + split. No MLflow, no I/O."""
    # --- input schema guard ---
    required = set(EXPECTED_FEATURE_COLUMNS) | {"isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Processed DataFrame is missing required columns: {missing}")

    df = df.copy()
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    df[['step', 'type', 'HourOfDay']] = df[['step', 'type', 'HourOfDay']].astype(float)
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test) -> float:
    """Pure scorer: returns accuracy. No MLflow, no I/O."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def train(data_file):
    mlflow.sklearn.autolog()
    mlflow.set_experiment("Decision Tree Hyperparameter Tuning")

    df = pd.read_csv(data_file)
    X_train, X_test, y_train, y_test = build_features(df)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            DT = DecisionTreeClassifier(**params, random_state=42)
            DT.fit(X_train, y_train)
            accuracy = evaluate_model(DT, X_test, y_test)
            mlflow.log_metric("test_accuracy", accuracy)
        return accuracy

    with mlflow.start_run(run_name="Decision Tree Hyperparameter Tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)
        print(f"Best Score: {study.best_value}")
        print(f"Best Params: {study.best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True)
    args = parser.parse_args()
    train(args.data_file)
