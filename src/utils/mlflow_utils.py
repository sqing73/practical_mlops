import argparse
import mlflow
from mlflow.tracking import MlflowClient

def get_best_run(experiment_name, metric_name):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1,
    )
    
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")
    
    return runs[0]

def get_best_accuracy(experiment_name, metric_name):
    best_run = get_best_run(experiment_name, metric_name)
    accuracy = best_run.data.metrics.get(metric_name, 0)
    print(accuracy)

def promote_model(experiment_name, model_name, metric_name, alias):
    best_run = get_best_run(experiment_name, metric_name)
    
    model_uri = f"runs:/{best_run.info.run_id}/model"
    
    # Register (creates version if model already exists)
    mv = mlflow.register_model(model_uri, model_name)
    print(f"Registered: {model_name} v{mv.version} (run {best_run.info.run_id})")
    
    # Set Alias (Replaces Deprecated Stages)
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=mv.version
    )
    print(f"Assigned alias '{alias}' to: {model_name} v{mv.version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow Utilities for CI/CD")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Get best accuracy
    get_acc_parser = subparsers.add_parser("get-best-accuracy", help="Get best accuracy from an experiment")
    get_acc_parser.add_argument("--experiment-name", required=True, help="Name of the MLflow experiment")
    get_acc_parser.add_argument("--metric-name", required=True, help="Metric to sort by (e.g., test_accuracy)")
    
    # Promote model
    promote_parser = subparsers.add_parser("promote-model", help="Promote the best model to a stage")
    promote_parser.add_argument("--experiment-name", required=True, help="Name of the MLflow experiment")
    promote_parser.add_argument("--model-name", required=True, help="Name of the model in registry")
    promote_parser.add_argument("--metric-name", required=True, help="Metric to sort by")
    promote_parser.add_argument("--alias", default="Staging", help="Target alias (default: Staging)")
    
    args = parser.parse_args()
    
    if args.command == "get-best-accuracy":
        get_best_accuracy(args.experiment_name, args.metric_name)
    elif args.command == "promote-model":
        promote_model(args.experiment_name, args.model_name, args.metric_name, args.alias)
    else:
        parser.print_help()
