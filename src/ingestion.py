import argparse
from pathlib import Path

import kaggle

root = Path(__file__).parent.parent

def download_data(dataset):
    cred_path = root / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        raise FileNotFoundError("Kaggle credentials not found")

    output_dir = root / "data" / "raw" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(dataset, path=output_dir, unzip=True, quiet=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from Kaggle")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to download")
    args = parser.parse_args()
    download_data(args.dataset)

