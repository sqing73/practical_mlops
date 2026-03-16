import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

root = Path(__file__).parent.parent

def draw_eda(data_file, artifacts_dir):
    artifacts_dir = root / artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / data_file)
    # Calculate the counts of each class
    fraud_counts = df['isFraud'].value_counts()
    # Create the visualizations
    plt.figure(figsize=(12, 5))
    # Plot 1: Bar Chart (Volume)
    plt.subplot(1, 2, 1)
    sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette='viridis', legend=False, hue=fraud_counts.index)
    plt.title('Count of Fraud vs Legitimate Transactions')
    plt.xlabel('Is Fraud? (0: No, 1: Yes)')
    plt.ylabel('Count')
    # Plot 2: Pie Chart (Ratio)
    plt.subplot(1, 2, 2)
    plt.pie(fraud_counts, labels=['Legitimate', 'Fraud'], autopct='%1.2f%%', startangle=140, colors=['#66b3ff','#ff9999'])
    plt.title('Ratio of Fraud vs Legitimate Transactions')
    plt.tight_layout()
    plt.savefig(artifacts_dir / "fraud_counts.png")
    plt.close()

    # Group by type to see where the fraud cases are occurring
    type_summary = df.groupby('type')['isFraud'].agg(['count', 'sum']).reset_index()
    type_summary.columns = ['type', 'total_transactions', 'fraud_cases']

    # Calculate fraud percentage for insight
    type_summary['fraud_percentage'] = (type_summary['fraud_cases'] / type_summary['total_transactions']) * 100

    # Visualization: Total Transactions vs Fraud Cases (Stacked)
    plt.figure(figsize=(10, 6))

    # We use a log scale because legitimate transactions usually dwarf fraud cases
    ax = sns.barplot(data=type_summary, x='type', y='total_transactions', color='#66b3ff', label='Total Transactions')
    sns.barplot(data=type_summary, x='type', y='fraud_cases', color='#ff9999', label='Fraud Cases')

    ax.set_yscale("log") # Log scale makes the small fraud counts visible
    plt.title('Transactions vs Fraud cases by Type (Log Scale)')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count (Log Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "fraud_counts_by_type.png")
    plt.close()

    # Specific view: Only the fraud cases
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=type_summary[type_summary['fraud_cases'] > 0],
        x='type',
        y='fraud_cases',
        hue='type',
        palette='magma',
        legend=False
    )
    plt.title('Types Containing Fraud Cases')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count of Fraud')
    plt.tight_layout()
    plt.savefig(artifacts_dir / "fraud_only_by_type.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw EDA")
    parser.add_argument("--data-file", type=str, required=True, help="Path to raw data")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Path to artifacts directory")
    args = parser.parse_args()
    draw_eda(args.data_file, args.artifacts_dir)


