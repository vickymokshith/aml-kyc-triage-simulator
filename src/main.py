"""Command-line interface for AML/KYC alert triage simulation.

This script loads synthetic alert, customer and transaction data,
prepares features, trains a logistic regression model, generates
priority scores, and writes the results to an output CSV. Run this
module as a script to execute the full workflow.
"""

import pandas as pd
from pathlib import Path

from triage_model import prepare_features, train_model, predict_priority


def main() -> None:
    """Run the triage simulation end-to-end."""
    # Define data directories
    data_dir = Path(__file__).resolve().parents[1] / "data"
    raw_dir = data_dir / "raw"
    output_dir = data_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    alerts = pd.read_csv(raw_dir / "alerts.csv")
    customers = pd.read_csv(raw_dir / "customers.csv")
    transactions = pd.read_csv(raw_dir / "transactions.csv")

    # Prepare features and train model
    X, y = prepare_features(alerts, customers, transactions)
    if y is None:
        raise ValueError("priority_flag column is missing in alerts; cannot train model")
    model = train_model(X, y)

    # Predict priority scores
    priority_scores = predict_priority(model, X)

    # Save results
    results = alerts[["alert_id", "customer_id", "alert_type"]].copy()
    results["priority_score"] = priority_scores
    results.to_csv(output_dir / "priority_scores.csv", index=False)

    print(f"Wrote {len(results)} priority scores to {output_dir / 'priority_scores.csv'}")


if __name__ == "__main__":
    main()
