"""Unit tests for the triage model utilities.

These tests validate the feature preparation and prediction functions
within the AML/KYC triage simulator. They use small in-memory
DataFrames to verify that predicted scores are computed and fall
within the [0, 1] probability range.
"""

import pandas as pd
from src.triage_model import prepare_features, train_model, predict_priority


def test_prepare_and_predict() -> None:
    """Ensure that the pipeline produces probability scores between 0 and 1."""
    # Small synthetic dataset
    alerts = pd.DataFrame(
        {
            "alert_id": ["A1", "A2", "A3"],
            "customer_id": ["C1", "C2", "C3"],
            "alert_type": ["AML", "KYC", "AML"],
            "priority_flag": [1, 0, 1],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "risk_category": ["High", "Low", "Medium"],
        }
    )
    transactions = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3"],
            "customer_id": ["C1", "C2", "C3"],
            "amount": [100.0, 50.0, 75.0],
            "tx_type": ["purchase", "deposit", "transfer"],
            "tx_country": ["US", "CA", "UK"],
        }
    )

    X, y = prepare_features(alerts, customers, transactions)
    model = train_model(X, y)
    scores = predict_priority(model, X)
    assert len(scores) == len(alerts)
    # All scores should be between 0 and 1
    assert (scores >= 0).all() and (scores <= 1).all()
