"""Model and utilities for AML/KYC alert triage.

This module provides helper functions to prepare features from alert,
customer, and transaction data, train a simple logistic regression
model, and generate priority scores for each alert. Only synthetic
data is used to illustrate typical data preparation steps and basic
modeling for AML and KYC alert prioritization.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression



def prepare_features(
    alerts: pd.DataFrame, customers: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Prepare feature matrix and target vector for model training.

    This function joins alert records with customer risk information and
    aggregated transaction statistics. It encodes categorical values into
    numeric representations suitable for logistic regression.

    Parameters
    ----------
    alerts: DataFrame
        Alert-level information including ``customer_id``, ``alert_type``
        and optional ``priority_flag`` indicating whether an alert was
        previously labelled as high priority.
    customers: DataFrame
        Customer-level demographics with a ``risk_category`` column.
    transactions: DataFrame
        Raw transaction data used to derive behavioral features such as
        average transaction amount and total count.

    Returns
    -------
    X : DataFrame
        Feature matrix with numeric columns ``risk_num``, ``is_aml``,
        ``mean_tx_amount``, and ``tx_count``.
    y : Series or None
        Target vector if ``priority_flag`` exists in alerts, otherwise
        ``None``.
    """
    # Aggregate transactions by customer
    tx_agg = (
        transactions.groupby("customer_id")
        .agg(
            mean_tx_amount=("amount", "mean"),
            tx_count=("transaction_id", "size"),
        )
        .reset_index()
    )

    # Map risk categories to numeric values
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    customers = customers.copy()
    customers["risk_num"] = customers["risk_category"].map(risk_map)

    # Merge alerts with customer risk and transaction features
    df = (
        alerts.merge(customers[["customer_id", "risk_num"]], on="customer_id", how="left")
        .merge(tx_agg, on="customer_id", how="left")
    )

    # Fill missing transaction statistics
    df["mean_tx_amount"] = df["mean_tx_amount"].fillna(df["mean_tx_amount"].mean())
    df["tx_count"] = df["tx_count"].fillna(0)

    # Encode alert type: 1 for AML, 0 otherwise
    df["is_aml"] = (df["alert_type"].str.upper() == "AML").astype(int)

    # Feature matrix
    X = df[["risk_num", "is_aml", "mean_tx_amount", "tx_count"]].copy()

    # Target vector if available
    y = df.get("priority_flag")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Train a logistic regression classifier on the provided data.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : Series
        Binary target vector.

    Returns
    -------
    model : LogisticRegression
        Fitted logistic regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def predict_priority(model: LogisticRegression, X: pd.DataFrame) -> pd.Series:
    """Predict priority scores for alerts.

    Parameters
    ----------
    model : LogisticRegression
        Trained classifier.
    X : DataFrame
        Feature matrix.

    Returns
    -------
    Series
        Predicted probability of being a high-priority alert for each row.
    """
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=X.index, name="priority_score")
