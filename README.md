# # aml-kyc-triage-simulator

![tests](https://github.com/vickymokshith/aml-kyc-triage-simulator/actions/workflows/tests.yml/badge.svg)

## Problem

Financial institutions receive thousands of AML and KYC alerts every day. Analysts need to triage these alerts quickly and accurately to focus on the most suspicious cases. Manual triage is time-consuming and inconsistent.

## Approach

This project simulates an AML/KYC alert triage and prioritization system using synthetic customer, transaction and alert data. It uses a logistic regression model to assign a priority score to each alert based on features such as customer risk and transaction patterns. The workflow includes:

- Generating synthetic data sets for customers, transactions and alerts.
- Extracting and engineering features from the raw data.
- Training a logistic regression model to predict the probability that an alert is suspicious.
- Saving the priority scores to a CSV file and producing a simple distribution plot.

## Results

This repository showcases a complete analytics pipeline: data generation, feature engineering, model training and evaluation. It demonstrates how to automate alert triage and provides a basis for more advanced models. The project structure, tests and CI pipeline reflect my experience building compliant analytics solutions that helped drive an **18% churn reduction** and **~$120K/year savings** in past roles.

## Running the pipeline

Install dependencies:

```
pip install -r requirements.txt
```

Run the model to generate priority scores:

```
python src/main.py
```

This will train a model, save scores to `data/outputs/priority_scores.csv` and output a distribution plot (see `dashboards/screenshots/priority_score_distribution.png`).

## Project Structure

- `src/triage_model.py` – Functions for preparing features, training the model and predicting scores.
- `src/main.py` – Command-line entry point that orchestrates data loading, model training and scoring.
- `tests/` – Unit tests for the model functions.
- `data/raw/` – Synthetic input data (customers.csv, transactions.csv, alerts.csv).
- `.github/workflows/tests.yml` – GitHub Actions workflow that runs the tests on each commit.

All data is synthetic and free of sensitive information. See `LICENSE` for terms of use.
