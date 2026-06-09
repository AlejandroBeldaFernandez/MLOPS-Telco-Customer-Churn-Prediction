import logging

import pandas as pd

REQUIRED_COLUMNS = [
    "customerID",
    "Churn",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "SeniorCitizen",
]


def validate_data(df: pd.DataFrame) -> bool:
    passed = True

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        passed = False

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logging.warning(f"Found {null_count} missing values")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logging.warning(f"Found {dup_count} duplicate rows")

    if len(df) < 100:
        logging.error(f"Dataset too small: {len(df)} rows")
        passed = False

    churn_rate = (df["Churn"] == "Yes").mean()
    logging.info(f"Churn rate: {churn_rate:.2%}")
    if churn_rate < 0.05 or churn_rate > 0.95:
        logging.error(f"Severe class imbalance: churn rate = {churn_rate:.2%}")
        passed = False

    logging.info(f"Data validation {'passed' if passed else 'failed'}")
    return passed
