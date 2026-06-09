import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import seed

logging.basicConfig(level=logging.INFO)


def preprocessing(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Clean, engineer features, split and build the preprocessor for the dataset.

    Handles TotalCharges type conversion, drops customerID, engineers avg_monthly_charges,
    splits into train/test and builds a ColumnTransformer with scaling, one-hot and ordinal encoding.

    Args:
        df (pd.DataFrame): Raw dataset returned by data_loader.

    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor_rf (unfitted ColumnTransformer).
    """
    # 11 rows have whitespace in TotalCharges — drop them before converting to numeric
    df = df[~df["TotalCharges"].str.contains("[^0-9.]")].copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    df.drop(columns=["customerID"], inplace=True)
    # Avoid division by zero for customers with tenure=0
    avg_monthly_charges = np.where(df["tenure"] != 0, df["TotalCharges"] / df["tenure"], 0)
    df["avg_monthly_charges"] = pd.to_numeric(avg_monthly_charges)
    logging.info(df.columns.to_list())

    X = df.drop(columns=["Churn"]).copy()
    y = df["Churn"].copy()
    y = y.map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler_columns = df.select_dtypes(include=np.number).columns
    # Binary columns use OrdinalEncoder (2 categories); multi-category columns use OneHotEncoder
    binary_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    categorical_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    preprocessor_rf = ColumnTransformer(
        [
            ("scaler", StandardScaler(), scaler_columns),
            ("encoder", OneHotEncoder(), categorical_columns),
            ("binarizer", OrdinalEncoder(), binary_columns),
        ],
        remainder="passthrough",
    )

    return X_train, X_test, y_train, y_test, preprocessor_rf
