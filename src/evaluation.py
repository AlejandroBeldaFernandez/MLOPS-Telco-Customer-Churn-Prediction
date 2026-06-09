import json
import logging
import os

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)


def evaluation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pipeline_final: Pipeline,
) -> dict[str, float]:
    """Evaluate the fitted pipeline on train and test sets and return metrics as a dict.

    Computes balanced_accuracy, ROC-AUC, precision and recall for both splits,
    chosen for their relevance on the imbalanced churn dataset.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.
        pipeline_final (Pipeline): Fitted pipeline from the train step.

    Returns:
        dict[str, float]: Metrics keyed as {train|test}_{metric_name}.
    """
    y_pred_train = pipeline_final.predict(X_train)
    y_pred = pipeline_final.predict(X_test)

    logging.info("-----------------------------------------------------------------------")
    logging.info("Metrics TRAIN")
    roc_train = roc_auc_score(y_train, pipeline_final.predict_proba(X_train)[:, 1])
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)

    logging.info(f"Confusion Matrix: {confusion_matrix(y_train, y_pred_train, labels=[0, 1])}")
    logging.info(f"ROC: {roc_train}")
    logging.info(f"Classification Report: {classification_report(y_train, y_pred_train, labels=[0, 1])}")
    logging.info(f"Balanced Accuracy: {balanced_accuracy_train}")
    logging.info(f"Precision: {precision_train}")
    logging.info(f"Recall: {recall_train}")

    logging.info("-----------------------------------------------------------------------")
    logging.info("Metrics TEST")
    roc_test = roc_auc_score(y_test, pipeline_final.predict_proba(X_test)[:, 1])
    balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    recall_test = recall_score(y_test, y_pred)
    logging.info(f"Confusion Matrix: {confusion_matrix(y_test, y_pred, labels=[0, 1])}")
    logging.info(f"ROC: {roc_test}")
    logging.info(f"Classification Report: {classification_report(y_test, y_pred, labels=[0, 1])}")
    logging.info(f"Balanced Accuracy: {balanced_accuracy_test}")
    logging.info(f"Precision: {precision_test}")
    logging.info(f"Recall: {recall_test}")
    metrics = {
        "balanced_accuracy_train": balanced_accuracy_train,
        "roc_train": roc_train,
        "precision_train": precision_train,
        "recall_train": recall_train,
        "balanced_accuracy_test": balanced_accuracy_test,
        "roc_test": roc_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f)
    return metrics
