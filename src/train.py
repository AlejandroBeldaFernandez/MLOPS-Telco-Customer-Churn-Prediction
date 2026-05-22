import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from src.config import seed
import logging
import numpy as np
from sklearn.pipeline import Pipeline
import optuna
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from typing import Any
import mlflow
from src.config import model_name
logging.basicConfig(level=logging.INFO)


def train(X_train:pd.DataFrame, y_train:pd.Series, preprocessor_rf:ColumnTransformer)->tuple[Pipeline,dict[str, Any]]:
    """Optimise a Random Forest pipeline with Optuna and return the fitted model.

    Runs 150 Optuna trials using 5-fold cross-validation with balanced_accuracy as objective,
    then fits the best pipeline on the full training set.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels (0=No churn, 1=Churn).
        preprocessor_rf (ColumnTransformer): Unfitted preprocessor from preprocessing step.

    Returns:
        tuple: Fitted Pipeline and best hyperparameters dict found by Optuna.
    """
    mlflow.autolog(log_models=False)  # Disable automatic model logging to avoid conflicts with manual logging
    optuna.logging.set_verbosity(optuna.logging.INFO)
    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor_rf),
            ('model', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth = max_depth,
                min_samples_leaf = min_samples_leaf,
                min_samples_split = min_samples_split,
                class_weight=class_weight,
                max_features = max_features,
                random_state=seed

            ))
        ])
        
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='balanced_accuracy')
        return scores.mean()

    # Maximise balanced_accuracy to handle class imbalance (churn ~26%)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_rf, n_trials=150)

    logging.info(study.best_params)
    best_params = study.best_params

    pipeline_final = Pipeline([
        ('preprocessor', preprocessor_rf),
        ('model', RandomForestClassifier(**best_params, random_state=seed))
    ])

    pipeline_final.fit(X_train, y_train)
    mlflow.sklearn.log_model(
            sk_model=pipeline_final,
            artifact_path=model_name,
            input_example=X_train.head(1)
        )
    joblib.dump(pipeline_final, "models/rf_model.pkl")

    return pipeline_final, best_params

    
    