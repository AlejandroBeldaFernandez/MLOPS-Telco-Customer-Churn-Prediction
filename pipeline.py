import logging

import mlflow
from dotenv import load_dotenv
from prefect import flow, task

from src.config import experiment_name, model_name
from src.data_loader import data_loader
from src.evaluation import evaluation
from src.model_registry import ModelRegistry
from src.monitor import monitoring, send_alert_email
from src.preprocessing import preprocessing
from src.train import train
from src.validation import validate_data

load_dotenv()


@task
def data_loader_task():
    return data_loader()


@task
def preprocessing_task(df):
    return preprocessing(df)


@task
def train_task(X_train, y_train, preprocessor_rf):
    return train(X_train, y_train, preprocessor_rf)


@task
def evaluation_task(X_train, X_test, y_train, y_test, pipeline_final):
    return evaluation(X_train, X_test, y_train, y_test, pipeline_final)


@task
def monitoring_task(reference_df, current_df):
    return monitoring(reference_df, current_df)


@task
def validate_data_task(df):
    return validate_data(df)


def champion_exists() -> bool:
    client = mlflow.tracking.MlflowClient()
    try:
        registered_model = client.get_registered_model(model_name)
        return "champion" in registered_model.aliases
    except Exception:
        return False


@flow
def main():
    """
    Main MLOps pipeline: drift detection → conditional retraining → model registration.

    Training is triggered when:
    - No @champion exists yet (first run), or
    - Drift is detected in production data.

    Drift detection uses two splits of the full dataset to simulate reference vs.
    production data. In a real deployment, current_df would come from logged
    API predictions accumulated since the last training run.

    Training always uses the full dataset regardless of which rows were used for
    drift analysis, so the registered model benefits from all available data.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    df = data_loader_task()
    is_valid = validate_data_task(df)

    if not is_valid:
        logging.error("Data validation failed. Aborting pipeline.")
        return

    no_champion = not champion_exists()

    # Simulate reference (historical) vs current (production) data for drift analysis.
    # In production: reference_df = training data snapshot; current_df = recent predictions.
    reference_df = df.sample(frac=0.5, random_state=42)
    current_df = df.drop(reference_df.index)
    drift_detected = monitoring_task(reference_df, current_df)

    should_train = no_champion or drift_detected

    if should_train:
        if no_champion:
            logging.info("No champion found. Training initial model.")
        else:
            logging.info("Drift detected. Retraining the model.")
        with mlflow.start_run():
            register = ModelRegistry(model_name=model_name, run_id=mlflow.active_run().info.run_id)
            X_train, X_test, y_train, y_test, preprocessor_rf = preprocessing_task(df)
            # preprocessor_rf is passed unfitted — the Pipeline fits it internally during train
            pipeline_final, best_params = train_task(X_train, y_train, preprocessor_rf)
            metrics = evaluation_task(X_train, X_test, y_train, y_test, pipeline_final)
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            version = register.register_model()
            register.promote_if_better(version)
            send_alert_email(metrics)
    else:
        logging.info("No drift detected and champion exists. Skipping retraining.")


if __name__ == "__main__":
    # To deploy with monthly schedule on a dedicated Prefect server:
    # from prefect.schedules import Cron
    # main.serve(schedules=[Cron("0 9 1 * *")])
    main()
