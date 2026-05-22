from src.data_loader import data_loader
from src.preprocessing import preprocessing
from src.train import train
from src.evaluation import evaluation
from src.config import experiment_name, model_name
from src.model_registry import ModelRegistry
import mlflow
from prefect import task, flow

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


@flow
def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        register = ModelRegistry(model_name=model_name, run_id=mlflow.active_run().info.run_id)
        df = data_loader_task()
        X_train, X_test, y_train, y_test, preprocessor_rf = preprocessing_task(df)
        # preprocessor_rf is passed unfitted — the Pipeline fits it internally during train
        pipeline_final, best_params = train_task(X_train, y_train, preprocessor_rf)
        metrics = evaluation_task(X_train, X_test, y_train, y_test, pipeline_final)
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        register.register_model()
        
if __name__ == "__main__":
    main()