import mlflow
import logging
class ModelRegistry:
    def __init__(self, model_name:str, run_id:str):
        self.model_name = model_name
        self.run_id = run_id
       
        
    
    def register_model(self):
        result = mlflow.register_model(model_uri=f"runs:/{self.run_id}/{self.model_name}", name=self.model_name)
        return result.version
    
    def get_model(self):
        return mlflow.sklearn.load_model(f"runs:/{self.run_id}/{self.model_name}")
    
    def promote_if_better(self, version):
        client = mlflow.tracking.MlflowClient()
        registered_model = client.get_registered_model(self.model_name)
        if "champion" not in registered_model.aliases:
            logging.info("No champion found. Promoting current model.")
            client.set_registered_model_alias(self.model_name, "champion", version)
            return
        
        latest_version = client.get_model_version_by_alias(self.model_name, "champion")
        latest_metrics = client.get_run(latest_version.run_id).data.metrics
        current_metrics = client.get_run(self.run_id).data.metrics
        
        if current_metrics["ROC_AUC_test"] > latest_metrics["ROC_AUC_test"]:
            logging.info("Current model outperforms champion. Promoting.")
            client.set_registered_model_alias(self.model_name, "champion", version)
        else:
            logging.info("Current model does not outperform champion. Keeping existing.")
    
   