import mlflow

class ModelRegistry:
    def __init__(self, model_name:str, run_id:str):
        self.model_name = model_name
        self.run_id = run_id
       
        
    
    def register_model(self):
        mlflow.register_model(model_uri=f"runs:/{self.run_id}/{self.model_name}", name=self.model_name)
    
    def get_model(self):
        return mlflow.sklearn.load_model(f"runs:/{self.run_id}/{self.model_name}")