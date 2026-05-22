from fastapi import FastAPI
from api.schema import CustomerData
import mlflow
import pandas as pd
import numpy as np
import os
from src.config import production_model_name
app = FastAPI()

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(mlflow_uri)
model = mlflow.sklearn.load_model(f'models:/{production_model_name}')


@app.post("/predict")
def predict_churn(data: CustomerData):
 
     df = pd.DataFrame([instance.model_dump() for instance in [data]])
     avg_monthly_charges = np.where(df['tenure'] != 0, df['TotalCharges'] / df['tenure'], 0)
     df['avg_monthly_charges'] = pd.to_numeric(avg_monthly_charges)
     prediction = model.predict(df)
     prediction = np.where(prediction == 1, 'Churn', 'No Churn')
     proba = model.predict_proba(df)
     return {"churn_prediction": prediction[0], "prediction_probability": proba[0].max()}

