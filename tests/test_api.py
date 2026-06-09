from unittest.mock import MagicMock, patch

import numpy as np

mock_model = MagicMock()
mock_model.predict.return_value = np.array([0])
mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])


with patch("mlflow.sklearn.load_model", return_value=mock_model):
    from api.api import app  # el mock está activo cuando se importa

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


def test_predict_response():
    response = client.post(
        "/predict",
        json={
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 50.0,
            "TotalCharges": 600.0,
        },
    )
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["churn_prediction"] == "No Churn" or response_json["churn_prediction"] == "Churn"
    assert response_json["prediction_probability"] > 0.0
