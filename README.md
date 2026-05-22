# MLOps Telco Customer Churn

End-to-end MLOps pipeline for predicting customer churn using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. Built as a final project for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) by DataTalks.Club.

The focus is on **MLOps infrastructure**, not model performance.

---

## Model

- **Algorithm:** Random Forest with Optuna hyperparameter optimization (150 trials)
- **Class imbalance:** handled via `class_weight` parameter
- **Preprocessing:** StandardScaler, OneHotEncoder, OrdinalEncoder via scikit-learn Pipeline
- **Metrics (test set):** Balanced Accuracy ~0.76, Recall (churn) ~0.76, ROC-AUC ~0.84

---

## Architecture

```
pipeline.py (Prefect)
    ├── data_loader    → loads CSV
    ├── preprocessing  → cleans, engineers features, splits train/test
    ├── train          → Optuna optimization, fits pipeline, logs model to MLflow
    └── evaluation     → computes metrics, logs to MLflow

MLflow          → experiment tracking + model registry (artifact store: S3/LocalStack)
Prefect         → pipeline orchestration
FastAPI         → REST API serving predictions from MLflow @champion model
Docker Compose  → runs all services
```

---

## Project Structure

```
├── api/
│   ├── api.py          # FastAPI app with /predict endpoint
│   └── schema.py       # Pydantic input schema
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   ├── Dockerfile.mlflow
│   └── Dockerfile.prefect
├── src/
│   ├── config.py       # global constants
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluation.py
│   └── model_registry.py
├── pipeline.py         # Prefect flow orchestrating all steps
└── requirements.txt
```

---

## How to Run

### 1. Start services

```bash
docker compose -f docker/docker-compose.yml up --build
```

This starts:
- **MLflow** at `http://localhost:5000`
- **Prefect** at `http://localhost:4200`
- **FastAPI** at `http://localhost:8000`

### 2. Train and register the model

```bash
python pipeline.py
```

### 3. Set the @champion alias

Go to `http://localhost:5000`, find the registered model `final_model_pipeline_rf` and assign the `@champion` alias to the desired version.

### 4. Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "tenure": 12,
    "MonthlyCharges": 65.0,
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "TotalCharges": 780.0
  }'
```

---

## Roadmap

- [x] EDA and feature engineering
- [x] Modular pipeline (data_loader, preprocessing, train, evaluation)
- [x] MLflow experiment tracking and model registry
- [x] Prefect orchestration
- [x] FastAPI prediction endpoint
- [x] Docker Compose (MLflow + Prefect + FastAPI)
- [ ] LocalStack + Terraform (S3 artifact store)
- [ ] Evidently + Grafana + PostgreSQL (model monitoring)
- [ ] pytest + pre-commit + GitHub Actions (CI/CD)
- [ ] Gradio/Streamlit on Hugging Face Spaces (demo)
