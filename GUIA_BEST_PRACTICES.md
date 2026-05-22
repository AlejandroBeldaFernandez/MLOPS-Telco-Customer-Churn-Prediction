# Guía de Best Practices MLOps — Módulo 6

Esta guía explica cómo aplicar las best practices del módulo 6 a cualquier proyecto propio de MLOps.
Cubre: tests unitarios, tests de integración, linter, pre-commit hooks, Makefile, Terraform+LocalStack y CI/CD con GitHub Actions.

---

## Estructura de carpetas recomendada

```
mi_proyecto/
├── batch.py / model.py          # código principal
├── Dockerfile
├── Makefile
├── Pipfile
├── .pre-commit-config.yaml
├── tests/
│   ├── __init__.py              # necesario para que pytest pueda importar tu código
│   └── test_batch.py
├── integration-tests/
│   ├── docker-compose.yaml
│   ├── run.sh
│   ├── test_docker.py           # (si usas contenedor HTTP)
│   └── test_kinesis.py          # (si usas Kinesis/streams)
└── infrastructure/              # (opcional, si usas Terraform)
    ├── main.tf
    ├── variables.tf
    └── modules/
        └── s3/
```

---

## 1. Unit Tests con pytest

### Concepto
Los unit tests prueban **funciones individuales en aislamiento**, sin tocar S3, bases de datos ni modelos reales.
El truco es separar la lógica de transformación (testeable) de la I/O (no testeable directamente).

### Paso 1 — Separar lógica de I/O

```python
# batch.py
def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    return prepare_data(df, categorical)          # <-- separado

def prepare_data(df, categorical):               # <-- esto es lo que testeamos
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df
```

### Paso 2 — Crear `tests/__init__.py`

Archivo vacío. Sin él, Python no puede importar tu código desde `tests/`.

```bash
touch tests/__init__.py
```

### Paso 3 — Escribir el test

```python
# tests/test_batch.py
import pandas as pd
from batch import prepare_data

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),   # 9 min → pasa
        (1, 1,    dt(1, 2), dt(1, 10)),       # 8 min → pasa
        (1, None, dt(1, 2, 0), dt(1, 2, 59)), # 0.98 min → filtrado
        (3, 4,    dt(1, 2, 0), dt(2, 2, 1)),  # 60.01 min → filtrado
    ]
    columns = ['PULocationID', 'DOLocationID',
               'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    result = prepare_data(df, ['PULocationID', 'DOLocationID'])

    expected = [
        {'PULocationID': '-1', 'DOLocationID': '-1', 'duration': 9.0},
        {'PULocationID': '1',  'DOLocationID': '1',  'duration': 8.0},
    ]
    assert result[['PULocationID', 'DOLocationID', 'duration']].to_dict(orient='records') == expected
```

> **Tip:** Compara DataFrames convirtiéndolos a lista de dicts con `.to_dict(orient='records')`.

### Paso 4 — Para modelos: usar un ModelMock

Si tu código llama a un modelo ML, no uses el modelo real en unit tests. Crea un mock:

```python
# tests/model_test.py
class ModelMock:
    def predict(self, X):
        return [10.0]   # valor fijo

def test_predict():
    model = ModelMock()
    service = ModelService(model=model)
    features = {'PU_DO': '130_205', 'trip_distance': 3.66}
    result = service.predict(features)
    assert result == 10.0
```

### Ejecutar tests

```bash
pytest tests/ -v
```

---

## 2. Integration Tests con Docker + LocalStack

### Concepto
Los integration tests prueban el sistema completo: el contenedor Docker con el modelo real,
más servicios externos simulados (S3, Kinesis) con LocalStack.

### Paso 1 — `docker-compose.yaml` en `integration-tests/`

```yaml
services:
  backend:
    image: ${LOCAL_IMAGE_NAME}
    ports:
      - "8080:8080"
    environment:
      - RUN_ID=test-run-001
      - MODEL_LOCATION=/app/model
      - KINESIS_ENDPOINT_URL=http://kinesis:4566
      - PREDICTIONS_STREAM_NAME=ride_predictions
      - AWS_DEFAULT_REGION=eu-west-1
      - AWS_ACCESS_KEY_ID=abc
      - AWS_SECRET_ACCESS_KEY=xyz
    volumes:
      - "./model:/app/model"   # monta el modelo local dentro del contenedor

  kinesis:
    image: localstack/localstack:0.13.0
    ports:
      - "4566:4566"
    environment:
      - SERVICES=kinesis       # solo activa el servicio que necesitas
```

> Para S3 en lugar de Kinesis: `SERVICES=s3`
> Para ambos: `SERVICES=s3,kinesis`

### Paso 2 — `run.sh`

```bash
#!/usr/bin/env bash

if [[ -z "${GITHUB_ACTIONS}" ]]; then
  cd "$(dirname "$0")"   # en local, muévete al directorio del script
fi

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=$(date +"%Y-%m-%d-%H-%M")
    export LOCAL_IMAGE_NAME="mi-modelo:${LOCAL_TAG}"
    docker build -t ${LOCAL_IMAGE_NAME} ..
fi

docker compose up -d
sleep 10

# Crear recursos en LocalStack (Kinesis stream, bucket S3, etc.)
export AWS_ACCESS_KEY_ID=abc
export AWS_SECRET_ACCESS_KEY=xyz
export AWS_DEFAULT_REGION=eu-west-1

aws --endpoint-url=http://localhost:4566 kinesis create-stream \
    --stream-name ride_predictions --shard-count 1

sleep 3

python test_docker.py
ERROR_CODE=$?
if [ ${ERROR_CODE} != 0 ]; then
    docker compose logs
    docker compose down
    exit ${ERROR_CODE}
fi

python test_kinesis.py
ERROR_CODE=$?
if [ ${ERROR_CODE} != 0 ]; then
    docker compose logs
    docker compose down
    exit ${ERROR_CODE}
fi

docker compose down
```

### Paso 3 — `test_docker.py` (probar el endpoint HTTP)

```python
import json
import requests
from deepdiff import DeepDiff

event = {
    "Records": [{
        "kinesis": {
            "data": "<base64_encoded_event>"   # ver sección 2.4
        }
    }]
}

response = requests.post(
    "http://localhost:8080/2015-03-31/functions/function/invocations",
    json=event
)
result = response.json()

expected = {
    "predictions": [{
        "model": "ride_duration_prediction_model",
        "version": "test-run-001",
        "prediction": {
            "ride_duration": 18.0,    # ajusta al valor real
            "ride_id": 256,
        }
    }]
}

diff = DeepDiff(result, expected, significant_digits=1)
print(f'diff={diff}')
assert diff == {}
```

### Paso 4 — Cómo generar el evento base64

```python
import base64, json

ride_event = {
    "ride": {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 3.66,
    },
    "ride_id": 256,
}
encoded = base64.b64encode(json.dumps(ride_event).encode('utf-8')).decode('utf-8')
print(encoded)
# Pega ese string como valor de "data" en tu event.json o test_docker.py
```

### Paso 5 — `test_kinesis.py` (verificar que se escribió en el stream)

```python
import boto3, json

def get_records(kinesis_client, stream_name):
    response = kinesis_client.get_shard_iterator(
        StreamName=stream_name,
        ShardId='shardId-000000000000',
        ShardIteratorType='TRIM_HORIZON',
    )
    shard_iterator = response['ShardIterator']
    records_response = kinesis_client.get_records(ShardIterator=shard_iterator)
    # IMPORTANTE: boto3 ya devuelve bytes, no base64 — no hay que decodificar en base64
    return [json.loads(r['Data'].decode('utf-8')) for r in records_response['Records']]

def test_kinesis():
    kinesis_client = boto3.client(
        'kinesis',
        endpoint_url='http://localhost:4566',
        region_name='eu-west-1',
        aws_access_key_id='abc',
        aws_secret_access_key='xyz',
    )
    records = get_records(kinesis_client, 'ride_predictions')
    assert len(records) == 1
    assert records[0]['prediction']['ride_id'] == 256
    print('test_kinesis PASSED')

if __name__ == '__main__':
    test_kinesis()
```

### Paso 6 — Preparar el modelo para los tests

Como el modelo está en `.gitignore`, hay que crearlo antes de correr los integration tests.
Crea un script `integration-tests/create_test_model.py`:

```python
import os, shutil
import mlflow, numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'model')

X_train = [
    {'PU_DO': '130_205', 'trip_distance': 3.66},
    {'PU_DO': '1_2',     'trip_distance': 1.0},
]
y_train = np.array([15.0, 7.0])

pipeline = make_pipeline(DictVectorizer(), LinearRegression())
pipeline.fit(X_train, y_train)

mlflow.set_tracking_uri('file:///tmp/mlruns_test')
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(pipeline, artifact_path='model')
    run_id = run.info.run_id

src = mlflow.artifacts.download_artifacts(f'runs:/{run_id}/model', dst_path='/tmp/mlflow_model')
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
shutil.copytree(src, OUTPUT_DIR)
print(f'Model saved to {OUTPUT_DIR}')
```

### Ejecutar integration tests

```bash
# Construir imagen y correr tests
LOCAL_IMAGE_NAME=mi-modelo:test bash integration-tests/run.sh
```

---

## 3. Linter y Formatter

### Instalar

```bash
pipenv install --dev isort black pylint
```

### Usar

```bash
isort .                        # ordena imports
black .                        # formatea código
pylint --recursive=y .         # analiza calidad
```

### Pylint — deshabilitar warnings molestos

```bash
pylint --recursive=y \
  --disable=missing-function-docstring,invalid-name,too-few-public-methods,\
missing-module-docstring,missing-class-docstring,unused-argument,\
missing-timeout,unused-import \
  .
```

---

## 4. Makefile

Automatiza los comandos más comunes:

```makefile
LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=mi-modelo:${LOCAL_TAG}

test:
	pytest tests/ -v

quality_checks:
	isort .
	black .
	pylint --recursive=y \
	  --disable=missing-function-docstring,invalid-name,too-few-public-methods,\
missing-module-docstring,missing-class-docstring,unused-argument,\
missing-timeout,unused-import \
	  .

build:
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration-tests/run.sh

setup:
	pipenv install --dev
```

### Usar

```bash
make test
make quality_checks
make build
make integration_test
```

---

## 5. Pre-commit Hooks

Los pre-commit hooks ejecutan automáticamente isort, black y pylint antes de cada `git commit`.
Si alguno falla, el commit se cancela hasta que arregles el código.

### Instalar pre-commit

```bash
pipenv install --dev pre-commit
pre-commit install   # activa los hooks en el repo
```

### `.pre-commit-config.yaml`

```yaml
exclude: mi_proyecto/homeworks/   # carpetas a ignorar (opcional)

repos:
- repo: https://github.com/pycqa/isort
  rev: 5.13.2
  hooks:
    - id: isort

- repo: https://github.com/psf/black
  rev: 22.6.0
  hooks:
    - id: black
      language_version: python3.9

- repo: local
  hooks:
    - id: pylint
      name: pylint
      entry: pylint
      language: system
      types: [python]
      args:
        - "--recursive=y"
        - "--disable=missing-function-docstring,invalid-name,too-few-public-methods,\
missing-module-docstring,missing-class-docstring,unused-argument,\
missing-timeout,unused-import"
```

### Usar

```bash
# Se ejecuta solo al hacer git commit
git commit -m "mi mensaje"

# Ejecutar manualmente sobre todos los archivos
pre-commit run --all-files
```

---

## 6. Terraform + LocalStack (Infraestructura como Código)

Terraform define la infraestructura (buckets S3, streams, etc.) en archivos `.tf`.
LocalStack simula AWS localmente.

### Estructura

```
infrastructure/
├── main.tf
├── variables.tf
├── docker-compose.yaml       # LocalStack + MLflow
└── modules/
    └── s3/
        ├── main.tf
        └── variables.tf
```

### `infrastructure/main.tf`

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"   # versión 4.x funciona bien con LocalStack
    }
  }
}

provider "aws" {
  region                      = "eu-west-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  s3_use_path_style           = true   # necesario para LocalStack

  endpoints {
    s3 = "http://localhost:4566"
  }
}

module "s3_bucket" {
  source      = "./modules/s3"
  bucket_name = var.bucket_name
}

output "mlflow_bucket" {
  value = module.s3_bucket.name
}
```

### `infrastructure/variables.tf`

```hcl
variable "bucket_name" {
  description = "Nombre del bucket S3"
  default     = "mlflow-artifacts"
}
```

### `infrastructure/modules/s3/main.tf`

```hcl
resource "aws_s3_bucket" "s3_bucket" {
  bucket        = var.bucket_name
  force_destroy = true
}

output "name" {
  value = aws_s3_bucket.s3_bucket.bucket
}
```

### `infrastructure/modules/s3/variables.tf`

```hcl
variable "bucket_name" {
  description = "Nombre del bucket"
}
```

### `infrastructure/docker-compose.yaml`

```yaml
services:
  localstack:
    image: localstack/localstack:0.13.0
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - AWS_ACCESS_KEY_ID=test
      - AWS_SECRET_ACCESS_KEY=test
      - MLFLOW_S3_ENDPOINT_URL=http://localstack:4566
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root s3://mlflow-artifacts
    depends_on:
      - localstack
```

### Comandos Terraform

```bash
cd infrastructure
docker compose up -d          # arrancar LocalStack

terraform init                # descargar providers
terraform plan                # ver qué va a crear
terraform apply               # crear infraestructura
terraform destroy             # destruir todo
```

> **Importante:** nunca subas `.terraform/` a git (contiene binarios de ~300MB).
> Añádelo al `.gitignore`:
> ```
> **/.terraform/
> ```

---

## 7. CI/CD con GitHub Actions

El workflow se ejecuta automáticamente en cada push a `main` que toque los archivos del proyecto.

### `.github/workflows/ci.tests.yml`

```yaml
name: CI Tests

on:
  push:
    branches:
      - main
    paths:
      - 'mi_proyecto/**'   # solo se activa si cambias archivos del proyecto

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        working-directory: mi_proyecto
        run: pip install pytest mlflow scikit-learn boto3

      - name: Run unit tests
        working-directory: mi_proyecto
        run: pytest tests/ -v

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests       # solo corre si unit-tests pasa
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        working-directory: mi_proyecto
        run: pip install requests deepdiff boto3 numpy==1.23.5 scikit-learn==1.0.2 mlflow

      - name: Install Docker Compose
        run: |
          mkdir -p ~/.docker/cli-plugins
          curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
            -o ~/.docker/cli-plugins/docker-compose
          chmod +x ~/.docker/cli-plugins/docker-compose

      - name: Create test model
        working-directory: mi_proyecto
        run: python integration-tests/create_test_model.py

      - name: Build and run integration tests
        working-directory: mi_proyecto
        run: |
          docker build -t mi-modelo:test .
          cd integration-tests
          LOCAL_IMAGE_NAME=mi-modelo:test bash run.sh
```

### Token de GitHub — scopes necesarios

El Personal Access Token necesita estos permisos:
- `repo` — para leer/escribir código
- `workflow` — para crear/modificar archivos en `.github/workflows/`

Sin el scope `workflow`, el push de archivos de CI dará error.

---

## 8. `.gitignore` esencial

```gitignore
# datos
*.parquet
*.csv

# modelos grandes
*.bin
*.pkl
mlruns/
mlflow.db
artifacts/

# integration test model (se genera en runtime)
mi_proyecto/integration-tests/model/

# terraform providers (binarios de 300MB)
**/.terraform/

# python
**/__pycache__/
**/*.pyc
```

---

## 9. Limpiar el historial de git si subiste archivos grandes

Si accidentalmente commiteaste archivos grandes (modelos, binarios de terraform...):

```bash
# Instalar git-filter-repo
pip install git-filter-repo

# Eliminar un directorio de TODO el historial
git filter-repo --path ruta/al/directorio --invert-paths --force

# Restaurar el remote (git-filter-repo lo elimina como medida de seguridad)
git remote add origin https://tu-token@github.com/usuario/repo.git

# Forzar push (reescribe el historial remoto)
git push --force origin main
```

> Después de limpiar: `git count-objects -vH` debería mostrar pocos MB en `size-pack`.

---

## 10. Flujo de trabajo completo para un proyecto nuevo

```bash
# 1. Estructurar el proyecto
mkdir -p mi_proyecto/{tests,integration-tests}
touch mi_proyecto/tests/__init__.py

# 2. Instalar dependencias de desarrollo
pipenv install --dev pytest isort black pylint pre-commit

# 3. Activar pre-commit
pre-commit install

# 4. Separar lógica de I/O en el código
#    → crear prepare_data(), save_data(), get_input_path(), get_output_path()

# 5. Escribir unit tests
#    → tests/test_batch.py con prepare_data y ModelMock

# 6. Crear Makefile con targets: test, quality_checks, build, integration_test

# 7. Crear integration-tests/
#    → docker-compose.yaml con LocalStack
#    → create_test_model.py
#    → run.sh
#    → test_docker.py y/o test_kinesis.py

# 8. (Opcional) Terraform para infraestructura
#    → infrastructure/main.tf apuntando a LocalStack

# 9. CI/CD
#    → .github/workflows/ci.tests.yml
#    → Token de GitHub con scope 'workflow'

# 10. Verificar .gitignore antes de cada push
git status   # revisar que no haya archivos grandes sin seguimiento
```

---

## Resumen de respuestas del Homework 6

| Pregunta | Respuesta |
|----------|-----------|
| Q1 — Bloque main | `if __name__ == '__main__':` |
| Q2 — Otro fichero en tests/ | `__init__.py` |
| Q3 — Filas esperadas | 2 |
| Q4 — Opción para LocalStack | `--endpoint-url` |
| Q5 — Tamaño del fichero | ~3620 bytes |
| Q6 — Suma de predicciones | 36.28 |
