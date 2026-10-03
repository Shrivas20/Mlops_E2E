# 🛡️ MLOps End-to-End: Network Security (Phishing Detection)

> A production-grade, end-to-end Machine Learning Operations (MLOps) project built to learn and demonstrate the complete lifecycle of an ML system — from raw data ingestion through model serving, with CI/CD automation and cloud deployment.

---

## 📌 Project Overview

This project implements a **Network Security / Phishing URL Detection** ML system using a fully automated MLOps pipeline. The goal is not just to build a model, but to practice the **engineering discipline** around ML — tracking experiments, automating pipelines, validating data, containerising the app, and deploying it to the cloud with continuous delivery.

| Property | Detail |
|---|---|
| **Domain** | Network Security — Phishing URL Detection |
| **Target Variable** | `Result` (binary: phishing or legitimate) |
| **Data Source** | MongoDB Atlas (`NetworkSecurity.PhishingData`) |
| **Model Serving** | FastAPI REST API |
| **Deployment** | AWS EC2 via Docker + GitHub Actions |

---

## 🏗️ Architecture & MLOps Pipeline

```
Raw CSV Data
     │
     ▼
[push_data.py] ──► MongoDB Atlas (Cloud Database)
                          │
                          ▼
              ┌───────────────────────┐
              │   Training Pipeline   │
              │                       │
              │  1. Data Ingestion    │  ◄── Pulls from MongoDB, splits train/test
              │  2. Data Validation   │  ◄── Schema check + KS-test drift detection
              │  3. Data Transform.   │  ◄── KNN Imputation via sklearn Pipeline
              │  4. Model Training    │  ◄── GridSearchCV across 6 classifiers
              └───────────────────────┘
                          │
                          ▼
                   MLflow Tracking
                 (Metrics + Models)
                          │
                          ▼
               final_model/model.pkl
                          │
                          ▼
              ┌─────────────────────┐
              │   FastAPI (app.py)  │
              │  POST /predict      │  ◄── Upload CSV → prediction output
              │  GET  /train        │  ◄── Re-trigger pipeline via API
              └─────────────────────┘
                          │
                          ▼
              Docker Container → AWS ECR → AWS EC2
```

---

## 🔧 Tech Stack

### Data & Storage
| Technology | Purpose |
|---|---|
| **MongoDB Atlas** | Cloud NoSQL database — stores raw phishing dataset |
| **PyMongo** | Python driver for MongoDB connectivity |
| **Certifi** | TLS/SSL certificate handling for secure MongoDB connection |
| **Pandas / NumPy** | Data manipulation and numerical computation |

### ML & Experimentation
| Technology | Purpose |
|---|---|
| **Scikit-learn** | ML algorithms, preprocessing pipelines, GridSearchCV |
| **MLflow** | Experiment tracking — logs metrics (F1, Precision, Recall) and model artifacts |
| **KNNImputer** | Handles missing values in the feature space |
| **GridSearchCV** | Hyperparameter tuning with 5-fold cross-validation |

#### Models Evaluated
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `AdaBoostClassifier`
- `LogisticRegression`
- `DecisionTreeClassifier`
- `KNeighborsClassifier`

### Serving
| Technology | Purpose |
|---|---|
| **FastAPI** | High-performance async REST API framework |
| **Uvicorn** | ASGI server to run FastAPI |
| **Jinja2 Templates** | HTML rendering for prediction output table |

### Infrastructure & DevOps
| Technology | Purpose |
|---|---|
| **Docker** | Containerisation — reproducible deployments |
| **GitHub Actions** | CI/CD pipeline — lint, build, push, deploy |
| **AWS ECR** | Elastic Container Registry — stores Docker images |
| **AWS EC2 (self-hosted runner)** | Deployment target — runs the latest Docker container |
| **uv / pyproject.toml** | Modern Python packaging and dependency management |

---

## 📁 Project Structure

```
Mlops_E2E/
│
├── src/
│   └── mlops_e2e/               # Main Python package
│       ├── components/          # Core ML pipeline steps
│       │   ├── data_ingestion.py      # Fetch from MongoDB, train/test split
│       │   ├── data_validation.py     # Schema check + drift detection (KS-test)
│       │   ├── data_transformation.py # KNN Imputation, save preprocessor
│       │   └── model_trainer.py       # Train, evaluate, MLflow tracking, save model
│       │
│       ├── pipeline/
│       │   ├── training_pipeline.py   # Orchestrates all 4 pipeline stages
│       │   └── batch_prediction.py    # Batch inference pipeline
│       │
│       ├── entity/
│       │   ├── config_entity.py       # Dataclasses for pipeline configuration
│       │   └── artifact_entity.py     # Dataclasses for pipeline artifacts (outputs)
│       │
│       ├── constants/
│       │   └── training_pipeline/     # All constants (dirs, filenames, hyperparams)
│       │
│       ├── utils/
│       │   ├── main_utils/utils.py    # YAML read, object save/load, array save/load
│       │   └── ml_utils/
│       │       ├── metric/            # Classification metric computation
│       │       └── model/estimator.py # NetworkSecurityModel wrapper (preprocessor + model)
│       │
│       ├── exception/execption.py     # Custom exception with traceback info
│       └── logging/logger.py          # Centralised logging setup
│
├── .github/
│   └── workflows/main.yml       # CI/CD: lint → build ECR image → deploy to EC2
│
├── Network_Data/
│   └── phisingData.csv          # Raw phishing dataset (used by push_data.py)
│
├── data_schema/
│   └── schema.yaml              # Column schema for data validation
│
├── templates/
│   └── table.html               # Jinja2 template for prediction output
│
├── Artifacts/                   # Timestamped pipeline run outputs (auto-generated)
├── final_model/                 # Latest model.pkl + preprocessor.pkl (for serving)
├── prediction_output/           # Batch prediction CSVs
├── logs/                        # Application logs
│
├── app.py                       # FastAPI application entry point
├── main.py                      # Script to run the full training pipeline manually
├── push_data.py                 # One-time script to push CSV data → MongoDB
├── dockerfile                   # Docker image definition
├── requirements.txt             # pip dependencies
├── pyproject.toml               # uv/PEP 517 project metadata & dependencies
└── .env                         # Environment variables (MONGO_DB_URL) — not committed
```

---

## 🔄 MLOps Pipeline — Step by Step

### 1. 📥 Data Ingestion
- Connects to **MongoDB Atlas** using the `MONGO_DB_URL` from environment variables
- Fetches the `PhishingData` collection from the `NetworkSecurity` database
- Performs an **80/20 train-test split**
- Saves raw data to the `Artifacts/<timestamp>/data_ingestion/` directory

### 2. ✅ Data Validation
- Validates that the incoming data matches the expected **schema** (column count from `data_schema/schema.yaml`)
- Performs **data drift detection** using the **Kolmogorov-Smirnov (KS) test** between train and test distributions
- Saves a drift report (`report.yaml`) for audit trails
- Routes data to `valid/` or `invalid/` directories

### 3. 🔄 Data Transformation
- Applies **KNN Imputation** (k=3, uniform weights) to handle missing values
- Wraps the imputer in a **scikit-learn Pipeline** for consistent train/test preprocessing
- Saves transformed data as `.npz` numpy arrays
- Persists the fitted preprocessor as `transformer.pkl` and `final_model/preprocessor.pkl`

### 4. 🤖 Model Training
- Trains **6 classifiers** simultaneously
- Uses **GridSearchCV** (5-fold CV) for hyperparameter tuning
- Selects the best model by **R² score** on the test set
- Logs metrics (F1, Precision, Recall) and model to **MLflow** for both train and test sets
- Saves the best model wrapped in a `NetworkSecurityModel` (preprocessor + model bundle)

---

## 🚀 CI/CD Pipeline (GitHub Actions)

The `.github/workflows/main.yml` automates **3 stages** on every push to `main`:

```
Push to main branch
       │
       ▼
┌─────────────────┐
│ 1. CI: Lint &   │
│    Unit Tests   │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ 2. CD: Build Docker │
│   image & push      │
│   to AWS ECR        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 3. Deploy: Pull &   │
│   run latest image  │
│   on AWS EC2        │
│   (self-hosted)     │
└─────────────────────┘
```

**Secrets required in GitHub repository settings:**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `ECR_REPOSITORY_NAME`
- `AWS_ECR_LOGIN_URI`

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- MongoDB Atlas account (with `MONGO_DB_URL`)
- Docker (for containerised run)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Shrivas20/Mlops_E2E.git
cd Mlops_E2E

# Using uv (recommended)
pip install uv
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Create a .env file and add your MongoDB connection string:
echo "MONGO_DB_URL=mongodb+srv://<user>:<password>@cluster.mongodb.net/" > .env
```

### 3. Push Data to MongoDB (one-time setup)

```bash
python push_data.py
```

### 4. Run Training Pipeline

```bash
# Via script
python main.py

# Via API endpoint (start server first, then call the endpoint)
# GET http://localhost:8080/train
```

### 5. Start the API Server

```bash
python app.py
# API available at: http://localhost:8080
# Swagger docs at:  http://localhost:8080/docs
```

### 6. Run with Docker

```bash
docker build -t mlops-e2e -f dockerfile .
docker run -p 8080:8080 --env-file .env mlops-e2e
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Redirects to Swagger UI (`/docs`) |
| `GET` | `/train` | Triggers the full training pipeline |
| `POST` | `/predict` | Upload a CSV file → returns predictions as HTML table |

### Example: Predict
```bash
curl -X POST "http://localhost:8080/predict" \
  -F "file=@your_data.csv"
```

---

## 🧪 MLflow Experiment Tracking

MLflow is used to log every training run:

```bash
# View the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open: http://localhost:5000
```

Each run logs:
- `f1_score`
- `precision_score`
- `recall_score`
- Trained model artifact

---

## 📐 Design Patterns Used

| Pattern | Where Used |
|---|---|
| **Config & Artifact Entities** | `entity/` — dataclasses decouple config from logic |
| **Custom Exception Handler** | `exception/execption.py` — captures file, line, and message |
| **Centralised Logger** | `logging/logger.py` — consistent logging across all modules |
| **Pipeline Orchestrator** | `pipeline/training_pipeline.py` — single entry point for all stages |
| **Model Wrapper Pattern** | `NetworkSecurityModel` bundles preprocessor + model for clean inference |
| **Constants Module** | `constants/training_pipeline/` — all magic values in one place |

---

## 🌱 What I Learned

This project was built to learn and apply the following MLOps concepts:

- ✅ **Modular ML code** — separating ingestion, validation, transformation, and training into independent components
- ✅ **Data versioning** — timestamped artifact directories per pipeline run
- ✅ **Drift detection** — using statistical tests to catch distribution shifts
- ✅ **Experiment tracking** — MLflow for reproducibility and comparison
- ✅ **Model packaging** — wrapping model + preprocessor for inference consistency
- ✅ **REST API serving** — FastAPI for synchronous inference and training triggers
- ✅ **Containerisation** — Docker for environment reproducibility
- ✅ **CI/CD** — GitHub Actions for automated testing, building, and deployment
- ✅ **Cloud deployment** — AWS ECR + EC2 for production-like hosting
- ✅ **Python packaging** — `pyproject.toml` with `uv` for modern dependency management

---

## 👤 Author

**Shrivas20**  
GitHub: [@Shrivas20](https://github.com/Shrivas20)

---

> ⭐ If you found this project helpful for learning MLOps, feel free to star the repository!