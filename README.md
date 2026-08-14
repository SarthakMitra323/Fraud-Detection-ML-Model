# Fraud Detection API 🛡️

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange)
![License: MIT](https://img.shields.io/badge/License-Apache_2.0-green.svg)

A production-ready, real-time Machine Learning API built with **FastAPI**, **XGBoost**, and **Scikit-Learn**. This API evaluates financial transactions and returns a fraud probability score along with explainable AI (XAI) risk signals.

## ✨ Features

* **Real-Time Inference:** Fast, optimized predictions using an XGBoost classifier.
* **Explainable AI (XAI):** Returns specific risk signals (e.g., "Transaction velocity is unusually high") to explain *why* a transaction was flagged.
* **Strict Validation:** Uses Pydantic to ensure all incoming JSON requests are perfectly formatted and mathematically valid before reaching the model.
* **Cross-Platform:** Model weights are saved natively as `.json` to prevent OS-level memory corruption across Linux/Windows environments.
* **Cloud-Native:** Fully containerized with Docker, ready to deploy to Google Cloud Run, AWS AppRunner, or anywhere Docker is supported.

## 📂 Project Structure

```text
fraud-api/
├── app/
│   ├── __init__.py        # Empty init file
│   ├── main.py            # FastAPI application and routing
│   ├── ml_engine.py       # ML loading, preprocessing, and inference logic
│   └── schemas.py         # Pydantic validation models
├── models/
│   ├── model.json         # XGBoost model weights
│   ├── preprocessor.joblib# Scikit-Learn ColumnTransformer
│   └── metadata.json      # Model thresholds and metadata
├── tests/
│   ├── legitimate_transaction.json
│   ├── account_takeover_fraud.json
│   ├── card_testing_fraud.json
│   └── wealthy_traveler_legitimate.json
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Getting Started (Local Development)

### 1. Clone the repository
```bash
git clone https://github.com/SarthakMitra323/Fraud-Detection-ML-Model.git
cd fraud-api
```

### 2. Create a Virtual Environment
```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
uvicorn app.main:app --reload --port 8080
```
Visit `http://localhost:8080/docs` in your browser to interact with the automatically generated Swagger UI.

## 🐳 Docker Deployment

To bypass local environment setups and run the API in a containerized environment:

```bash
# Build the image
docker build -t fraud-api .

# Run the container
docker run -p 8080:8080 fraud-api
```

## 📡 API Usage & Testing

**Endpoint:** `POST /api/v1/predict`

### Testing with Sample Payloads

You can use the included JSON fixtures in the `tests/` directory to test the API via `curl`:

#### 1. Test an Account Takeover Attack
```bash
curl -X 'POST' \
  'http://localhost:8080/api/v1/predict' \
  -H 'Content-Type: application/json' \
  -d @tests/account_takeover_fraud.json
```

#### 2. Test a Standard Legitimate Payment
```bash
curl -X 'POST' \
  'http://localhost:8080/api/v1/predict' \
  -H 'Content-Type: application/json' \
  -d @tests/legitimate_transaction.json
```

#### 3. Test High-Velocity Card Testing
```bash
curl -X 'POST' \
  'http://localhost:8080/api/v1/predict' \
  -H 'Content-Type: application/json' \
  -d @tests/card_testing_fraud.json
```

### Example Response
```json
{
  "prediction": "fraud",
  "fraud_probability": 0.9854,
  "risk_score": 98,
  "risk_level": "HIGH",
  "risk_signals": [
    {
      "feature": "balance_change",
      "impact": "high",
      "reason": "The change in account balance is severe."
    },
    {
      "feature": "is_new_device",
      "impact": "high",
      "reason": "Transaction originated from a newly observed device."
    }
  ],
  "model": {
    "name": "fraud_detection_model",
    "version": "1.0.1"
  },
  "timestamp": "2026-08-14T12:00:00Z"
}
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the `LICENSE` file for details.
