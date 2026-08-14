import os
import uvicorn # type: ignore
from fastapi import FastAPI, Depends, Request # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import pandas as pd

from app.schemas import TransactionRequest, PredictionResponse
from app.ml_engine import FraudPredictionEngine
from fastapi.responses import RedirectResponse


app = FastAPI(title="Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_engine = FraudPredictionEngine(model_dir="models")

@app.get("/", include_in_schema=False)
def root():
    # Automatically redirect users to the Swagger documentation
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": ml_engine.model is not None}

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    df = pd.DataFrame([transaction.model_dump()])
    results = ml_engine.predict_batch(df)
    return results[0]

if __name__ == "__main__":
    # Read port assigned by Cloud Run (default to 8080 if local)
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)