import os
import time
from typing import Dict
import uvicorn # type: ignore
from fastapi import FastAPI, Depends, Request, HTTPException # type: ignore
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

# ==============================================================================
# TOKEN BUCKET RATE LIMITER
# ==============================================================================
class TokenBucketLimiter:
    def __init__(self, capacity: int, refill_rate_per_minute: int):
        """
        :param capacity: The max burst of requests allowed instantly.
        :param refill_rate_per_minute: How many tokens regenerate every 60 seconds.
        """
        self.capacity = capacity
        self.refill_rate_per_sec = refill_rate_per_minute / 60.0
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

    def __call__(self, request: Request):
        # 1. Get real client IP (crucial for Render's reverse proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        ip = forwarded.split(",")[0].strip() if forwarded else request.client.host
        
        now = time.time()

        # 2. Initialize IP bucket if it is their first request
        if ip not in self.tokens:
            self.tokens[ip] = float(self.capacity)
            self.last_update[ip] = now

        # 3. Refill the bucket based on exact elapsed time
        time_elapsed = now - self.last_update[ip]
        new_tokens = time_elapsed * self.refill_rate_per_sec
        
        self.tokens[ip] = min(float(self.capacity), self.tokens[ip] + new_tokens)
        self.last_update[ip] = now

        # 4. Consume a token or block the request
        if self.tokens[ip] >= 1.0:
            self.tokens[ip] -= 1.0
        else:
            raise HTTPException(
                status_code=429,
                detail=f"Too Many Requests. Limit is {int(self.refill_rate_per_sec * 60)} per minute."
            )

# Create an instance for the predict route (20 requests per minute burst)
standard_limit = TokenBucketLimiter(capacity=20, refill_rate_per_minute=20)

@app.get("/", include_in_schema=False)
def root():
    # Automatically redirect users to the Swagger documentation
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": ml_engine.model is not None}

# Inject the standard Token Bucket limit
@app.post("/api/v1/predict", response_model=PredictionResponse, dependencies=[Depends(standard_limit)])
def predict(transaction: TransactionRequest):
    df = pd.DataFrame([transaction.model_dump()])
    results = ml_engine.predict_batch(df)
    return results[0]

if __name__ == "__main__":
    # Read port assigned by Cloud Run (default to 8080 if local)
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
