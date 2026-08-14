from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class TransactionType(str, Enum):
    TRANSFER = "TRANSFER"
    PAYMENT = "PAYMENT"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    transaction_type: TransactionType
    old_balance: float = Field(..., ge=0)
    new_balance: float = Field(..., ge=0)
    device_age_days: int = Field(..., ge=0)
    account_age_days: int = Field(..., ge=0)
    transactions_last_hour: int = Field(..., ge=0)
    transactions_last_day: int = Field(..., ge=0)
    failed_transactions_last_day: int = Field(..., ge=0)
    distance_from_usual_location_km: float = Field(..., ge=0)
    is_new_device: bool
    is_new_location: bool

class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest]

class RiskSignal(BaseModel):
    feature: str
    impact: str
    reason: str

class ModelInfo(BaseModel):
    name: str
    version: str

class PredictionResponse(BaseModel):
    prediction: str
    fraud_probability: float
    risk_score: int
    risk_level: str
    risk_signals: List[RiskSignal]
    model: ModelInfo
    timestamp: str

class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]

class ErrorResponse(BaseModel):
    code: str
    message: str