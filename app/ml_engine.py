import os
import json
import logging
import datetime
from typing import List, Dict
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Configurable Risk Thresholds
FRAUD_THRESHOLD = 0.65
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline identical to training logic."""
    df_feat = data.copy()
    df_feat['balance_change'] = df_feat['old_balance'] - df_feat['new_balance']
    df_feat['amount_to_old_balance_ratio'] = df_feat['amount'] / (df_feat['old_balance'] + 1)
    df_feat['transaction_amount_log'] = np.log1p(df_feat['amount'])
    df_feat['velocity_risk'] = df_feat['transactions_last_hour'] * df_feat['transactions_last_day']
    return df_feat


class FraudPredictionEngine:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "model.json")
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
        self.metadata_path = os.path.join(model_dir, "metadata.json")

        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.feature_names = None

        self.load_artifacts()

    def load_artifacts(self):
        try:
            # 1. Load XGBoost natively
            self.model_path = os.path.join(self.model_dir, "model.json")
            self.model = XGBClassifier()
            self.model.load_model(self.model_path)
            
            # 2. Load Preprocessor via joblib
            self.preprocessor = joblib.load(self.preprocessor_path)
            
            # 3. Load Metadata
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
                
            self.feature_names = self.preprocessor.get_feature_names_out()
            logger.info("ML Engine: Artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"ML Engine: Failed to load artifacts. Error: {e}")

    def predict_batch(self, transactions_df: pd.DataFrame) -> List[Dict]:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model artifacts not loaded.")

        # 1. Feature Engineering
        df_engineered = engineer_features(transactions_df)

        # 2. Preprocessing
        X_processed = self.preprocessor.transform(df_engineered)

        # 3. Model Inference
        probs = self.model.predict_proba(X_processed)[:, 1]

        # 4. Feature Importance
        if hasattr(self.model, "feature_importances_"):
            global_importance = self.model.feature_importances_
        else:
            global_importance = np.abs(self.model.coef_[0])

        results = []
        for i, prob in enumerate(probs):
            risk_score = int(prob * 100)

            if prob < LOW_RISK_THRESHOLD:
                risk_level = "LOW"
            elif prob < HIGH_RISK_THRESHOLD:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            prediction = "fraud" if prob >= FRAUD_THRESHOLD else "legitimate"

            local_scores = np.abs(X_processed[i]) * global_importance
            top_indices = np.argsort(local_scores)[-3:][::-1]

            risk_signals = []
            for idx in top_indices:
                raw_feat_name = self.feature_names[idx] if self.feature_names is not None else f"feature_{idx}"
                clean_feat = raw_feat_name.split('__')[-1]

                reasons = {
                    "is_new_device": "Transaction originated from a newly observed device.",
                    "distance_from_usual_location_km": "Transaction occurred unusually far from normal location.",
                    "amount": "The transaction amount is highly anomalous for this profile.",
                    "transactions_last_hour": "Transaction velocity (count per hour) is unusually high.",
                    "is_new_location": "Transaction originated from a newly observed location.",
                    "balance_change": "The change in account balance is severe.",
                    "failed_transactions_last_day": "A high number of recent transaction failures were detected."
                }

                reason_text = reasons.get(clean_feat, f"Anomalous pattern detected in {clean_feat}.")

                if local_scores[idx] > 0.01:
                    risk_signals.append({
                        "feature": clean_feat,
                        "impact": "high" if local_scores[idx] > 0.5 else "medium",
                        "reason": reason_text
                    })

            results.append({
                "prediction": prediction,
                "fraud_probability": round(float(prob), 4),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_signals": risk_signals,
                "model": {
                    "name": (self.metadata or {}).get("model_name", "fraud_detection_model"),
                    "version": (self.metadata or {}).get("version", "1.0.0")
                },
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
            })

        return results