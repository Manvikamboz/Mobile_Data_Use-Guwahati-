import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager


# ---------------------------------------------------------------------------
# Structured JSON Logging Setup
# Handbook Section 9: "Use the logging module with a JSON formatter: one
# JSON line per prediction, written to the screen (stdout), never to a file.
# Never log a customer's personal details."
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)


logger = logging.getLogger("mp238_api")
logger.setLevel(logging.INFO)
# Avoid duplicate handlers if reloaded
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonFormatter())
    logger.addHandler(stream_handler)
logger.propagate = False
 
 
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model

    if not DEFAULT_MODEL_PATH.exists():
        raise RuntimeError(
            "Model file not found: models/model.joblib"
        )

    _model = joblib.load(DEFAULT_MODEL_PATH)

    logger.info(
        "Model loaded successfully",
        extra={
            "extra_data": {
                "event": "model_loaded"
            }
        }
    )

    yield

    _model = None


app = FastAPI(
    title="MP238 Mobile Data Use Prediction Service",
    description="Predicts daily mobile data use (gb_per_day) for Guwahati Telecom Circle Office",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model():
    global _model
    if _model is None and DEFAULT_MODEL_PATH.exists():
        _model = joblib.load(DEFAULT_MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Pydantic Schemas
# Client's two rules in PROBLEM.md Section 5:
#   1. video_minutes above 550 is impossible for us. Reject it. (le=550)
#   2. apps_installed below 5 is always a typing mistake. Reject it. (ge=5)
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(..., description="Customer age in years")
    screen_hours: float = Field(..., description="Screen time in hours")
    apps_installed: int = Field(
        ...,
        ge=5,
        description="Number of apps installed (must be >= 5; below 5 is rejected as typing mistake)",
    )
    plan_gb_limit: float = Field(..., description="Daily data limit of the plan in GB")
    video_minutes: float = Field(
        ...,
        le=550,
        description="Video minutes watched in a day (must be <= 550; above 550 is impossible)",
    )
    wifi_hours: float = Field(..., description="Time on home Wi-Fi in hours")
    phone_tier: str = Field(..., description="Phone tier (basic, mid, flagship)")
    circle: str = Field(..., description="Area type (Urban, Semi-urban, Rural)")


class PredictResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint required by Handbook Section 9 and k8s probes."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Predicts mobile data usage (gb_per_day) from customer daily activity features.
    Logs structured JSON line to stdout without any customer personal details.
    """
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Ensure models/model.joblib is built.",
        )

    # Convert request payload to DataFrame matching feature column names
    input_data = {
        "age": [payload.age],
        "screen_hours": [payload.screen_hours],
        "apps_installed": [payload.apps_installed],
        "plan_gb_limit": [payload.plan_gb_limit],
        "video_minutes": [payload.video_minutes],
        "wifi_hours": [payload.wifi_hours],
        "phone_tier": [payload.phone_tier],
        "circle": [payload.circle],
    }
    input_df = pd.DataFrame(input_data)

    try:
        raw_pred = model.predict(input_df)
        pred_value = float(raw_pred[0])
    except Exception as e:
        logger.error(
            "Prediction inference error",
            extra={"extra_data": {"error": str(e)}},
        )
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Structured JSON log (stdout only, no PII)
    logger.info(
        "Prediction generated",
        extra={
            "extra_data": {
                "event": "prediction",
                "phone_tier": payload.phone_tier,
                "circle": payload.circle,
                "plan_gb_limit": payload.plan_gb_limit,
                "prediction_gb_per_day": round(pred_value, 4),
            }
        },
    )

    return {"prediction": float(pred_value)}
