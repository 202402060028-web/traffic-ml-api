# app.py
import os
import sys
from typing import Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Traffic Congestion Classifier")

# -------- Pydantic input model: only the count features ----------
class TrafficCounts(BaseModel):
    CarCount: float = Field(..., example=120.0)
    BikeCount: float = Field(..., example=80.0)
    BusCount: float = Field(..., example=15.0)
    TruckCount: float = Field(..., example=25.0)

# -------- Helper to load artifacts safely ----------
def load_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)

# Load model and scaler at startup (fail fast with clear message)
try:
    MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pkl")
    SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.pkl")

    model = load_artifact(MODEL_PATH)
    scaler = load_artifact(SCALER_PATH)
except Exception as e:
    # If startup fails, log and exit so Render shows the failure
    print("ERROR loading artifacts:", e)
    sys.stderr.write(f"ERROR loading artifacts: {e}\n")
    # Stop startup so you'll see the error immediately in render logs
    raise

# If model has classes_ we can decode numeric predictions to labels
_label_map = None
try:
    if hasattr(model, "classes_"):
        # model.classes_ might be array of strings or numbers
        _label_map = list(model.classes_)
        print("Detected model.classes_:", _label_map)
except Exception:
    _label_map = None

# ---------- Routes ----------
@app.get("/")
def read_root():
    return {"message": "Traffic Congestion Model API Running"}

@app.post("/predict")
def predict(counts: TrafficCounts):
    """
    Predict traffic situation from counts:
    Input JSON sample:
    {
      "CarCount": 120,
      "BikeCount": 80,
      "BusCount": 15,
      "TruckCount": 25
    }
    """
    try:
        # Build feature vector in the exact order your model expects
        features = np.array([[ 
            counts.CarCount,
            counts.BikeCount,
            counts.BusCount,
            counts.TruckCount
        ]], dtype=float)

        # Apply scaler (if provided)
        try:
            features_scaled = scaler.transform(features)
        except Exception as e:
            # if scaler.transform fails, provide informative message
            raise RuntimeError(f"Scaler transform error: {e}")

        # Predict
        pred = model.predict(features_scaled)

        # If prediction is an array, take first
        if isinstance(pred, (list, tuple, np.ndarray)):
            pred_val = pred[0]
        else:
            pred_val = pred

        # Try decode to label if classes_ present
        decoded_label: Optional[str] = None
        if _label_map is not None:
            # If model returned integer index, try mapping
            try:
                # If pred_val is integer-like and within range
                if isinstance(pred_val, (int, np.integer)):
                    idx = int(pred_val)
                    if 0 <= idx < len(_label_map):
                        decoded_label = str(_label_map[idx])
                else:
                    # If model returned the original class label (string), use it
                    decoded_label = str(pred_val)
            except Exception:
                decoded_label = None

        # If we couldn't decode into a readable label, return numeric prediction as string too
        response = {"prediction_numeric": int(pred_val) if isinstance(pred_val, (int, np.integer)) else str(pred_val)}
        if decoded_label is not None:
            response["prediction_label"] = decoded_label
        else:
            response["prediction_label"] = str(pred_val)

        return response

    except HTTPException:
        raise
    except Exception as e:
        # Return 500 with message for easier debugging
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
