from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np

from .model import get_model, get_feature_names, has_predict_proba, get_classes

app = FastAPI()


class PredictRequest(BaseModel):
    features: Optional[Dict[str, Any]] = None
    X: Optional[List[Any]] = None


@app.get("/model-info")
def model_info():
    try:
        m = get_model()
        feature_names = get_feature_names(m)
        if feature_names is not None:
            try:
                feature_names = [str(x) for x in list(feature_names)]
            except Exception:
                feature_names = [str(x) for x in feature_names]
        return {
            "model_type": type(m).__name__,
            "has_feature_names": feature_names is not None,
            "feature_names": feature_names,
            "has_predict_proba": has_predict_proba(m),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        m = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if payload.features is None and payload.X is None:
        raise HTTPException(status_code=400, detail="Debe enviar 'features' o 'X'.")

    X_input = None

    if payload.features is not None:
        fn = get_feature_names(m)
        if fn is not None:
            missing = [f for f in fn if f not in payload.features]
            if missing:
                raise HTTPException(status_code=400, detail=f"Faltan features: {missing}")
            row = [payload.features[f] for f in fn]
            X_input = np.array([row])
        else:
            keys = sorted(payload.features.keys())
            row = [payload.features[k] for k in keys]
            X_input = np.array([row])

    elif payload.X is not None:
        X_input = [payload.X]

    try:
        y_pred = m.predict(X_input)
        result = {"y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)}
        if has_predict_proba(m):
            y_proba = m.predict_proba(X_input)
            result["y_proba"] = y_proba.tolist() if hasattr(y_proba, "tolist") else list(y_proba)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
