import os
import pickle
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

_model: Optional[Any] = None

def _load_model(path: str) -> Any:
    try:
        return load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def download_file(url: str, dest: str) -> None:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def get_model() -> Any:
    global _model
    if _model is not None:
        return _model
    load_dotenv()
    model_path_env = os.environ.get("MODEL_PATH")
    model_url_env = os.environ.get("MODEL_URL")
    if model_path_env and os.path.exists(model_path_env):
        _model = _load_model(model_path_env)
        return _model
    if model_url_env:
        tmp_dir = tempfile.gettempdir()
        tmp_file = os.path.join(tmp_dir, "model.pkl")
        if not os.path.exists(tmp_file) or os.path.getsize(tmp_file) == 0:
            download_file(model_url_env, tmp_file)
        _model = _load_model(tmp_file)
        return _model
    default_path = os.path.expanduser("~/Downloads/mejor_modelo (2).pkl")
    if os.path.exists(default_path):
        _model = _load_model(default_path)
        return _model
    raise HTTPException(status_code=500, detail="Modelo no encontrado. Configure 'MODEL_PATH' o 'MODEL_URL'.")

def get_feature_names(model: Optional[Any] = None) -> Optional[List[str]]:
    m = model or get_model()
    return getattr(m, "feature_names_in_", None)

def has_predict_proba(model: Optional[Any] = None) -> bool:
    m = model or get_model()
    return hasattr(m, "predict_proba")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    features: Optional[Dict[str, Any]] = None
    X: Optional[List[Any]] = None

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {
        "message": "API de modelo desplegada",
        "endpoints": ["/health", "/model-info", "/predict", "/docs"],
    }

@app.get("/model-info")
def model_info():
    try:
        m = get_model()
        fn = get_feature_names(m)
        feature_names = None
        if fn is not None:
            try:
                feature_names = [str(x) for x in list(fn)]
            except Exception:
                feature_names = [str(x) for x in fn]
        return {
            "model_type": type(m).__name__,
            "has_feature_names": feature_names is not None,
            "feature_names": feature_names,
            "has_predict_proba": has_predict_proba(m),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(payload: PredictRequest):
    m = get_model()
    if payload.features is None and payload.X is None:
        raise HTTPException(status_code=400, detail="Debe enviar 'features' o 'X'.")
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
    else:
        X_input = np.array([payload.X])
    y_pred = m.predict(X_input)
    result = {"y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)}
    if has_predict_proba(m):
        y_proba = m.predict_proba(X_input)
        result["y_proba"] = y_proba.tolist() if hasattr(y_proba, "tolist") else list(y_proba)
    return result
