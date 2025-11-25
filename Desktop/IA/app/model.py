import os
import pickle
from joblib import load
from typing import Any, List, Optional
import tempfile
import requests

_model: Optional[Any] = None

def _load_model(path: str) -> Any:
    try:
        return load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def get_model() -> Any:
    global _model
    if _model is not None:
        return _model
    model_path_env = os.environ.get("MODEL_PATH")
    model_url_env = os.environ.get("MODEL_URL")
    if model_path_env and os.path.exists(model_path_env):
        _model = _load_model(model_path_env)
        return _model
    if model_url_env:
        tmp_dir = tempfile.gettempdir()
        tmp_file = os.path.join(tmp_dir, "model.pkl")
        if not os.path.exists(tmp_file):
            r = requests.get(model_url_env, timeout=30)
            r.raise_for_status()
            with open(tmp_file, "wb") as f:
                f.write(r.content)
        _model = _load_model(tmp_file)
        return _model
    default_path = os.path.expanduser("~/Downloads/mejor_modelo (2).pkl")
    if os.path.exists(default_path):
        _model = _load_model(default_path)
        return _model
    raise FileNotFoundError("Modelo no encontrado. Configure 'MODEL_PATH' o 'MODEL_URL'.")
    return _model

def get_feature_names(model: Optional[Any] = None) -> Optional[List[str]]:
    m = model or get_model()
    return getattr(m, "feature_names_in_", None)

def has_predict_proba(model: Optional[Any] = None) -> bool:
    m = model or get_model()
    return hasattr(m, "predict_proba")

def get_classes(model: Optional[Any] = None) -> Optional[List[Any]]:
    m = model or get_model()
    return getattr(m, "classes_", None)
