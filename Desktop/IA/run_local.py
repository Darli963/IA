#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
from pathlib import Path

try:
    import requests
except ImportError:
    print("Falta 'requests'. Instala dependencias: python3 -m pip install requests python-dotenv uvicorn fastapi joblib numpy scikit-learn lightgbm")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Falta 'python-dotenv'. Instala: python3 -m pip install python-dotenv")
    sys.exit(1)


DEFAULT_MODEL_URL = "https://github.com/Darli963/IA/releases/download/v1.0.0/mejor_modelo.2.pkl"
ENV_FILE = Path(".env")
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
BASE = f"http://{SERVER_HOST}:{SERVER_PORT}"


def ensure_env_file():
    if ENV_FILE.exists():
        return
    ENV_FILE.write_text(
        f"MODEL_URL={DEFAULT_MODEL_URL}\nMODEL_PATH=\n",
        encoding="utf-8",
    )
    print(f"Creado .env con MODEL_URL por defecto\n{DEFAULT_MODEL_URL}")


def load_env():
    load_dotenv()
    model_url = os.environ.get("MODEL_URL", "").strip()
    model_path = os.environ.get("MODEL_PATH", "").strip()
    return model_url, model_path


def download_file(url: str, dest: Path):
    print(f"Descargando modelo desde {url} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Descarga completa ({size_mb:.2f} MB)")


def ensure_local_model(model_url: str, model_path: str) -> str:
    # Si MODEL_PATH está vacío, usar ./model.pkl
    if not model_path:
        model_path = str(Path("model.pkl").resolve())
        os.environ["MODEL_PATH"] = model_path

    dest = Path(model_path)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Modelo local existente: {dest}")
        return model_path

    if not model_url:
        print("MODEL_URL no definido y no existe modelo local. El backend intentará su propio flujo, pero es probable que falle.")
        return model_path

    try:
        download_file(model_url, dest)
    except requests.HTTPError as e:
        print(f"Error HTTP al descargar el modelo: {e}")
    except Exception as e:
        print(f"Error al descargar el modelo: {e}")
    return model_path


def start_server():
    env = os.environ.copy()
    cmd = ["uvicorn", "app.model:app", "--host", SERVER_HOST, "--port", str(SERVER_PORT), "--reload"]
    print(f"Iniciando servidor: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)
    return proc


def wait_for(url: str, timeout: float = 30.0):
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r
            last_err = f"HTTP {r.status_code}: {r.text}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.5)
    raise RuntimeError(f"No responde {url} tras {timeout}s. Último error: {last_err}")


def main():
    ensure_env_file()
    model_url, model_path = load_env()
    model_path = ensure_local_model(model_url, model_path)

    print(f"Variables de entorno efectivas:")
    print(f"- MODEL_URL = {os.environ.get('MODEL_URL', '')}")
    print(f"- MODEL_PATH = {os.environ.get('MODEL_PATH', '')}")

    proc = start_server()
    try:
        print("Esperando /health ...")
        r_health = wait_for(f"{BASE}/health", timeout=40)
        print(f"/health → {r_health.status_code} {r_health.text}")

        print("Esperando /model-info ...")
        try:
            r_info = wait_for(f"{BASE}/model-info", timeout=40)
            print(f"/model-info → {r_info.status_code}")
            try:
                print(json.dumps(r_info.json(), ensure_ascii=False, indent=2))
            except Exception:
                print(r_info.text)
        except Exception as e:
            print(f"/model-info no respondió correctamente: {e}")

        print(f"Servidor corriendo en {BASE} (Ctrl+C para detener)")
        proc.wait()
    except KeyboardInterrupt:
        print("Deteniendo servidor...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    except Exception as e:
        print(f"Error durante verificación: {e}")
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
