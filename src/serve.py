"""Instrumented model server.

Loads an MLflow-logged model via MODEL_URI and exposes:
  POST /invocations  — MLflow dataframe_split payload, returns {"predictions": [...]}
  GET  /metrics      — Prometheus scrape endpoint
  GET  /health       — liveness
  GET  /ready        — readiness (200 only after model load + dummy predict)

Run:
  $env:MODEL_URI = "runs:/<run_id>/model_classifier"
  python src/serve.py                       # bind host/port read from config.yaml
  uvicorn src.serve:app --host 0.0.0.0 --port 5002   # explicit override
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from src.serve_metrics import (
    CLIENT_REQUESTS,
    IN_FLIGHT,
    MODEL_INFO,
    MODEL_MEMORY_BYTES,
    PAYLOAD_ROWS,
    PREDICTION_SCORE,
)

def _resolve_model_uri(raw: str) -> str:
    """Translate `auto:<name>` -> `models:/<name>/Production` (or the latest
    version if no Production stage exists). `runs:/...` and `models:/...` URIs
    pass through unchanged. External wrapper — preserves the existing
    load_model call unchanged."""
    if not raw or not raw.startswith("auto:"):
        return raw
    name = raw.split("auto:", 1)[1].strip()
    from mlflow import MlflowClient
    client = MlflowClient()
    prod = client.get_latest_versions(name, stages=["Production"])
    if prod:
        v = prod[0]
        print(f"[serve] auto: {name} -> models:/{name}/Production "
              f"(v{v.version}, run={v.run_id})")
        return f"models:/{name}/Production"
    none = client.get_latest_versions(name, stages=["None"])
    if none:
        v = none[0]
        print(f"[serve] auto: {name} no Production stage; "
              f"falling back to v{v.version}")
        return f"models:/{name}/{v.version}"
    raise SystemExit(f"auto:{name}: no versions found in MLflow Registry")


MODEL_URI = _resolve_model_uri(os.environ.get("MODEL_URI", ""))
if not MODEL_URI:
    raise SystemExit("MODEL_URI env var is required "
                     "(e.g. runs:/<run_id>/model_classifier, "
                     "models:/<name>/Production, or auto:<name>)")

print(f"loading model from {MODEL_URI} ...")
_model = mlflow.pyfunc.load_model(MODEL_URI)
print("model loaded")

# Additionally load via the sklearn flavor to access predict_proba.
# The classifier's predict() returns string labels which the PREDICTION_SCORE
# float-coercion observer skips, leaving the prediction-drift panels empty.
# Observing max-class probability instead gives a value bounded in [0, 1] that
# matches the existing histogram buckets and populates on every request.
try:
    import mlflow.sklearn
    _sklearn_model = mlflow.sklearn.load_model(MODEL_URI)
    _has_proba = hasattr(_sklearn_model, "predict_proba")
    print(f"sklearn flavor loaded (predict_proba={_has_proba})")
except Exception as exc:
    print(f"sklearn load failed ({type(exc).__name__}: {exc}); proba instrumentation off")
    _sklearn_model, _has_proba = None, False

_READY = False


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _mlflow_run_id_from_uri(uri: str) -> str:
    if uri.startswith("runs:/"):
        return uri.split("/", 2)[1]
    return "unknown"


try:
    import psutil
    MODEL_MEMORY_BYTES.labels(model_uri=MODEL_URI).set(psutil.Process().memory_info().rss)
except Exception:
    pass

MODEL_INFO.labels(
    model_uri=MODEL_URI,
    git_commit=_git_commit(),
    mlflow_run_id=_mlflow_run_id_from_uri(MODEL_URI),
).set(1)

REQUESTS = Counter(
    "prediction_requests_total",
    "Total HTTP requests to the model server",
    ["endpoint", "status"],
)
LATENCY = Histogram(
    "prediction_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
ERRORS = Counter(
    "prediction_errors_total",
    "Total exceptions raised while handling requests",
    ["endpoint", "exception"],
)

app = FastAPI(title="reversion-model-server")
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def instrument(request: Request, call_next):
    endpoint = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status_class = f"{response.status_code // 100}xx"
        REQUESTS.labels(endpoint=endpoint, status=status_class).inc()
        return response
    except Exception as exc:
        ERRORS.labels(endpoint=endpoint, exception=type(exc).__name__).inc()
        REQUESTS.labels(endpoint=endpoint, status="5xx").inc()
        raise
    finally:
        LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)


@app.middleware("http")
async def track_in_flight_and_client(request: Request, call_next):
    endpoint = request.url.path
    client_id = request.headers.get("x-client-id", "anonymous")
    CLIENT_REQUESTS.labels(client_id=client_id, endpoint=endpoint).inc()
    IN_FLIGHT.labels(endpoint=endpoint).inc()
    try:
        return await call_next(request)
    finally:
        IN_FLIGHT.labels(endpoint=endpoint).dec()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.get("/ready")
def ready() -> JSONResponse:
    if not _READY:
        return JSONResponse({"status": "not-ready"}, status_code=503)
    return JSONResponse({"status": "ready", "model_uri": MODEL_URI})


@app.on_event("startup")
def _mark_ready() -> None:
    global _READY
    try:
        # Sanity predict on a single zero row matching the feature count.
        # If this fails, /ready stays 503 so orchestrators don't route traffic.
        sig = getattr(_model.metadata, "get_input_schema", lambda: None)()
        n_cols = len(sig.inputs) if sig else 12
        dummy = pd.DataFrame([[0.0] * n_cols])
        _model.predict(dummy)
        _READY = True
    except Exception as exc:
        print(f"readiness check failed: {type(exc).__name__}: {exc}")
        _READY = False


@app.post("/invocations")
async def invocations(request: Request) -> JSONResponse:
    payload = await request.json()
    split = payload.get("dataframe_split")
    if not split:
        return JSONResponse(
            {"error": "expected {'dataframe_split': {'columns': [...], 'data': [[...]]}}"},
            status_code=400,
        )
    df = pd.DataFrame(split["data"], columns=split["columns"])
    PAYLOAD_ROWS.labels(endpoint="/invocations").observe(len(df))
    preds = _model.predict(df)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    for p in preds:
        try:
            PREDICTION_SCORE.labels(endpoint="/invocations").observe(float(p))
        except (TypeError, ValueError):
            pass
    if _has_proba:
        try:
            for max_p in _sklearn_model.predict_proba(df).max(axis=1):
                PREDICTION_SCORE.labels(endpoint="/invocations").observe(float(max_p))
        except Exception:
            pass
    return JSONResponse({"predictions": preds})


if __name__ == "__main__":
    import uvicorn
    import yaml

    _cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(_cfg_path) as _f:
        _cfg = yaml.safe_load(_f)
    _serve_cfg = _cfg.get("serve", {}) or {}
    _bind_host = _serve_cfg.get("bind_host", "0.0.0.0")
    _port = int(_serve_cfg.get("port", 5002))
    print(f"[serve] launching uvicorn on {_bind_host}:{_port} (from config.yaml)")
    uvicorn.run(app, host=_bind_host, port=_port)
