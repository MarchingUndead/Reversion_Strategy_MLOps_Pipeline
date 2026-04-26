"""Additional Prometheus instruments for the model server.

Defined in a separate module so src/serve.py stays thin and the existing
Counter/Histogram/Counter trio there is left untouched. All instruments
register against the default REGISTRY so the existing /metrics ASGI mount
exports them automatically.

Imported by src/serve.py. Run via `uvicorn src.serve:app`.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

IN_FLIGHT = Gauge(
    "inference_in_flight",
    "Number of requests currently being served",
    ["endpoint"],
)

MODEL_MEMORY_BYTES = Gauge(
    "model_memory_bytes",
    "Resident set size of the server process at boot, after model load",
    ["model_uri"],
)

PAYLOAD_ROWS = Summary(
    "payload_rows",
    "Number of rows in each /invocations request payload",
    ["endpoint"],
)

PREDICTION_SCORE = Histogram(
    "prediction_score",
    "Distribution of model output values (foundation for prediction-drift checks)",
    ["endpoint"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_INFO = Gauge(
    "model_info",
    "Static gauge (always 1) carrying provenance labels for the loaded model",
    ["model_uri", "git_commit", "mlflow_run_id"],
)

CLIENT_REQUESTS = Counter(
    "client_requests_total",
    "Per-client request count, separate from prediction_requests_total to keep cardinality bounded",
    ["client_id", "endpoint"],
)
