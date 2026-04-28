"""Shared MLflow helpers for mlflow_train.py and mlflow_grid.py.

Both drivers produce three model heads (classifier, duration regressor,
revert regressor) and log the same seven metrics. This module centralises
the common run setup, metric computation, and artefact logging so the two
scripts only differ where they genuinely should: how they build/fit the
estimators (frozen RF in train, swappable family in grid).

Does NOT touch any function in src/model.py — model.py helpers stay untouched
per the project's preserve-core-logic rule.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def start_run(default_experiment: str,
              *,
              override_experiment: str | None = None,
              tracking_root: Path | None = None):
    """Open or join an MLflow run. Returns a context manager.

    If a run is already active (e.g. via `mlflow run`), return a nullcontext
    so we don't nest. Otherwise set tracking_uri + experiment from env
    (with sensible fallbacks) and call `start_run()`.

    Resolution order:
        tracking_uri = $MLFLOW_TRACKING_URI or file:{tracking_root}
        experiment   = override_experiment or $MLFLOW_EXPERIMENT_NAME or default_experiment
    """
    if mlflow.active_run() is not None:
        return contextlib.nullcontext(mlflow.active_run())
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri is None and tracking_root is not None:
        tracking_uri = f"file:{Path(tracking_root).as_posix()}"
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = (override_experiment
                  or os.environ.get("MLFLOW_EXPERIMENT_NAME")
                  or default_experiment)
    mlflow.set_experiment(experiment)
    return mlflow.start_run()


def compute_head_metrics(y_klass_true: Any, y_klass_pred: Any,
                         y_dur_true: Any,   y_dur_pred: Any,
                         y_rev_true: Any,   y_rev_pred: Any) -> dict[str, float]:
    """Seven-metric block shared by train and grid.

    Returns a dict suitable for `mlflow.log_metrics(...)` directly.
    """
    y_dur_true_a = np.asarray(y_dur_true, dtype=float)
    y_dur_pred_a = np.asarray(y_dur_pred, dtype=float)
    y_rev_true_a = np.asarray(y_rev_true, dtype=float)
    y_rev_pred_a = np.asarray(y_rev_pred, dtype=float)
    return {
        "n_test":        float(len(y_klass_true)),
        "accuracy":      float(accuracy_score(y_klass_true, y_klass_pred)),
        "f1_macro":      float(f1_score(y_klass_true, y_klass_pred,
                                        average="macro", zero_division=0)),
        "mae_duration":  float(mean_absolute_error(y_dur_true_a, y_dur_pred_a)),
        "mae_revert":    float(mean_absolute_error(y_rev_true_a, y_rev_pred_a)),
        "rmse_duration": float(np.sqrt(np.mean((y_dur_true_a - y_dur_pred_a) ** 2))),
        "rmse_revert":   float(np.sqrt(np.mean((y_rev_true_a - y_rev_pred_a) ** 2))),
    }


def log_three_heads(clf, reg_dur, reg_rev) -> None:
    """Log all three sklearn artefacts under the names register_best.py expects."""
    mlflow.sklearn.log_model(clf,     artifact_path="model_classifier")
    mlflow.sklearn.log_model(reg_dur, artifact_path="model_duration")
    mlflow.sklearn.log_model(reg_rev, artifact_path="model_revert")
