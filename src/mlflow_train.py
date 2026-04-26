"""MLflow-wrapped training entry point.

Thin wrapper around src/model.py. Does NOT modify any pipeline function —
just mutates the in-memory config dict to pass hyperparameters and calls
train_position()/evaluate() as-is, then logs everything to MLflow.

Skeleton scope: position 0 only. One MLflow run per invocation.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import numpy as np
import mlflow
import mlflow.sklearn

SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import model as M


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-n-estimators", type=int, default=M._cfg["model"]["rf_n_estimators"])
    parser.add_argument("--rf-max-depth",    type=int, default=M._cfg["model"]["rf_max_depth"])
    parser.add_argument("--position",        type=int, default=0)
    args = parser.parse_args()

    M._cfg["model"]["rf_n_estimators"] = args.rf_n_estimators
    M._cfg["model"]["rf_max_depth"]    = args.rf_max_depth

    feature_cols = M._cfg["model"]["feature_cols"]
    position     = args.position

    events = M.prepare_events(M.load_events_all())
    train  = M.split_events(events, "train")
    val    = M.split_events(events, "val")
    print(f"train={len(train)}  val={len(val)}")

    if mlflow.active_run() is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file:{(ROOT / 'mlruns').as_posix()}"
        experiment   = os.environ.get("MLFLOW_EXPERIMENT_NAME", "reversion-skeleton")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        run_ctx = mlflow.start_run()
    else:
        run_ctx = contextlib.nullcontext(mlflow.active_run())

    with run_ctx:
        mlflow.log_params({
            "rf_n_estimators": args.rf_n_estimators,
            "rf_max_depth":    args.rf_max_depth,
            "random_state":    M._cfg["model"]["random_state"],
            "position":        position,
        })

        trio = M.train_position(train, position, feature_cols)
        if trio is None:
            raise SystemExit(f"no training data for position={position}")
        clf, reg_dur, reg_rev = trio

        eval_df = M.evaluate(val, position, feature_cols, {position: trio})
        metrics = M.eval_metrics(eval_df)

        rmse_duration = float(np.sqrt(np.mean((eval_df["duration_sec"] - eval_df["pred_dur"]) ** 2)))
        rmse_revert   = float(np.sqrt(np.mean((eval_df["revert_delta"] - eval_df["pred_rev"]) ** 2)))

        mlflow.log_metrics({
            "n_test":        float(metrics["n_test"]),
            "accuracy":      metrics["accuracy"],
            "f1_macro":      metrics["f1_macro"],
            "mae_duration":  metrics["mae_duration"],
            "mae_revert":    metrics["mae_revert"],
            "rmse_duration": rmse_duration,
            "rmse_revert":   rmse_revert,
        })

        sample_path = ROOT / "mlflow_tmp_sample_predictions.csv"
        eval_df.head(200).to_csv(sample_path, index=False)
        mlflow.log_artifact(str(sample_path), artifact_path="predictions")
        sample_path.unlink(missing_ok=True)

        mlflow.sklearn.log_model(clf,     artifact_path="model_classifier")
        mlflow.sklearn.log_model(reg_dur, artifact_path="model_duration")
        mlflow.sklearn.log_model(reg_rev, artifact_path="model_revert")

        print("logged run:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
