"""MLflow-wrapped training entry point.

Thin wrapper around src/model.py. Does NOT modify any pipeline function —
just mutates the in-memory config dict to pass hyperparameters and calls
train_position()/evaluate() as-is, then logs everything to MLflow.

Skeleton scope: position 0 only. One MLflow run per invocation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import mlflow

SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import model as M
from mlflow_utils import compute_head_metrics, log_three_heads, start_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-n-estimators", type=int, default=M._cfg["model"]["rf_n_estimators"])
    parser.add_argument("--rf-max-depth",    type=int, default=M._cfg["model"]["rf_max_depth"])
    parser.add_argument("--position",        type=int, default=0)
    parser.add_argument("--symbol",          required=True, choices=M._cfg["symbols"],
                        help="Train a model specific to this symbol — per-(symbol, position).")
    args = parser.parse_args()

    M._cfg["model"]["rf_n_estimators"] = args.rf_n_estimators
    M._cfg["model"]["rf_max_depth"]    = args.rf_max_depth

    feature_cols = M._cfg["model"]["feature_cols"]
    position     = args.position

    events = M.prepare_events(M.load_events_all())
    train  = M.split_events(events, "train")
    val    = M.split_events(events, "val")
    # Pre-filter by symbol externally — model.py:train_position / evaluate are
    # untouched (preserve-core-logic). They still filter by position internally.
    train = train[train["symbol"] == args.symbol]
    val   = val[val["symbol"]   == args.symbol]
    print(f"symbol={args.symbol}  train={len(train)}  val={len(val)}")

    with start_run("reversion-grid", tracking_root=ROOT / "mlruns"):
        mlflow.set_tags({"position": str(position), "symbol": args.symbol})
        mlflow.log_params({
            "rf_n_estimators": args.rf_n_estimators,
            "rf_max_depth":    args.rf_max_depth,
            "random_state":    M._cfg["model"]["random_state"],
            "position":        position,
            "symbol":          args.symbol,
        })

        trio = M.train_position(train, position, feature_cols)
        if trio is None:
            raise SystemExit(f"no training data for symbol={args.symbol} position={position}")
        clf, reg_dur, reg_rev = trio

        eval_df = M.evaluate(val, position, feature_cols, {position: trio})
        mlflow.log_metrics(compute_head_metrics(
            eval_df["klass"],        eval_df["pred_klass"],
            eval_df["duration_sec"], eval_df["pred_dur"],
            eval_df["revert_delta"], eval_df["pred_rev"],
        ))

        # In-memory CSV via mlflow.log_text — avoids temp-file writes, which
        # fail when ROOT (/opt/airflow/project/) is mounted read-only in the
        # airflow-scheduler container.
        import io
        _buf = io.StringIO()
        eval_df.head(200).to_csv(_buf, index=False)
        mlflow.log_text(_buf.getvalue(), "predictions/sample_predictions.csv")

        log_three_heads(clf, reg_dur, reg_rev)
        print("logged run:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
