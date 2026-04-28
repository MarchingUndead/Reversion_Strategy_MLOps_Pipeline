"""Single-run training driver for the model-architecture grid search.

One invocation = one (model_type, position, hyperparameters) point on the grid.
Logs all 3 heads (classifier / duration regressor / revert regressor) and full
metrics to MLflow under tags `model_type` and `position` so you can filter the
runs in the UI and register the winners.

Reuses M.load_events_all / M.prepare_events / M.split_events / M._clean from
src/model.py for data loading. Does NOT call M.train_position because that
function is hardcoded to RandomForest; this script swaps the underlying
estimator family based on --model.

Usage:
  python src/mlflow_grid.py --model rf  --position 0 --n-estimators 200 --max-depth 10
  python src/mlflow_grid.py --model xgb --position 2 --n-estimators 400 --max-depth 5 --learning-rate 0.05

Wrapper: scripts/grid_search.sh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import mlflow

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import model as M
from mlflow_utils import compute_head_metrics, log_three_heads, start_run
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class XGBStringClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper that pairs a fitted XGBClassifier with its
    LabelEncoder so .predict() returns the original string labels.

    Why: XGBoost requires numeric class targets, so mlflow_grid LabelEncodes
    before fit. The raw fitted XGB produces ints on .predict(); downstream
    consumers (backtest.evaluate, classification_report) expect strings and
    crash with "Mix of label input types". Logging this wrapper instead of
    the bare XGBClassifier closes the loop.

    Note: We do NOT re-fit. The model is fitted externally and passed in.
    """

    def __init__(self, model=None, label_encoder=None):
        self.model = model
        self.label_encoder = label_encoder

    def fit(self, X, y=None):
        return self  # already fitted upstream

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.label_encoder.classes_


def build_models(model_type, n_estimators, max_depth, learning_rate, random_state):
    """Return three unfitted estimators: (classifier, duration_regressor, revert_regressor)."""
    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        # n_jobs=1 inside each task: pool runs 2 mapped tasks in parallel, and
        # n_jobs=-1 inside both grabs all cores -> CPU thrashing -> tasks hang.
        common = dict(n_estimators=n_estimators, max_depth=max_depth,
                      n_jobs=1, random_state=random_state)
        return (
            RandomForestClassifier(**common),
            RandomForestRegressor(**common),
            RandomForestRegressor(**common),
        )
    if model_type == "xgb":
        import xgboost as xgb
        # Same reason: each task stays on 1 core so 2 parallel xgb tasks don't
        # fight over all cores. Drops per-task wall time vs n_jobs=-1 in this
        # specific 2-tasks-in-pool topology.
        common = dict(n_estimators=n_estimators, max_depth=max_depth,
                      learning_rate=learning_rate, random_state=random_state,
                      n_jobs=1, tree_method="hist")
        return (
            xgb.XGBClassifier(eval_metric="mlogloss", **common),
            xgb.XGBRegressor(**common),
            xgb.XGBRegressor(**common),
        )
    raise ValueError(f"unknown --model: {model_type!r} (choose 'rf' or 'xgb')")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    parser.add_argument("--position", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="xgb only; ignored for rf")
    parser.add_argument("--random-state", type=int,
                        default=M._cfg["model"]["random_state"])
    parser.add_argument("--experiment", default=None,
                        help="MLflow experiment name (default: $MLFLOW_EXPERIMENT_NAME or 'reversion-grid')")
    args = parser.parse_args()

    feature_cols = M._cfg["model"]["feature_cols"]

    # --- data load (reuses model.py utilities, untouched) -----------------
    events = M.prepare_events(M.load_events_all())
    train  = M.split_events(events, "train")
    val    = M.split_events(events, "val")

    train_sub = M._clean(train[train["position"] == args.position], feature_cols)
    val_sub   = M._clean(val[val["position"]   == args.position], feature_cols)
    if len(train_sub) < 30 or val_sub.empty:
        raise SystemExit(
            f"position={args.position}: insufficient data "
            f"(train={len(train_sub)}, val={len(val_sub)})"
        )

    X_train = train_sub[feature_cols].to_numpy()
    y_klas  = train_sub["klass"].to_numpy()
    y_dur   = train_sub["duration_sec"].to_numpy()
    y_rev   = train_sub["revert_delta"].to_numpy()

    X_val   = val_sub[feature_cols].to_numpy()

    # XGBoost classifier requires numeric class labels.
    le = LabelEncoder().fit(y_klas) if args.model == "xgb" else None
    y_klas_fit = le.transform(y_klas) if le is not None else y_klas

    clf, reg_dur, reg_rev = build_models(
        args.model, args.n_estimators, args.max_depth,
        args.learning_rate, args.random_state,
    )

    # --- MLflow run -------------------------------------------------------
    with start_run("reversion-grid",
                   override_experiment=args.experiment,
                   tracking_root=ROOT / "mlruns"):
        mlflow.set_tags({
            "model_type": args.model,
            "position":   str(args.position),
        })
        mlflow.log_params({
            "model":         args.model,
            "position":      args.position,
            "n_estimators":  args.n_estimators,
            "max_depth":     args.max_depth,
            "learning_rate": args.learning_rate if args.model == "xgb" else None,
            "random_state":  args.random_state,
            "n_train":       len(train_sub),
            "n_val":         len(val_sub),
        })

        clf.fit(X_train, y_klas_fit)
        reg_dur.fit(X_train, y_dur)
        reg_rev.fit(X_train, y_rev)

        pred_k_raw = clf.predict(X_val)
        pred_k     = le.inverse_transform(pred_k_raw) if le is not None else pred_k_raw
        pred_dur   = reg_dur.predict(X_val)
        pred_rev   = reg_rev.predict(X_val)

        metrics = compute_head_metrics(
            val_sub["klass"],        pred_k,
            val_sub["duration_sec"], pred_dur,
            val_sub["revert_delta"], pred_rev,
        )
        mlflow.log_metrics(metrics)

        # For XGB, log the wrapper (model + label encoder) so consumers get
        # strings back. RF naturally returns strings; log it as-is.
        clf_to_log = (XGBStringClassifier(model=clf, label_encoder=le)
                      if le is not None else clf)
        log_three_heads(clf_to_log, reg_dur, reg_rev)

        run_id = mlflow.active_run().info.run_id
        lr_str = f"{args.learning_rate:g}" if args.model == "xgb" else "-"
        print(f"[grid] model={args.model} pos={args.position} "
              f"n_est={args.n_estimators} max_depth={args.max_depth} lr={lr_str}  "
              f"f1={metrics['f1_macro']:.4f} acc={metrics['accuracy']:.4f} "
              f"mae_dur={metrics['mae_duration']:.1f} mae_rev={metrics['mae_revert']:.3f}  "
              f"run_id={run_id}")


if __name__ == "__main__":
    main()
