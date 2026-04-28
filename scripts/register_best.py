"""Register the top run per (position, symbol) from the grid-search experiment.

For each (position, symbol) pair, finds the run with the highest metrics.f1_macro
and registers all three heads (classifier, duration, revert) from that run
under names:

    reversion-classifier-pos{pos}-{symbol}
    reversion-duration-pos{pos}-{symbol}
    reversion-revert-pos{pos}-{symbol}

Optionally promotes the new version to Production with --promote.

Usage:
    python scripts/register_best.py
    python scripts/register_best.py --promote
    python scripts/register_best.py --experiment reversion-grid --metric f1_macro
    python scripts/register_best.py --position 2 --symbol BAJFINANCE --promote
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

HEADS = ["classifier", "duration", "revert"]
_cfg = yaml.safe_load(open(ROOT / "config.yaml"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="reversion-grid",
                        help="MLflow experiment to search (default: reversion-grid)")
    parser.add_argument("--metric", default="f1_macro",
                        help="Metric to rank runs by (default: f1_macro)")
    parser.add_argument("--order", choices=["asc", "desc"], default="desc",
                        help="asc = lower is better; desc = higher (default: desc)")
    parser.add_argument("--position", type=int, choices=[0, 1, 2], default=None,
                        help="Register only this position (default: all 3)")
    parser.add_argument("--symbol", choices=_cfg["symbols"], default=None,
                        help="Register only this symbol (default: all configured symbols)")
    parser.add_argument("--promote", action="store_true",
                        help="Transition the new version to Production")
    parser.add_argument("--prefer-rf", action="store_true", default=False,
                        help="Restrict winner search to tags.model_type='rf'. "
                             "Off by default — XGB now logs through XGBStringClassifier "
                             "in mlflow_grid.py, so its predictions return strings and "
                             "won't break backtest.evaluate. Use this flag if you need "
                             "to force RF for some reason.")
    parser.add_argument("--no-prefer-rf", dest="prefer_rf", action="store_false")
    args = parser.parse_args()

    tracking_uri = (os.environ.get("MLFLOW_TRACKING_URI")
                    or f"file:{(ROOT / 'mlruns').as_posix()}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"ERROR: experiment {args.experiment!r} not found at {tracking_uri}",
              file=sys.stderr)
        sys.exit(1)
    print(f"experiment: {args.experiment}  (id={exp.experiment_id})")

    positions = [args.position] if args.position is not None else [0, 1, 2]
    symbols   = [args.symbol]   if args.symbol   is not None else _cfg["symbols"]

    for sym in symbols:
        for pos in positions:
            filter_string = f"tags.position = '{pos}' and tags.symbol = '{sym}'"
            if args.prefer_rf:
                filter_string += " and tags.model_type = 'rf'"
            runs = mlflow.search_runs(
                [exp.experiment_id],
                filter_string=filter_string,
                order_by=[f"metrics.{args.metric} {args.order.upper()}"],
                max_results=1,
            )
            if runs.empty:
                print(f"sym={sym} pos={pos}: no runs matching {filter_string!r} — skipped")
                continue

            row     = runs.iloc[0]
            run_id  = row["run_id"]
            mvalue  = row.get(f"metrics.{args.metric}", float("nan"))
            model   = row.get("tags.model_type", "?")
            params  = {
                "n_estimators":  row.get("params.n_estimators"),
                "max_depth":     row.get("params.max_depth"),
                "learning_rate": row.get("params.learning_rate"),
            }
            print(f"\nsym={sym} pos={pos}: winner model={model}  "
                  f"{args.metric}={mvalue:.4f}  params={params}  run_id={run_id}")

            for head in HEADS:
                name = f"reversion-{head}-pos{pos}-{sym}"
                try:
                    client.create_registered_model(name)
                    print(f"  created registry entry: {name}")
                except MlflowException as e:
                    if "already exists" not in str(e).lower():
                        raise
                    # name already exists — fine

                src = f"runs:/{run_id}/model_{head}"
                mv = client.create_model_version(name=name, source=src, run_id=run_id)
                print(f"  registered {name} v{mv.version}  <- {src}")

                if args.promote:
                    client.transition_model_version_stage(
                        name=name, version=mv.version, stage="Production",
                        archive_existing_versions=True,
                    )
                    print(f"  promoted  {name} v{mv.version} -> Production")

    print("\ndone.")


if __name__ == "__main__":
    main()
