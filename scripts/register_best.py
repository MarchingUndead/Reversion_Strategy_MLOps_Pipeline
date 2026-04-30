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
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# MLflow's file-store registry writes meta.yaml updates via tempfile +
# shutil.move. Inside the airflow-scheduler container, the rename always
# crosses filesystems (Docker Desktop on Windows treats each bind mount as a
# separate device), so shutil.move falls back to copy2 → copystat → os.utime,
# and utime with explicit nanosecond timestamps fails with EPERM on the
# Windows-mounted FS. The file copy itself succeeds; only the timestamp
# preservation fails. MLflow's `last_updated_timestamp` lives inside the
# meta.yaml content, not on the FS mtime, so swallowing this EPERM is
# harmless. Patch is local to this script — applies for the duration of the
# register run only.
_orig_copystat = shutil.copystat
def _tolerant_copystat(*args, **kwargs):
    try:
        _orig_copystat(*args, **kwargs)
    except (OSError, PermissionError):
        pass
shutil.copystat = _tolerant_copystat

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

                # Symbol grouping tag on the registered model itself, so
                # `search_registered_models("tags.symbol = 'X'")` returns all 9
                # entries for that symbol. Position and head are deliberately
                # NOT tagged — they're already in the registered name
                # (`reversion-<head>-pos<pos>-<symbol>`) and parsing the name
                # avoids tag/name divergence.
                client.set_registered_model_tag(name=name, key="symbol", value=sym)

                src = f"runs:/{run_id}/model_{head}"
                mv = client.create_model_version(name=name, source=src, run_id=run_id)
                print(f"  registered {name} v{mv.version}  <- {src}")

                # Per-version provenance: hyperparameters + winning metric land on
                # the version row in the UI so reviewers don't need to click into
                # the source run to see what was registered.
                version_tags = {
                    "run_id":        str(run_id),
                    "model_type":    str(model),
                    "n_estimators":  str(params.get("n_estimators")),
                    "max_depth":     str(params.get("max_depth")),
                    "learning_rate": str(params.get("learning_rate")),
                    args.metric:     f"{mvalue:.4f}",
                }
                for k, v in version_tags.items():
                    client.set_model_version_tag(
                        name=name, version=mv.version, key=k, value=v,
                    )

                if args.promote:
                    client.transition_model_version_stage(
                        name=name, version=mv.version, stage="Production",
                        archive_existing_versions=True,
                    )
                    print(f"  promoted  {name} v{mv.version} -> Production")

    print("\ndone.")


if __name__ == "__main__":
    main()
