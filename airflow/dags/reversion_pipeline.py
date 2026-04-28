"""Airflow DAG that mirrors the DVC pipeline.

This DAG wraps the existing `python src/*.py` entry points; it does NOT
re-implement any pipeline logic (preserve-core-logic rule). Each task is a
thin BashOperator that shells out to the same script `dvc repro` would run,
plus the A6-rubric primitives the DVC graph cannot express:

  - FileSensor    -> waits for raw data under data/raw/dates/ before any work
  - Pool          -> caps grid-search concurrency on the heaviest stage
  - Dynamic task mapping -> one mapped task per (model, position, hyper) combo
  - EmailOperator -> sends a "pipeline complete" email via Airflow's SMTP

Setup (run once, outside this file):

    airflow pools set grid_search 2 "Cap concurrency for grid_search.sh"
    airflow connections add smtp_default \
        --conn-type smtp --conn-host smtp.gmail.com --conn-port 587 \
        --conn-login "$SMTP_USER" --conn-password "$SMTP_APP_PASSWORD"

Render to PNG (also outside this file):

    airflow dags show reversion_pipeline --save report/airflow_dag.png

The static Mermaid version of this diagram is committed at
`report/airflow_dag.md` so reviewers without an Airflow install can still see
the structure.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.sensors.filesystem import FileSensor

# Project root: env var wins (compose stack sets PROJECT_ROOT=/opt/airflow/project),
# otherwise fall back to walking up from this file's location for local Airflow runs.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

ALERT_TO = os.environ.get("AIRFLOW_ALERT_EMAIL", "you@example.com")

DEFAULT_ARGS = {
    "owner":            "reversion",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email_on_retry":   False,
    "email":            [ALERT_TO],
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

# Grid sweep is defined once in config.yaml::grid (single source of truth).
# Edit that block to scale the sweep up/down; the DAG and scripts/grid_search.sh
# both rebuild from the same lists. Heavy ML imports stay out of DAG-parse time.
_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
_POSITIONS = _cfg["backtest"]["positions"]


def _expand(model_type: str, spec: dict) -> list[dict]:
    return [
        {"model": model_type, "position": p,
         "n_estimators": n, "max_depth": d, "learning_rate": lr}
        for p in _POSITIONS
        for n in spec["n_estimators"]
        for d in spec["max_depth"]
        for lr in spec["learning_rate"]
    ]


GRID_COMBOS = _expand("rf", _cfg["grid"]["rf"]) + _expand("xgb", _cfg["grid"]["xgb"])


def _processed_is_populated() -> bool:
    """True if data/processed/ already has at least one CSV per known symbol.

    Used by ShortCircuitOperator to skip preprocess when output exists. Honours
    the user's 'run preprocess only if processed data missing' rule without
    editing src/preprocess.py (preserve-core-logic).
    """
    proc = PROJECT_ROOT / "data" / "processed"
    if not proc.is_dir():
        return False
    for sym in _cfg["symbols"]:
        if not any(proc.glob(f"{sym}_*.csv")):
            return False
    return True


def _events_is_populated() -> bool:
    """True if data/events/<symbol>.csv exists for every configured symbol."""
    evt = PROJECT_ROOT / "data" / "events"
    if not evt.is_dir():
        return False
    return all((evt / f"{sym}.csv").is_file() for sym in _cfg["symbols"])


with DAG(
    dag_id="reversion_pipeline",
    description="Reversion-strategy MLOps pipeline: preprocess -> events -> grid-train -> register -> backtest -> email.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["reversion", "mlops", "a6"],
) as dag:

    wait_for_raw_data = FileSensor(
        task_id="wait_for_raw_data",
        filepath=str(PROJECT_ROOT / "data" / "raw" / "dates" / "trading_calendar.csv"),
        poke_interval=10,
        timeout=120,
        mode="reschedule",
    )

    # Skip both preprocess and events when processed+events already exist.
    # ignore_downstream_trigger_rules=False means the skip only propagates to
    # the immediate downstream task (preprocess); downstream-of-downstream
    # (events, grid_search, ...) respect their own trigger_rule. We set
    # trigger_rule="none_failed" on grid_search so it runs whether preprocess
    # ran or was skipped.
    check_data = ShortCircuitOperator(
        task_id="check_data",
        python_callable=lambda: not (_processed_is_populated() and _events_is_populated()),
        ignore_downstream_trigger_rules=False,
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_ROOT} && python src/preprocess.py",
    )

    events = BashOperator(
        task_id="events",
        bash_command=f"cd {PROJECT_ROOT} && python src/events.py",
    )

    @task(pool="grid_search", trigger_rule="none_failed")
    def run_grid_combo(combo: dict) -> str:
        """One mapped task per grid combination; pool caps concurrency.

        Passes --experiment reversion-grid explicitly so the MLFLOW_EXPERIMENT_NAME
        env var (which serve.py / mlflow_train.py read for their own purposes) can't
        silently divert grid runs into the wrong experiment.
        """
        import subprocess
        cmd = [
            "python", "src/mlflow_grid.py",
            "--model",         combo["model"],
            "--position",      str(combo["position"]),
            "--n-estimators",  str(combo["n_estimators"]),
            "--max-depth",     str(combo["max_depth"]),
            "--learning-rate", str(combo["learning_rate"]),
            "--experiment",    "reversion-grid",
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        return f"{combo['model']}-pos{combo['position']}-n{combo['n_estimators']}-d{combo['max_depth']}"

    grid_search = run_grid_combo.expand(combo=GRID_COMBOS)

    register_best = BashOperator(
        task_id="register_best",
        bash_command=f"cd {PROJECT_ROOT} && python scripts/register_best.py --promote",
    )

    backtest_val = BashOperator(
        task_id="backtest_val",
        bash_command=f"cd {PROJECT_ROOT} && python src/backtest.py --split val",
    )

    notify_complete = EmailOperator(
        task_id="notify_complete",
        to=ALERT_TO,
        subject="[reversion] pipeline run {{ ds }} complete",
        html_content=(
            "<p>The reversion pipeline finished at <b>{{ ts }}</b>.</p>"
            "<p>DAG run id: {{ run_id }}</p>"
            "<p>See MLflow UI for the new runs and Grafana for serving health.</p>"
        ),
        # all_done so an unconfigured SMTP doesn't poison the DAG. Failures
        # surface in the task log instead of failing the whole run.
        trigger_rule="all_done",
    )

    wait_for_raw_data >> check_data >> preprocess >> events >> grid_search >> register_best >> backtest_val >> notify_complete
