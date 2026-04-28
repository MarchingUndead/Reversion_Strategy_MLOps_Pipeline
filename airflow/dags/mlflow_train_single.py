"""Manual one-shot MLflow training run, triggerable from the Airflow UI.

This DAG is the user-facing answer to "let me kick off an MLflow run from a UI."
It exposes mlflow_grid.py's CLI as Airflow params, so an operator clicks
"Trigger DAG w/ config", fills the form, and gets one MLflow run logged under
the chosen experiment.

Reuses src/mlflow_grid.py verbatim — preserve-core-logic.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import yaml
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())


with DAG(
    dag_id="mlflow_train_single",
    description="Manual one-shot MLflow training run (rf or xgb, single (symbol, position)).",
    schedule=None,
    catchup=False,
    start_date=datetime(2026, 1, 1),
    max_active_runs=4,
    tags=["reversion", "mlflow", "manual"],
    params={
        "model":         Param("rf",  type="string", enum=["rf", "xgb"]),
        "symbol":        Param(_cfg["symbols"][0], type="string", enum=_cfg["symbols"]),
        "position":      Param(0,     type="integer", enum=[0, 1, 2]),
        "n_estimators":  Param(200,   type="integer"),
        "max_depth":     Param(10,    type="integer"),
        "learning_rate": Param(0.1,   type="number"),
        "experiment":    Param("reversion-grid", type="string"),
    },
) as dag:

    BashOperator(
        task_id="train",
        bash_command=(
            f"cd {PROJECT_ROOT} && python src/mlflow_grid.py "
            "--model {{ params.model }} "
            "--symbol {{ params.symbol }} "
            "--position {{ params.position }} "
            "--n-estimators {{ params.n_estimators }} "
            "--max-depth {{ params.max_depth }} "
            "--learning-rate {{ params.learning_rate }} "
            "--experiment {{ params.experiment }}"
        ),
    )
