"""Periodic smoke tests for the live serving stack.

Runs every 30 min. Each task shells out to scripts/smoke.py <subcommand>;
that script lives at /opt/airflow/project/scripts/ via the existing bind
mount. Failures email ALERT_TO via the Airflow SMTP path.

Subcommands exercised here are the read-only ones — no load generation, no
alert firing. Those stay manual via scripts/smoke.py {load,fire-alerts}.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ALERT_TO     = os.environ.get("AIRFLOW_ALERT_EMAIL", "you@example.com")

# Inside the airflow container, services are reachable by docker-compose
# service name on the default bridge network.
_PREFIX = (
    "export SMOKE_SERVING_URL=http://serving:5002 && "
    "export SMOKE_PROM_URL=http://prometheus:9090 && "
    "export SMOKE_ALERT_URL=http://alertmanager:9093 && "
    f"cd {PROJECT_ROOT} && "
)

DEFAULT_ARGS = {
    "owner":            "reversion",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email":            [ALERT_TO],
    "retries":          0,
    "retry_delay":      timedelta(seconds=30),
}


with DAG(
    dag_id="smoke_tests",
    description="Read-only smoke tests against the serving + monitoring stack, every 30 min.",
    default_args=DEFAULT_ARGS,
    schedule="*/30 * * * *",
    catchup=False,
    start_date=datetime(2026, 1, 1),
    max_active_runs=1,
    tags=["reversion", "smoke", "monitoring"],
) as dag:

    health  = BashOperator(task_id="health",
                           bash_command=_PREFIX + "python scripts/smoke.py health")
    ready   = BashOperator(task_id="ready",
                           bash_command=_PREFIX + "python scripts/smoke.py ready")
    predict = BashOperator(task_id="predict",
                           bash_command=_PREFIX + "python scripts/smoke.py predict")
    prom_up = BashOperator(task_id="prom_up",
                           bash_command=_PREFIX + "python scripts/smoke.py prom-up")

    [health, ready] >> predict
