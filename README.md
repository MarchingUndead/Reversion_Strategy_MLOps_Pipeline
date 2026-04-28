# Reversion Strategy — MLOps Pipeline

Detect statistical outliers in the futures–cash basis on Indian equity futures,
classify whether they revert, diverge, or continue, and backtest an arbitrage
book that bets on the predicted outcome.

## Quick start

```bash
cp .env.example .env             # then edit SMTP_USER, SMTP_APP_PASSWORD, ALERT_TO, MODEL_URI
docker compose up -d             # one image (reversion:local) + monitoring stack
```

Open:
- Airflow UI:    http://localhost:8080  (`airflow` / `airflow`)
- Streamlit:     http://localhost:8501
- MLflow UI:     http://localhost:5000
- Grafana:       http://localhost:3000  (`admin` / `admin`)
- Prometheus:    http://localhost:9090
- Alertmanager:  http://localhost:9093

Trigger the full pipeline from Airflow → `reversion_pipeline`. Trigger one
MLflow run from Airflow → `mlflow_train_single` → *Trigger DAG w/ config*.
Smoke tests run every 30 min via the `smoke_tests` DAG.

## Documentation

- **Design + user manual**: [report/report.pdf](report/report.pdf) (build with `make report`)
  &mdash; merges the previous HLD, LLD, architecture, Airflow DAG, and user manual.
- **Open issues**: [report/ISSUES.md](report/ISSUES.md)
- **Test plan**: [report/test_plan.md](report/test_plan.md)
- **Notebook → module traceability**: [report/archive/ntbk2code.md](report/archive/ntbk2code.md)

## Make targets

```bash
make stack         # docker compose up -d
make report        # build report/report.pdf
make test          # pytest -q
make smoke         # health / ready / predict / prom-up
make purge         # DESTRUCTIVE: down -v + rm -rf mlruns/*
```
