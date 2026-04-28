# Reversion Strategy — MLOps Pipeline

Detect statistical outliers in the futures–cash basis on Indian equity futures, classify whether each outlier reverts, diverges, or continues to EOD, and backtest an arbitrage book that trades on the predicted outcome.

The repo is the full pipeline plus its MLOps surface: ingestion, EDA-grade conditional-distribution stats, event detection, per-(symbol, position) classifier + duration + revert-delta heads, MLflow tracking and Registry, FastAPI serving, Streamlit UIs, Airflow orchestration, and Prometheus + Grafana + Alertmanager monitoring. Eleven Compose services. One image.

---

## Table of contents

1. [What's in the box](#whats-in-the-box)
2. [Prerequisites](#prerequisites)
3. [First-time setup](#first-time-setup)
4. [Bring the stack up](#bring-the-stack-up)
5. [Running the pipeline end-to-end](#running-the-pipeline-end-to-end)
6. [Running one MLflow run from the UI](#running-one-mlflow-run-from-the-ui)
7. [Inspecting results](#inspecting-results)
8. [Promoting / swapping the served model](#promoting--swapping-the-served-model)
9. [Smoke tests, load, alert-firing](#smoke-tests-load-alert-firing)
10. [Tests](#tests)
11. [Configuration](#configuration)
12. [Repository layout](#repository-layout)
13. [Hidden 2025 holdout discipline](#hidden-2025-holdout-discipline)
14. [Building the report PDF](#building-the-report-pdf)
15. [Troubleshooting](#troubleshooting)
16. [Documentation index](#documentation-index)

---

## What's in the box

| Layer | Tool | Where |
|---|---|---|
| Build / packaging | Docker (one image: `reversion:local`) | `Dockerfile`, `docker-compose.yml` |
| Data lineage | DVC (content-hashed stages) | `dvc.yaml`, `dvc.lock` |
| Orchestration | Airflow 2.10.3 (LocalExecutor + Postgres) | `airflow/dags/` |
| Experiment tracking | MLflow (file backend) | `mlruns/`, `MLproject` |
| Model registry | MLflow Registry (`Production` / `Archived` / `None` stages) | served via `auto:` URI |
| Serving | FastAPI / `mlflow.pyfunc` on port 5002 | `src/serve.py`, `src/serve_metrics.py` |
| UI | Streamlit on port 8501 | `src/streamlit_app.py` |
| Metrics | Prometheus on port 9090, scrape every 15 s | `monitoring/prometheus.yml` |
| Alerts | 17 rules across `infra` / `serving` / `system` groups | `monitoring/alerts/` |
| Email routing | Alertmanager on port 9093 | `monitoring/alertmanager.template.yml` |
| Dashboards | Grafana on port 3000 (3 dashboards pre-provisioned) | `monitoring/grafana/dashboards/` |

The eleven Compose services: `streamlit`, `mlflow-ui`, `serving`, `airflow-init`, `airflow-webserver`, `airflow-scheduler`, `postgres`, `prometheus`, `alertmanager`, `grafana`, `node-exporter`. Six share `reversion:local`; five are upstream images.

---

## Prerequisites

- **Docker Desktop** (Windows / macOS) or **Docker Engine + Compose plugin** (Linux). Tested on Docker Desktop 4.x.
- **Disk**: ~6 GB for images + `mlruns/`. Raw data adds another ~5 GB if you have it.
- **RAM**: 8 GB minimum, 16 GB comfortable. Pipeline peaks at ~6 GB during preprocess.
- **Ports free on host**: 3000, 5000, 5002, 8080, 8501, 9090, 9093, 9100. If any is taken, change the corresponding `*_PORT` in `.env`.
- **Gmail account with App Password** if you want alert / DAG-failure emails. Generate at <https://myaccount.google.com/apppasswords> (requires 2-Step Verification).
- **Optional, for running scripts on the host**: Python 3.10 + a virtualenv with `pip install -r requirements.txt`.

---

## First-time setup

### 1. Clone

```bash
git clone <repo-url> Reversion_Strategy_MLOps_Pipeline
cd Reversion_Strategy_MLOps_Pipeline
```

### 2. Create your `.env`

```bash
cp .env.example .env
```

Open `.env` and fill in at least the SMTP block if you want email alerts:

```ini
SMTP_USER=you@gmail.com
SMTP_APP_PASSWORD=<16-char app password, no spaces>
ALERT_TO=you@gmail.com
```

The `MODEL_URI` line ships pointing at `auto:reversion-classifier-pos0` — that resolves to whatever the registry's latest Production version is. If your first stack boot has nothing registered yet, `serving` will exit and restart-loop until you run the pipeline once and `register_best.py` populates the registry. That's expected.

`AIRFLOW_SECRET_KEY` ships as a placeholder. Replace it with any 32+ char string of your choosing — it's used by the Airflow webserver / scheduler internal request signing; both containers must agree.

### 3. Place raw data

The pipeline reads from `data/raw/` with this layout:

```
data/raw/
  2022-24/
    BAJFINANCE.csv                       # cash equity ticks (single file per symbol)
    BAJFINANCE22JANFUT.csv               # one file per (symbol, year, month) futures contract
    BAJFINANCE22FEBFUT.csv
    ...
    RELIANCE.csv
    RELIANCE22JANFUT.csv
    ...
    INDIAVIX.csv                         # India VIX ticks for the whole window
  2025/                                  # 2025 ticks; HIDDEN HOLDOUT — see below
    BAJFINANCE.csv
    BAJFINANCE25JANFUT.csv
    ...
  dates/
    trading_calendar.csv
    expiry_dates.csv
```

The repo tracks data via DVC (`data/raw/2022-24.dvc`, `data/raw/2025.dvc`, `data/raw/dates.dvc`). If you have access to the DVC remote, run `dvc pull`. Otherwise drop the CSVs into the paths above by hand. The `wait_for_raw_data` FileSensor in `reversion_pipeline` blocks on `data/raw/dates/trading_calendar.csv`; that file existing is the trigger.

### 4. Build and start

```bash
docker compose up -d --build
```

`--build` is needed only the first time (or after `requirements.txt` / `Dockerfile` changes); subsequent `docker compose up -d` is fine.

The `airflow-init` service is idempotent: it migrates the metadata DB, creates the `airflow / airflow` admin user, the `grid_search` Airflow pool (cap 2), and the `smtp_default` connection from your `.env` SMTP vars.

---

## Bring the stack up

After first-time setup, day-to-day:

```bash
docker compose up -d        # bring up
docker compose stop         # pause (preserves containers)
docker compose down         # tear down (preserves volumes — Airflow DB, mlruns, grafana state)
docker compose down -v      # DESTRUCTIVE: also drops named volumes
docker compose ps           # what's running
docker compose logs -f serving       # tail one service
```

UIs, with default credentials:

| Service | URL | Login |
|---|---|---|
| Airflow | <http://localhost:8080> | `airflow` / `airflow` |
| Streamlit | <http://localhost:8501> | none |
| MLflow | <http://localhost:5000> | none |
| Grafana | <http://localhost:3000> | `admin` / `admin` (skip the password-change prompt) |
| Prometheus | <http://localhost:9090> | none |
| Alertmanager | <http://localhost:9093> | none |
| Serving (API) | <http://localhost:5002/health> | none |
| node-exporter | <http://localhost:9100/metrics> | none |

To verify everything is wired correctly before you trigger anything:

```bash
curl http://localhost:5002/health        # expect {"status":"ok",...}
curl http://localhost:5002/ready         # 503 until model loads, then 200
curl http://localhost:9090/-/ready       # Prometheus ready
docker compose ps                        # every service "Up" or "Up (healthy)"
```

If `serving` is restart-looping on a fresh stack with an empty registry, that's expected — see [§ Promoting / swapping the served model](#promoting--swapping-the-served-model). The fix is to run the pipeline once.

---

## Running the pipeline end-to-end

Once raw data is in place and the stack is up:

1. Open <http://localhost:8080> and log in.
2. In the DAGs list, find **`reversion_pipeline`**. If it's paused (gray toggle), click the toggle to un-pause.
3. Click the play button on the right → **Trigger DAG**.
4. Watch the Graph view. Expected sequence:
   - `wait_for_raw_data` (FileSensor on `data/raw/dates/trading_calendar.csv`; 2-min timeout).
   - `check_data` (ShortCircuitOperator; if `data/processed/` and `data/events/` are already populated, the next two tasks skip).
   - `preprocess` → `events`.
   - `grid_search` (dynamically mapped: one task per (symbol × position × model_type × hp); `pool=grid_search` caps concurrency at 2).
   - `register_best` (`python scripts/register_best.py --promote`).
   - `backtest_val` (`python src/backtest.py --split val`).
   - `notify_complete` (EmailOperator; `trigger_rule=all_done` so SMTP problems don't poison the run).
5. On a cold dev box this takes ~15 minutes the first time. Re-triggering the same data takes ~4 minutes (the `check_data` short-circuit skips preprocess + events).

Failures email `ALERT_TO` via the `smtp_default` connection.

The DAG also runs automatically on `@weekly` schedule (Sundays at 00:00 UTC) once un-paused. Pause it again (toggle to gray) if you don't want the schedule.

---

## Running one MLflow run from the UI

For ad-hoc experimentation without re-running the full grid:

1. Airflow UI → DAGs → **`mlflow_train_single`**.
2. Un-pause if needed.
3. Click the play button → **Trigger DAG w/ config**.
4. Fill the form:
   - `model`: `rf` or `xgb`
   - `symbol`: dropdown populated from `config.yaml::symbols` (`BAJFINANCE`, `RELIANCE`)
   - `position`: 0, 1, or 2 (near, mid, far month)
   - `n_estimators`, `max_depth`, `learning_rate`: hyperparameters (learning_rate is ignored by RF)
   - `experiment`: defaults to `reversion-grid`; pick a different name to keep ad-hoc runs separate
5. Submit. The new run appears in MLflow at <http://localhost:5000> under the chosen experiment.

This DAG wraps `python src/mlflow_grid.py` with one combo, so it logs the exact same artefacts the full grid does (3 head models, sample predictions, synthetic example, all params + tags + metrics).

---

## Inspecting results

### Streamlit (<http://localhost:8501>)

Three tabs:

**Tab 1 — Event explorer + Registered models**

- The expander at the top lists every name in the MLflow Registry and the version currently in each stage. It also shows the live `MODEL_URI` (queried from `serving:5002/health`).
- *Pin this version*: writes `MODEL_URI=models:/<name>/<version>` to `.env` and runs `docker compose up -d --force-recreate serving`.
- *Promote to Production*: calls `client.transition_model_version_stage(stage="Production", archive_existing_versions=True)`.
- The lower part of the tab runs `events.run(symbol, year, month, day)` and shows the detection events for that day. Year is capped at 2024 — 2025 is the hidden holdout.

**Tab 2 — Backtest viewer**

- Multi-select trade-log files from `data/backtest_logs/`.
- See per-file PnL summary (n_trades, win_rate, gross / net), the cumulative-equity line plot, and the raw trades table.

**Tab 3 — Direct prediction**

- Pick (symbol, session date, contract expiry month). Position derives automatically as `(expiry_month - session_month) mod 12` and is clamped to `{0, 1, 2}`.
- Pick a head: `classifier`, `duration`, or `revert`.
- Fill the 12 features. The resolved registered model name (`reversion-{head}-pos{N}-{symbol}`) is shown above the form.
- Click **Predict (direct load)**. The model is loaded in-process via `mlflow.sklearn.load_model("runs:/...")` — the FastAPI server is not on this path. For classifiers the response also includes per-class `predict_proba` bars.

### MLflow UI (<http://localhost:5000>)

- **Experiments** → `reversion-grid` shows every grid run with its params, metrics (`f1_macro`, `accuracy`, `mae_duration`, `mae_revert`, `rmse_duration`, `rmse_revert`, `n_test`), and tags (`symbol`, `position`, `model_type`).
- **Models** lists every registered name (`reversion-classifier-pos0-BAJFINANCE`, etc.), versions, and current_stage.
- Click any run → **Artifacts** to see `model_classifier/`, `model_duration/`, `model_revert/`, and `predictions/sample_predictions.csv` + `predictions/synthetic_example.json`.

### Grafana (<http://localhost:3000>)

Three pre-provisioned dashboards under **Dashboards** → **Browse**:

- **Reversion — Overview**: service-health stat panels at top, throughput + latency in the middle, host CPU / memory / disk below.
- **Model Serving**: request rate, latency quantiles (p50 / p95 / p99), error rate, status-code split, 4xx ratio, payload-rows, prediction-score quantiles + heatmap, latency-vs-1h-baseline, and the model-provenance table (`model_uri`, `git_commit`, `mlflow_run_id`).
- **System Metrics**: CPU%, memory%, rootfs disk%, network I/O via `node-exporter`.

### Prometheus (<http://localhost:9090>)

- **Alerts** tab → all 17 rules grouped by `infra` / `serving` / `system`. Inactive rules are gray; pending = orange; firing = red.
- **Targets** tab → `up{job="model-server"}` and `up{job="node-exporter"}` should both be `UP`.

### Alertmanager (<http://localhost:9093>)

- Lists currently-firing alerts and any active silences.
- The `ModelServerDown` rule is configured as an inhibition source: when it fires, all `area=serving` alerts are silenced to keep the inbox clean during a known outage.

---

## Promoting / swapping the served model

The serving container loads exactly one classifier head per process; switching models means writing a new `MODEL_URI` to `.env` and force-recreating the container.

### Two front-ends, one backend

Both surfaces below shell out to `scripts/swap_model.py`. CLI:

```bash
# list every registered name + versions + stages
python scripts/swap_model.py --list

# pin a specific version
python scripts/swap_model.py \
  --name reversion-classifier-pos0-BAJFINANCE \
  --version 5

# pin the Production-stage version (re-resolves on every container start)
python scripts/swap_model.py \
  --name reversion-classifier-pos0-BAJFINANCE \
  --stage Production

# write auto:<name> — resolver in src/serve.py picks Production at startup
python scripts/swap_model.py \
  --name reversion-classifier-pos0-BAJFINANCE \
  --auto

# stage the .env change without bouncing the container
python scripts/swap_model.py \
  --name reversion-classifier-pos0-BAJFINANCE \
  --version 5 \
  --no-restart
```

Each invocation (without `--no-restart`) ends with `docker compose up -d --force-recreate serving`.

GUI: Streamlit tab 1 → **Registered models** expander → pick a name → pick a version → click *Pin this version* or *Promote to Production*. The currently-served `MODEL_URI` is shown live in the same expander.

### Promote winners after a grid sweep

The Airflow `reversion_pipeline` does this for you (last-but-one task), but you can also promote by hand:

```bash
docker compose exec airflow-scheduler \
  python /opt/airflow/project/scripts/register_best.py --promote
```

Per (symbol, position) pair, this picks the run with the highest `f1_macro` from `tags.symbol=... and tags.position=...`, registers all three head artefacts under `reversion-{head}-pos{p}-{symbol}`, and transitions the new version to `Production` with `archive_existing_versions=True` (atomic cutover). Eighteen registered names total for the current 2-symbol setup.

---

## Smoke tests, load, alert-firing

`scripts/smoke.py` is one tool with several subcommands. The Airflow `smoke_tests` DAG runs the read-only ones every 30 minutes; the rest are manual.

```bash
# read-only checks (same ones the smoke_tests DAG runs)
python scripts/smoke.py health
python scripts/smoke.py ready
python scripts/smoke.py predict
python scripts/smoke.py prom-up

# load generation
python scripts/smoke.py load --mode ok        --rps 5  --duration 120
python scripts/smoke.py load --mode bad-json  --rps 2  --duration 60
python scripts/smoke.py load --mode bad-schema --rps 2  --duration 60
python scripts/smoke.py load --mode exception --rps 2  --duration 60
python scripts/smoke.py load --mode big       --rps 1  --duration 30
python scripts/smoke.py load --mode burst     --rps 50 --duration 10
python scripts/smoke.py load --mode mix       --rps 5  --duration 300

# verify SMTP works in isolation
python scripts/smoke.py email-test

# fire real alert rules end-to-end (waits the per-alert `for:` window, then checks)
python scripts/smoke.py fire-alerts --mode exceptions
python scripts/smoke.py fire-alerts --mode four-xx
python scripts/smoke.py fire-alerts --mode slow
python scripts/smoke.py fire-alerts --mode spike
python scripts/smoke.py fire-alerts --mode all
```

After `fire-alerts`: wait the per-alert `for:` window (1–5 min depending on rule), then check <http://localhost:9090/alerts>, <http://localhost:9093>, and your inbox.

To run the smoke checks against the in-Compose stack from your host (the URLs above default to `localhost`), the script auto-detects whether you're inside the airflow container (uses `serving:5002`) or on the host (uses `localhost:5002`).

---

## Tests

### Unit tests (pytest)

16 cases on the pure-function helpers in `src/preprocess.py`, `src/model.py`, `src/backtest.py`. No mocked databases — all fixtures are small in-memory CSV strings.

Run on the host:

```bash
pytest -v --tb=short
```

Or inside the airflow-scheduler container (matches CI environment exactly):

```bash
docker compose exec airflow-scheduler pytest -v --tb=short
```

`pytest.ini` sets `pythonpath=src`, `testpaths=tests`, and `python_files=test_*.py` so `tests/alerts_test.yml` is automatically excluded.

### Promtool alert-rule tests

17 cases covering every alert rule under `monitoring/alerts/`. Run via:

```bash
bash scripts/test_alerts.sh
```

This dockers up `prom/prometheus:v2.54.1` with `--entrypoint promtool`, mounts `monitoring/alerts` and `tests`, and runs `promtool check rules` then `promtool test rules tests/alerts_test.yml`.

### Live-stack smoke (continuous)

The `smoke_tests` Airflow DAG (`*/30 * * * *`) is the closest thing to a continuous-integration loop in this repo — it tests the running deployment, not the source. Email-on-failure to `ALERT_TO`.

---

## Configuration

### `config.yaml` (single source of truth)

Every `src/*.py` reads `config.yaml` via `yaml.safe_load` at import. The Airflow DAG re-reads it at parse time to expand the grid combos. Edit one file, every consumer picks it up.

Key sections you might tune:

| Section | What it does |
|---|---|
| `paths.*` | Where raw / processed / events / models live (relative to repo root) |
| `symbols` | List of tickers — adding one means the next pipeline run trains models for it (no code change) |
| `preprocess.vix_thresh` | High-VIX day filter (default 20) |
| `events.{out_thresh, rev_thresh, min_ticks}` | Event detector thresholds |
| `model.feature_cols` | The 12 features used by every head; changing this changes the API contract |
| `model.{train,val,dev_test,hidden}_range` | Inclusive YYYY-MM bounds for the four splits |
| `backtest.{lot_size, order_size, txn_cost_rate}` | Trade-economics knobs |
| `grid.{rf, xgb}` | Grid sweep cartesian — add hyperparam values to scale up |

### `.env` (secrets and ports)

| Var | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_PORT` | MLflow file backend + UI port |
| `MODEL_URI`, `SERVE_HOST`, `SERVE_PORT` | Serving model selector + bind config |
| `AIRFLOW_SECRET_KEY` | Webserver / scheduler internal request signing — must be the same in both containers |
| `STREAMLIT_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT`, `GRAFANA_ADMIN_USER`, `GRAFANA_ADMIN_PASSWORD`, `NODE_EXPORTER_PORT`, `ALERTMANAGER_PORT` | Service ports + Grafana credentials |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_APP_PASSWORD`, `ALERT_TO` | Gmail SMTP (Alertmanager + Airflow EmailOperator) |
| `AIRFLOW_UID` | Linux/WSL2: your host UID; Windows: leave blank or 50000 |

`.env` is gitignored; commit `.env.example` only.

---

## Repository layout

```
.
├── Dockerfile                 # one image: reversion:local
├── MLproject                  # alternative entrypoint: mlflow run .
├── docker-compose.yml         # 11 services
├── config.yaml                # SINGLE SOURCE OF TRUTH for paths, hyperparams, splits
├── pytest.ini                 # testpaths=tests, pythonpath=src
├── python_env.yaml            # for MLproject
├── requirements.txt
├── .env.example               # copy to .env, fill in SMTP / MODEL_URI
├── airflow/
│   └── dags/
│       ├── reversion_pipeline.py     # @weekly: preprocess → events → grid → register → backtest → email
│       ├── mlflow_train_single.py    # manual ad-hoc training
│       └── smoke_tests.py            # */30 * * * *: health + ready + predict + prom-up
├── src/
│   ├── preprocess.py          # raw ticks → per-(symbol, dte, bucket) shards
│   ├── events.py              # rolling z-score → detection + resolution events
│   ├── model.py               # split, clean, train_position, evaluate (preserve-core-logic)
│   ├── mlflow_train.py        # baseline single-config run
│   ├── mlflow_grid.py         # grid sweep CLI; canonical training entry
│   ├── mlflow_utils.py        # logging helpers
│   ├── backtest.py            # per-trade ledger; loads via runs:/ for RO-mount safety
│   ├── serve.py               # FastAPI; /invocations, /health, /ready, /metrics
│   ├── serve_metrics.py       # Prometheus instruments
│   ├── streamlit_app.py       # 3 tabs: events / backtest viewer / direct prediction
│   └── plots.py               # shared matplotlib helpers
├── scripts/
│   ├── register_best.py       # pick + register + promote per-(symbol, position) winners
│   ├── swap_model.py          # rewrite MODEL_URI in .env, force-recreate serving
│   ├── smoke.py               # health/ready/predict/prom-up + load + fire-alerts + email-test
│   ├── test_alerts.sh         # promtool check rules + test rules
│   └── build_report.sh        # latexmk wrapper for report/report.pdf
├── monitoring/
│   ├── prometheus.yml         # 15 s scrape, 2 targets
│   ├── alertmanager.template.yml   # severity-based routing, ModelServerDown inhibits
│   ├── alerts/
│   │   ├── infra.yml          # ModelServerDown, NodeExporterDown, NoTraffic
│   │   ├── serving.yml        # exceptions / 5xx / 4xx / latency / inflight (9 rules)
│   │   └── system.yml         # CPU / memory / disk / load (5 rules)
│   └── grafana/
│       ├── datasources/       # provisioned Prometheus datasource
│       └── dashboards/        # overview, model_serving, system_metrics (JSON)
├── tests/
│   ├── test_preprocess.py     # 4 cases
│   ├── test_model.py          # 9 cases
│   ├── test_backtest.py       # 3 cases
│   └── alerts_test.yml        # 17 promtool unit tests
├── data/
│   ├── raw/                   # DVC-tracked inputs (2022-24, 2025, dates)
│   ├── processed/             # DVC out: preprocess output, per-(symbol,dte,bucket) shards
│   ├── events/                # DVC out: detection + resolution events per symbol
│   └── backtest_logs/         # DVC out: trade ledgers, 6 files per split
├── mlruns/                    # MLflow file backend (DVC-ignored, gitignored)
├── dvc.yaml, dvc.lock         # data pipeline lineage
└── report/
    ├── report.tex             # consolidated design + user manual (compiled to PDF)
    ├── report.pdf             # build with: bash scripts/build_report.sh
    ├── ISSUES.md              # open issues + RESOLVED/DECISION change-log
    ├── test_plan.md           # full test strategy
    ├── images/                # screenshots referenced by report.tex
    └── archive/
        └── ntbk2code.md       # notebook → src/ traceability
```

---

## Hidden 2025 holdout discipline

`data/raw/2025/` is a one-shot final-evaluation set. Routine scripts cannot read it:

- `src/backtest.py --split hidden` exits with `--split hidden requires --confirm-holdout`.
- Streamlit tab 1's Event Explorer caps year selection at 2024.
- `scripts/smoke.py load` explicitly skips 2025 events files when sampling payloads.
- `dvc.yaml::backtest_hidden` is `frozen: true` — `dvc repro` won't run it.

To run the hidden-holdout backtest deliberately:

```bash
python src/backtest.py --split hidden --confirm-holdout
```

Use sparingly. Each run on the hidden split contaminates its statistical independence from training.

---

## Building the report PDF

Requires `latexmk` + `pdflatex`. On Windows install MiKTeX; on Debian/Ubuntu:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra latexmk
```

Then:

```bash
bash scripts/build_report.sh           # one-shot → report/report.pdf
bash scripts/build_report.sh --watch   # rebuild on save (latexmk -pvc)
```

The report consolidates the previous HLD, LLD, architecture, Airflow DAG, and user-manual markdown files into one PDF.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `serving` container restart-loops | `MODEL_URI` resolves to nothing (empty registry) | Trigger `reversion_pipeline` once so `register_best.py` populates the registry, or run it directly: `docker compose exec airflow-scheduler python /opt/airflow/project/scripts/register_best.py --promote` |
| `serving` keeps failing with `name=reversion-classifier-pos0 not found` | `.env::MODEL_URI` points at the pre-symbol naming | Rewrite to a per-symbol name: `python scripts/swap_model.py --name reversion-classifier-pos0-BAJFINANCE --auto` |
| DAG visible but trigger does nothing | DAG is paused, or `wait_for_raw_data` is blocking | Toggle the DAG on; verify `data/raw/dates/trading_calendar.csv` exists |
| `PermissionError` in scheduler logs | Stale ownership on the named `airflow_logs` volume | `docker compose down`; `docker volume rm <stack>_airflow_logs`; `docker compose up -d`. If chown didn't take, run a one-shot: `docker run --rm -v <stack>_airflow_logs:/logs --user 0:0 alpine sh -c "chown -R 50000:0 /logs"` |
| MLflow runs not visible in UI; `[Errno 13] Permission denied: '/C:'` in scheduler logs | Experiment was first created host-side on Windows so `artifact_location` in `meta.yaml` is `file:C:/...` which the Linux container can't parse | `docker compose down`; `Remove-Item -Recurse -Force mlruns` (or `rm -rf mlruns`); `docker compose up -d`; trigger the DAG so the container creates the experiment with a Linux artifact_location. Don't run `python src/mlflow_grid.py` on the host until the experiment is well-established |
| Two experiments share the same name | Race in `set_experiment` during boot — two processes both created `reversion-grid` | `docker compose down`; manually delete the duplicate id under `mlruns/<id>`; `docker compose up -d` |
| `[Errno 13] Permission denied: '/opt/airflow/project/...'` during MLflow logging | Training script tried to write a temp file under `ROOT/`, which is bind-mounted RO inside the airflow scheduler | Already fixed: `mlflow_grid.py` / `mlflow_train.py` use `mlflow.log_text` / `mlflow.log_dict`. If you reintroduce a temp-file pattern, write under `/tmp/` not `ROOT/` |
| Streamlit "Cannot reach `http://serving:5002/...`" | `serving` down or unhealthy | `docker compose logs serving`; fix `MODEL_URI` |
| Alert emails never arrive | `.env` was empty when `docker compose up` ran | `docker compose down`; fix `.env`; `docker compose up -d`. Then `python scripts/smoke.py email-test` to verify |
| Grafana dashboards empty | Prometheus can't reach `serving:5002` | `curl http://localhost:9090/api/v1/targets` and inspect status |
| Two `Production` versions appear briefly during register | `register_best.py --promote` is mid-cutover | Wait for it to finish — `archive_existing_versions=True` is atomic |

---

## Documentation index

- **Full design + user manual**: [report/report.pdf](report/report.pdf) (build with `bash scripts/build_report.sh`)
- **Open issues + change log**: [report/ISSUES.md](report/ISSUES.md)
- **Test plan (long form)**: [report/test_plan.md](report/test_plan.md)
- **Notebook → module traceability**: [report/archive/ntbk2code.md](report/archive/ntbk2code.md)

The report is the canonical reference for: rubric cross-walk, architectural decisions (`Section 3`), low-level API specifications (`Section 4`), CI/CD architecture (`Section 11` — native-stack design, no external CI), and the test report (Appendix B).
