# Reversion Strategy — MLOps Pipeline

Detect statistical outliers in the futures–cash basis on Indian equity futures
and classify whether they revert, diverge, or continue. Produces a labeled
events table from raw exchange ticks, which a model layer consumes to generate
signals and backtest PnL.

Three notebooks (`src/distributions.ipynb`, `src/events.ipynb`, `src/model.ipynb`)
have been flattened into five library modules. `config.yaml` is the single
source of truth for every parameter. See [`ntbk2code.md`](ntbk2code.md) for the
function-by-function mapping.

The authoritative specs also live at the repo root:

- [`strategy_plan_v4.md`](strategy_plan_v4.md) — v4 clean-slate spec
- [`strategy_plan_phase_2.md`](strategy_plan_phase_2.md) — Phase 2 plan
- [`CLAUDE.md`](CLAUDE.md) — working rules and invariants

---

## Layout

```
Reversion_Strategy_MLOps_Pipeline/
├── config.yaml                <- single source of truth
├── ntbk2code.md               <- notebook → .py mapping
└── src/
    ├── preprocess.py          <- raw ticks → per-(dte,bucket) CSVs + expanding dist stats
    ├── events.py              <- processed → outlier/reversion events per symbol
    ├── plots.py               <- spread overlay plot + equity curve
    ├── model.py               <- load events, engineer features, train + evaluate
    ├── backtest.py            <- take reversion-predicted trades, report Sharpe / win rate
    └── *.ipynb                <- exploratory notebooks (kept)
```

Stages read and write only `data/raw/` (read-only) and `data/processed/`, so
each is independently runnable once the previous stage's outputs exist.

---

## Install

```
pip install -r requirements.txt
```

Requires Python 3.10+. All dependencies (pandas, numpy, scikit-learn, pyyaml,
matplotlib, mlflow, prometheus-client, streamlit, xgboost, pyarrow, dvc) are
listed in `requirements.txt`.

---

## Run the full pipeline

```
# 1. Raw ticks -> per-(symbol, dte, bucket) CSVs under data/processed/,
#    each carrying the streaming expanding (dist_mean, dist_std, dist_count).
python src/preprocess.py

# 2. Scan every (symbol, contract, session) for outlier runs; persist
#    per-symbol events to data/processed/events/{symbol}.csv
python src/events.py

# 3. Load events, engineer features, time-split, train a classifier + two
#    regressors per contract position (near / mid / far), print metrics.
python src/model.py

# 4. Re-run the training pipeline, take trades when the classifier predicts
#    reversion, print per-position Sharpe / win_rate and render the equity
#    curve.
python src/backtest.py
```

Every tunable the pipeline touches — thresholds, window sizes, symbol list,
train/test year split, RF hyperparams, schemas, paths — lives in `config.yaml`.
Flip a value there and rerun the relevant stage; nothing in the `.py` files
needs editing.

### Example: single-day plot

The `run()` helper in `src/events.py` renders the spread + cash overlay with
event markers for one session:

```python
from events import run
run("BAJFINANCE", 2024, "AUG", day=2)   # day-of-month OR full YYYYMMDD
```

---

## What was done for MLOps integration readiness

The user asked how each piece was set up for Docker / MLflow / Prometheus /
Grafana / Streamlit. Short answer: the refactor deliberately did NOT wire any
of them in — it prepared the code so each can be added in isolation without
touching core logic.

### Docker

- Every `.py` module is a runnable stage via `if __name__ == "__main__":`.
  That gives a clean 1:1 mapping to a container entrypoint:
  `CMD ["python", "src/events.py"]`.
- All paths in `config.yaml` (`paths.raw_root`, `paths.processed`, …) are
  **relative** to the project root (`Path(__file__).resolve().parents[1]`).
  A Docker volume mounted at `/app` keeps all paths valid with no code change.
- `requirements.txt` is flat and frozen; the image build is one `pip install -r`.
- Stage dependencies are filesystem-only (each reads the previous stage's
  output from disk), so multi-container pipelines via `docker compose` work
  out of the box — no shared in-memory state, no implicit ordering beyond
  "preprocess before events before model before backtest."
- `data/raw/` was marked read-only at the filesystem level (see commit history)
  so a misbehaving container can't corrupt the source data.

### MLflow

- `train_position()` and `evaluate()` are pure functions that return plain
  Python objects (tuples of sklearn estimators / augmented DataFrames). A later
  PR can wrap the `__main__` in `with mlflow.start_run():` and call
  `mlflow.log_params(_cfg["model"])`, `mlflow.log_metric("mae_duration", ...)`,
  `mlflow.sklearn.log_model(clf, ...)` without touching the function bodies.
- `_cfg["model"]` is a single flat dict with the exact hyperparameters the
  model uses — ideal for `mlflow.log_params(_cfg["model"])` as a one-liner.
- `mlflow` is already declared in `requirements.txt`; no dep change needed.

### Prometheus

- `track()` returns a list of event dicts and `backtest()` returns a trades
  DataFrame — both are observation points. Metrics (event rate, sharpe,
  win-rate, duration histograms) can be exported from the `__main__` block
  using `prometheus_client.Counter` / `Histogram` / `Gauge` without modifying
  the core functions.
- `prometheus-client` is in `requirements.txt` for when the pipeline shifts
  from batch to a long-running service; at that point each module gets a
  `start_http_server(port)` call in its `__main__`.

### Grafana

- Grafana is a downstream consumer. Two data paths are ready for it:
  - The Prometheus endpoints described above.
  - The persisted `data/processed/events/*.csv` files — can be ingested via
    the CSV or (later) Parquet data source plugin for a "live events"
    dashboard without the pipeline being aware of Grafana at all.

### Streamlit

- `streamlit` is declared in `requirements.txt`.
- `plots.plot()` and `plots.plot_equity_curve()` both return `(fig, ax)` so
  they drop straight into `st.pyplot(fig)` without refactor.
- `events.run()` and `backtest.backtest()` return DataFrames suitable for
  `st.dataframe(...)` or `st.metric(...)`.
- A thin `streamlit_app.py` can wire these together later:
  ```python
  import streamlit as st
  from events import run
  sym  = st.selectbox("symbol", _cfg["symbols"])
  year = st.number_input("year", 2022, 2025)
  run(sym, year, st.selectbox("month", _cfg["months"]))
  ```
- No code change in the library modules is required.

### DVC

- `requirements.txt` includes `dvc`. Raw data CSVs under `data/raw/*.csv` were
  already marked read-only; `.dvc` metadata files are untouched and still
  track them.
- `data/processed/` is the natural DVC output stage: `dvc stage add -n preprocess
  -d src/preprocess.py -d data/raw -o data/processed` (not executed here,
  but the boundaries match).

---

## `config.yaml` as single source of truth

Every `.py` module does the same three lines at the top:

```python
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load(open(ROOT / "config.yaml"))
```

Sections:
- `paths.*` — all filesystem locations
- `symbols`, `months`, `train_years`, `buckets` — discrete label sets
- `schemas.{fut,eq,vix}` — header-less CSV column names
- `preprocess.{vix_thresh, warmup_months, vix_file}`
- `events.{out_thresh, rev_thresh, min_ticks}` — detection / resolution rules
- `model.{feature_cols, train_year_cutoff, test_year, rf_*, random_state}`
- `backtest.{positions, position_labels}`

Hardcoded values in the notebooks were extracted to this file. The function
bodies themselves are unchanged.

---

## Invariants

See [`CLAUDE.md`](CLAUDE.md) for the full rules. The critical ones:

- **I1** — no future data in distributions (snapshot before update, per session)
- **I2** — causal features (no peeking past `det_idx`)
- **I3** — forward scan is preprocessing-only
- **I4** — events schema is a contract; update it first
- **I5** — models consumed only through the `ReversionModel` ABC
- **I6** — all config lives in `config.yaml`; libraries take slices

---

## Notebooks

The three notebooks under `src/` are kept as-is — useful for interactive
exploration of single contracts, days, or distribution snapshots. They share
the same function names as the `.py` modules, so moving code between them
is trivial once you've validated an idea.
