# Reversion Strategy — MLOps Pipeline

Detect statistical outliers in the futures–cash basis on Indian equity futures,
classify whether they revert, diverge, or continue, and backtest an arbitrage
book that bets on the predicted outcome. Three exploratory notebooks
(`distributions`, `events`, `model` + a `backtest` sandbox) were flattened into
five library modules with `config.yaml` as the single source of truth.

See [`ntbk2code.md`](ntbk2code.md) for the function-by-function mapping from
notebook cells to `.py` modules.

## Layout

```
Reversion_Strategy_MLOps_Pipeline/
├── config.yaml                  <- single source of truth
├── ntbk2code.md                 <- notebook -> .py mapping
├── requirements.txt
└── src/
    ├── preprocess.py            <- raw ticks -> per-(dte, bucket) CSVs + expanding dist stats
    ├── events.py                <- processed -> outlier / reversion events per symbol
    ├── plots.py                 <- spread overlay plot + equity curve
    ├── model.py                 <- load events, engineer features, train + evaluate
    ├── backtest.py              <- trade rules, transaction costs, Sharpe / win rate
    └── notebooks/
        ├── distributions.ipynb  <- exploratory counterpart of preprocess.py
        ├── events.ipynb         <- exploratory counterpart of events.py + plots.py
        ├── model.ipynb          <- exploratory counterpart of model.py
        └── backtest.ipynb       <- sandbox for the current backtest logic
```

Notebooks live under `src/notebooks/` and resolve paths as `Path("../..")` so
they still read the same `data/raw/`, `data/processed/`, and `config.yaml`
when run from that directory.

## Install

```
pip install -r requirements.txt
```

Python 3.10+. Everything needed (pandas, numpy, scikit-learn, pyyaml,
matplotlib, mlflow, prometheus-client, streamlit, xgboost, pyarrow, dvc) is
pinned in `requirements.txt`.

## Run the pipeline

```
# 1. raw ticks -> per-(symbol, dte, bucket) CSVs with streaming (dist_mean, dist_std, dist_count)
python src/preprocess.py

# 2. scan every (symbol, contract, session); persist events -> data/processed/events/{symbol}.csv
python src/events.py

# 3. load events, engineer features, time-split 2022-23 / 2024, train 3 classifiers + 6 regressors
python src/model.py

# 4. trade on the 2024 holdout: reversion -> fade the spread, divergence -> follow it,
#    continuation -> skip. Report per-position Sharpe / win rate and the equity curve.
python src/backtest.py
```

Every tunable (thresholds, window sizes, symbol list, train/test cutoff, RF
hyperparams, lot size, transaction cost, schemas, paths) lives in
`config.yaml`. Flip a value and rerun the affected stage — no code change.

### Interactive use

```python
# from src/
from events import run
run("BAJFINANCE", 2024, "AUG", day=2)   # per-day overlay plot with event markers
```

The notebooks under `src/notebooks/` do the equivalent interactively.

## `config.yaml` — single source of truth

Each `.py` module reads the same file at import:

```python
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load(open(ROOT / "config.yaml"))
```

Sections:

| section | used by | controls |
|---|---|---|
| `paths.*` | all | raw / processed / events / dates / train / test folders |
| `symbols`, `months`, `train_years` | preprocess, events | universe + iteration |
| `buckets` | preprocess | intraday time-bucket boundaries |
| `schemas.{fut, eq, vix}` | preprocess | header-less CSV column names |
| `preprocess.{vix_thresh, warmup_months, vix_file}` | preprocess | filters + warmup window |
| `events.{out_thresh, rev_thresh, min_ticks}` | events | outlier detection + resolution rules |
| `model.{feature_cols, train_year_cutoff, test_year, rf_*, random_state}` | model | feature list, split, RF hyperparams |
| `backtest.{positions, position_labels, lot_size, order_size, txn_cost_rate}` | backtest | order-sizing + cost assumption |

## Backtest semantics

- `pred_klass == 'reversion'`  → **SHORT** overvalued leg / **LONG** undervalued leg.
- `pred_klass == 'divergence'` → **OPPOSITE** position (bet on widening).
- `pred_klass == 'continuation'` → **no trade**.

One lot = 10 units, order size = 1 lot. Transaction cost = 0.01% of notional on each leg (`config.yaml::backtest.txn_cost_rate`). PnL reported in rupees, gross and net of costs, with an equity curve per full pipeline run. Detailed per-trade logs (entry/exit action, cashflows, reasoning string) are written to `data/processed/backtest_logs/trades_position_{0|1|2}_{near|mid|far}.csv`.

Lookahead bias is guarded at three points:
1. Features are `det_*` only — captured at the detection tick in preprocessing.
2. Entry uses `pred_klass`, never the actual `klass`.
3. Exit is at `res_timestamp`, which is the first forward tick crossing rev/div — the same tick live execution would hit. The forward scan is preprocessing-only (`I3` in [`CLAUDE.md`](CLAUDE.md)).

The `backtest.ipynb` notebook walks through this with per-trade numeric detail and an explicit list of illegal normalisations that would leak the future (none of which are done).

## MLOps integration readiness

The refactor sets each stage up so the following can be added without touching core logic.

### Docker

- Every `.py` is a runnable stage via `if __name__ == "__main__":`. That maps 1:1 to a container entrypoint: `CMD ["python", "src/events.py"]`.
- All paths in `config.yaml` are **relative** to `Path(__file__).resolve().parents[1]`. A volume mounted at `/app` keeps everything valid with no code change.
- Stages are filesystem-coupled (`preprocess` → `events` → `model` → `backtest`), so `docker compose` multi-service pipelines work out of the box with no shared state.
- `data/raw/` is marked read-only at the filesystem level — a misbehaving container can't corrupt source data.

### MLflow

- `train_position()` and `evaluate()` are pure functions. Wrapping `model.py::__main__` in `with mlflow.start_run():` and adding `mlflow.log_params(_cfg["model"])`, `mlflow.log_metric("mae_duration", ...)`, `mlflow.sklearn.log_model(clf, ...)` is a later additive PR.
- `_cfg["model"]` is a flat dict of exactly the hyperparameters the model uses — one-line `log_params`.

### Prometheus

- `track()` returns event dicts, `backtest()` returns a trades DataFrame. Counters / histograms (event rate, sharpe, win-rate, duration, txn cost) can be exported from the `__main__` blocks without modifying the core functions.
- When the pipeline shifts from batch to a long-running service, each module's `__main__` gets a `prometheus_client.start_http_server(port)` call.

### Grafana

- Read-only consumer. Two paths are ready: the Prometheus endpoints above, or the persisted `data/processed/events/*.csv` and `data/processed/backtest_logs/*.csv` via the CSV / Parquet data source plugin.

### Streamlit

- `plots.plot()` and `plots.plot_equity_curve()` both return `(fig, ax)` → `st.pyplot(fig)` drops in directly.
- `events.run()` returns an events DataFrame; `backtest.backtest()` returns a trades DataFrame — both are `st.dataframe`-ready.
- A thin `streamlit_app.py` can wire them together later with no library-side change.

### DVC

- `dvc` is in `requirements.txt`. Raw data is already filesystem-readonly and `.dvc` metadata files are untouched. `data/processed/` is the natural DVC output stage.

## Invariants

See [`CLAUDE.md`](CLAUDE.md) for the full rules. The critical ones:

- **I1** — no future data in distributions (snapshot before update, per session)
- **I2** — causal features (no peeking past `det_idx`)
- **I3** — forward scan is preprocessing-only
- **I4** — events schema is a contract; update it first
- **I5** — models consumed only through the `ReversionModel` ABC
- **I6** — all config lives in `config.yaml`; libraries take slices
