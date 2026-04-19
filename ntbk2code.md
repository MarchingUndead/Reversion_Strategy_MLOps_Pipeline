# Notebook → Python module split

Mirror of the planning doc driving this refactor. Four notebooks under
`src/notebooks/` are the exploratory counterparts of five library modules
under `src/`. `config.yaml` is the single source of truth for every parameter.

## File layout

```
Reversion_Strategy_MLOps_Pipeline/
├── config.yaml                  <- all tunables; consumed by every src/*.py
└── src/
    ├── preprocess.py            <- distributions.ipynb
    ├── events.py                <- events.ipynb (track + event extraction)
    ├── plots.py                 <- events.ipynb (plot) + model.ipynb (equity curve)
    ├── model.py                 <- model.ipynb (load + train + evaluate)
    ├── backtest.py              <- model.ipynb (backtest fn) + backtest.ipynb (v2 rules)
    └── notebooks/
        ├── distributions.ipynb
        ├── events.ipynb
        ├── model.ipynb
        └── backtest.ipynb
```

Notebooks set `root = Path("../..").resolve()` so paths still resolve to
`data/raw/`, `data/processed/`, and `config.yaml` when run from
`src/notebooks/`.

## Function-to-file mapping

### `src/preprocess.py` ← `notebooks/distributions.ipynb`

| Function | Notebook cell |
|---|---|
| `find_first_gte(arr, target)` | `ed198820` |
| `find_exact(arr, target)` | `ed198820` |
| `get_dte(day_str)` | `ed198820` |
| `get_bucket(time_str)` | `ed198820` |
| `get_high_vix_days(df_vix, threshold)` | `d09d92e7` |
| `store_group(df_group, symbol, dte, bucket)` | `8ed54d03` |
| `__main__` | cells `cde1b29f` (raw → per-(dte,bucket) CSVs) + `1e24a23c` (expanding distribution stats + warmup mask) |

Module-level state retained: `trading_cal`, `expiry_dates`, `trading_days_sorted`,
`expiry_list`, `df_vix`, `high_vix_days`.

### `src/events.py` ← `notebooks/events.ipynb`

| Function | Notebook cell |
|---|---|
| `read_processed(symbol)` | `da1038ae` |
| `get_contracts_for(year, month_str)` | `ad680c00` |
| `_resolve_day(day, year, m_num)` | `ad680c00` |
| `slice_processed(full, contract_tok, year, month_str, day=None)` | `ad680c00` |
| `_first_run_start(is_out, lo, hi, k)` | `11053cb6` |
| `_find_resolution(z, start, hi, out_thresh, rev_thresh, side)` | `11053cb6` |
| `_find_extremum(spread, lo, hi, side)` | `11053cb6` |
| `track(session_df, out_thresh, rev_thresh, min_ticks)` | `11053cb6` |
| `run(symbol, year, month_str, day, ...)` | `a4e7c0cc` — imports `plot` from `plots.py` |
| `_snap(session, idx, prefix)` | `a6500c68` |
| `_event_row(session, symbol, contract, day, e)` | `a6500c68` |
| `extract_events_symbol(symbol, ...)` | `a6500c68` |
| `extract_all_events(...)` | `a6500c68` |
| `__main__` | `extract_all_events(**_cfg["events"])` |

Module-level cache `_processed_cache = {}` is preserved.

### `src/plots.py` ← `notebooks/events.ipynb` + `notebooks/model.ipynb`

| Function | Source |
|---|---|
| `plot(df, events, symbol, contract, out_thresh, rev_thresh)` | events.ipynb cell `edf95070` |
| `plot_equity_curve(combined)` | model.ipynb cell `73c0db19` — inline equity block wrapped in a function |

No `__main__`. Pure presentation, imported by `events.py` and `backtest.py`.

### `src/model.py` ← `notebooks/model.ipynb`

| Function | Notebook cell |
|---|---|
| `load_events_all()` | `9ee905b2` |
| `_session_month_from_day(d)` | `50a773f3` |
| `_clean(df, cols)` | `fb732176` |
| `train_position(train_df, position, feature_cols)` | `fb732176` |
| `evaluate(test_df, position, feature_cols, models)` | `e2700e3e` |
| `__main__` | `50a773f3` (feature/target engineering) + `05dc933a` (time split) + per-position `train_position` and `evaluate` loops |

### `src/backtest.py` ← `notebooks/model.ipynb` + `notebooks/backtest.ipynb`

| Function | Source |
|---|---|
| `backtest(eval_df, label, trade_log_path=None)` | backtest.ipynb cell `cell-06-backtest-fn` — v2 rules (divergence sign-flip, continuation skipped, txn costs) supersedes model.ipynb cell `73c0db19` |
| `__main__` | imports training pipeline from `model.py`, runs `backtest(evals[p], label, log_path)` per position, then `plot_equity_curve(combined)` |

## Preservation rules

1. **No logic changes.** Function bodies copied verbatim from the notebooks; module-level constants became `_cfg["..."]` lookups.
2. **Same names.** Every notebook function name is preserved, including underscore helpers. The only new name is `plot_equity_curve` (the notebook's inline equity block had no name).
3. **Same comments.** Notebook comments were kept; no new commentary added.
4. **Layout.** `imports → _cfg → module-level state → functions → if __name__ == "__main__":`.

## Dropped from the .py files

- `events.ipynb` cell `4733c0f3` — the `dist_lookup` builder. Superseded by `read_processed`; no function depends on it.
- `events.ipynb` cell `f3fa547a` — raw-2025 loader that built `plot_data`. Superseded by `slice_processed`.
- `events.ipynb` cell `85ed7572` — `log_df = run(...)` demo call.
- `distributions.ipynb` cell `8e380234` — `df.tail(10)` sanity peek.
- `distributions.ipynb` cell `2e10e088` unused `train_vix`/`test_vix` Path objects.
- `model.ipynb` cell `2459b460` — 2025-stub print.

All dropped items remain in the notebooks for reference.

## Execution order

```
python src/preprocess.py    # raw ticks  -> data/processed/*.csv
python src/events.py        # processed  -> data/processed/events/*.csv
python src/model.py         # events     -> per-position classifier + regressors
python src/backtest.py      # model      -> per-position PnL + equity curve + trade logs
```

Each stage consumes the previous stage's output from disk, so each runs
independently once its inputs exist.
