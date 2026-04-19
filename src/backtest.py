# ---------- backtest ----------
# Trade rule: open a trade at detection iff the classifier predicts 'reversion'.
# For side=+1 (spread abnormally high) we short the spread; for side=-1 we go long.
# PnL per event = -side * (res_spread - det_spread), in spread-percent units.
# Close at the resolution time (res_*). No slippage / no financing.

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from plots import plot_equity_curve

ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load(open(ROOT / "config.yaml"))


def backtest(eval_df, label):
    if eval_df is None or eval_df.empty:
        print(f"{label}: no eval data"); return None
    trades = eval_df[eval_df["pred_klass"] == "reversion"].copy()
    if trades.empty:
        print(f"{label}: model never predicted reversion -> 0 trades"); return None
    trades["pnl"] = -trades["side"] * trades["spread_change"]
    n      = len(trades)
    wins   = (trades["pnl"] > 0).sum()
    mean_p = trades["pnl"].mean()
    std_p  = trades["pnl"].std(ddof=1) if n > 1 else np.nan
    sharpe = (mean_p / std_p * np.sqrt(n)) if std_p and std_p > 0 else np.nan
    total  = trades["pnl"].sum()
    print(f"{label}: trades={n}  win_rate={wins/n:.2%}  mean_pnl={mean_p:+.4f}%  "
          f"std={std_p:.4f}  sharpe≈{sharpe:.2f}  total_pnl={total:+.3f}%")
    return trades


if __name__ == "__main__":
    from model import (load_events_all, _session_month_from_day, train_position,
                       evaluate, MONTHS)

    events = load_events_all()
    events["session_month"]  = events["day_fut"].apply(_session_month_from_day)
    events["contract_month"] = events["contract"].map({m: i for i, m in enumerate(MONTHS)})
    events["position"]       = (events["contract_month"] - events["session_month"]) % 12
    events = events[events["position"].isin([0, 1, 2])].reset_index(drop=True)

    events["duration_sec"]  = (events["res_timestamp"] - events["det_timestamp"]).dt.total_seconds()
    events["revert_delta"]  = events["det_z_score"].abs() - events["res_z_score"].abs()
    events["spread_change"] = events["res_spread"] - events["det_spread"]
    events["det_fut_ba"]    = events["det_fut_askprice"] - events["det_fut_bidprice"]
    events["det_eq_ba"]     = events["det_eq_askprice"]  - events["det_eq_bidprice"]

    events["year"] = events["det_timestamp"].dt.year
    train = events[events["year"] <= _cfg["model"]["train_year_cutoff"]].reset_index(drop=True)
    test  = events[events["year"] == _cfg["model"]["test_year"]].reset_index(drop=True)

    feature_cols = _cfg["model"]["feature_cols"]
    models = {p: train_position(train, p, feature_cols) for p in _cfg["backtest"]["positions"]}
    evals  = {p: evaluate(test, p, feature_cols, models) for p in _cfg["backtest"]["positions"]}

    print("\n=== backtest (held-out) ===")
    all_trades = []
    for p, label in zip(_cfg["backtest"]["positions"], _cfg["backtest"]["position_labels"]):
        t = backtest(evals.get(p), f"position={p} ({label})")
        if t is not None:
            all_trades.append(t)

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True).sort_values("det_timestamp")
        plot_equity_curve(combined)
