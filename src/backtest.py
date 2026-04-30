# ---------- backtest ----------
# Trade rule:
#   pred_klass == 'reversion'    -> bet on the spread narrowing
#   pred_klass == 'divergence'   -> bet on the spread widening (sign-flipped position)
#   pred_klass == 'continuation' -> no trade
#
# Positions (V = lot_size * order_size):
#   sign = direction * side  where direction = {reversion: +1, divergence: -1}
#   sign = +1 -> SHORT 1 lot FUT + LONG 1 lot CASH
#   sign = -1 -> LONG  1 lot FUT + SHORT 1 lot CASH
#
# Cash flows (rupee units):
#   entry = sign * V * (det_fut_ltp - det_ltp)
#   exit  = -sign * V * (res_fut_ltp - res_ltp)
#   cost  = txn_cost_rate * V * [(det_fut + det_eq) + (res_fut + res_eq)]
#   pnl   = entry + exit - cost
#
# No slippage, no financing. Close at res_timestamp (first forward crossing of
# rev/div, or EOD for continuation).

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from plots import plot_equity_curve

ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load(open(ROOT / "config.yaml"))

LOT_SIZE      = _cfg["backtest"]["lot_size"]
ORDER_SIZE    = _cfg["backtest"]["order_size"]
TXN_COST_RATE = _cfg["backtest"]["txn_cost_rate"]
V             = LOT_SIZE * ORDER_SIZE


def backtest(eval_df, label, trade_log_path=None):
    if eval_df is None or eval_df.empty:
        print(f"{label}: no eval data"); return None

    direction_map = {"reversion": 1, "divergence": -1, "continuation": 0}
    trades = eval_df.copy()
    trades["direction"] = trades["pred_klass"].map(direction_map)
    skipped = (trades["direction"] == 0).sum()
    trades  = trades[trades["direction"] != 0].copy()
    if trades.empty:
        print(f"{label}: all {skipped} predictions were continuation -> 0 trades"); return None

    trades["sign"]           = trades["direction"] * trades["side"]
    trades["det_abs_spread"] = trades["det_fut_ltp"] - trades["det_ltp"]
    trades["res_abs_spread"] = trades["res_fut_ltp"] - trades["res_ltp"]

    trades["entry_cashflow"] =  trades["sign"] * V * trades["det_abs_spread"]
    trades["exit_cashflow"]  = -trades["sign"] * V * trades["res_abs_spread"]

    trades["entry_notional"] = V * (trades["det_fut_ltp"] + trades["det_ltp"])
    trades["exit_notional"]  = V * (trades["res_fut_ltp"] + trades["res_ltp"])
    trades["txn_cost"]       = TXN_COST_RATE * (trades["entry_notional"] + trades["exit_notional"])

    trades["pnl_gross"] = trades["entry_cashflow"] + trades["exit_cashflow"]
    trades["pnl"]       = trades["pnl_gross"] - trades["txn_cost"]

    trades["entry_action"] = trades["sign"].map({
         1: f"SHORT {ORDER_SIZE} lot FUT + LONG {ORDER_SIZE} lot CASH",
        -1: f"LONG {ORDER_SIZE} lot FUT + SHORT {ORDER_SIZE} lot CASH",
    })
    trades["exit_action"] = trades["sign"].map({
         1: f"BUY {ORDER_SIZE} lot FUT + SELL {ORDER_SIZE} lot CASH (close)",
        -1: f"SELL {ORDER_SIZE} lot FUT + BUY {ORDER_SIZE} lot CASH (close)",
    })

    def _reason(r):
        thesis   = "narrow" if r["direction"] == 1 else "widen"
        fut_move = "drop"   if r["sign"] == 1      else "rise"
        return (f"det_z={r['det_z_score']:+.2f} -> model predicts {r['pred_klass']}; "
                f"spread expected to {thesis}, fut expected to {fut_move}. "
                f"actual: {r['klass']} at {r['res_timestamp']} (res_z={r['res_z_score']:+.2f}).")
    trades["reasoning"] = trades.apply(_reason, axis=1)

    by_pred = dict(trades["pred_klass"].value_counts())
    n      = len(trades)
    wins   = (trades["pnl"] > 0).sum()
    mean_p = trades["pnl"].mean()
    std_p  = trades["pnl"].std(ddof=1) if n > 1 else np.nan
    sharpe = (mean_p / std_p * np.sqrt(n)) if std_p and std_p > 0 else np.nan
    g_tot  = trades["pnl_gross"].sum()
    c_tot  = trades["txn_cost"].sum()
    n_tot  = trades["pnl"].sum()
    print(f"{label}:")
    print(f"  skipped_continuations = {skipped}")
    print(f"  trades_by_pred = {by_pred}")
    print(f"  trades = {n}  win_rate = {wins/n:.2%}  mean_pnl = ₹{mean_p:+.2f}  "
          f"std = ₹{std_p:.2f}  sharpe ≈ {sharpe:.2f}")
    print(f"  pnl_gross = ₹{g_tot:+.2f}   txn_costs = ₹{c_tot:.2f}   pnl_net = ₹{n_tot:+.2f}")

    if trade_log_path is not None:
        log_cols = ["det_timestamp", "symbol", "contract", "side", "pred_klass", "klass",
                    "direction", "sign",
                    "entry_action", "det_fut_ltp", "det_ltp", "det_abs_spread", "det_z_score",
                    "entry_cashflow",
                    "exit_action", "res_timestamp", "res_fut_ltp", "res_ltp", "res_abs_spread",
                    "res_z_score", "exit_cashflow",
                    "entry_notional", "exit_notional", "txn_cost",
                    "pnl_gross", "pnl", "reasoning"]
        trades[log_cols].to_csv(trade_log_path, index=False)
        print(f"  trade log -> {trade_log_path}")

    return trades


def _load_models_mlflow(positions, symbols, versions):
    """Load (clf, reg_dur, reg_rev) per (symbol, position) from the MLflow Registry.

    Versions sourced from config.yaml::registry.versions — one int per
    (symbol, position) cell, applied to all 3 heads. Cells not listed (or
    whose load fails) return None so the existing in-process retrain
    fallback in main() can pick them up.

    External wrapper — does NOT modify model.py's pipeline functions.

    Resolves the version through MlflowClient and loads via runs:/<run_id>/...
    rather than models:/<name>/<version> to avoid MLflow writing the
    registered_model_meta sidecar (fails on read-only mlruns mounts in the
    airflow scheduler container).
    """
    import mlflow.sklearn
    from mlflow import MlflowClient
    client = MlflowClient()
    out = {}
    for sym in symbols:
        sym_versions = versions.get(sym) or {}
        for p in positions:
            v_num = sym_versions.get(p)
            if v_num is None:
                print(f"[backtest] sym={sym} pos={p}: not pinned in "
                      f"config.registry.versions — will fall back to retrain")
                out[(sym, p)] = None
                continue
            try:
                trio = []
                for head in ("classifier", "duration", "revert"):
                    name = f"reversion-{head}-pos{p}-{sym}"
                    v = client.get_model_version(name, str(v_num))
                    trio.append(mlflow.sklearn.load_model(
                        f"runs:/{v.run_id}/model_{head}"))
                out[(sym, p)] = tuple(trio)
                print(f"[backtest] sym={sym} pos={p}: loaded v{v_num}")
            except Exception as exc:
                print(f"[backtest] sym={sym} pos={p}: load v{v_num} failed "
                      f"({type(exc).__name__}: {exc})")
                out[(sym, p)] = None
    return out


if __name__ == "__main__":
    from model import (load_events_all, prepare_events, split_events,
                       train_position, evaluate)

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "dev_test", "hidden"], default="hidden",
                        help="which split to backtest on (default: hidden = 2025). "
                             "Models still train only on 2022-24 (train_range). "
                             "Hidden output is observation-only; do not feed results "
                             "back into tuning decisions.")
    parser.add_argument("--retrain", action="store_true",
                        help="force in-process retraining instead of loading from MLflow Registry")
    parser.add_argument("--confirm-holdout", action="store_true",
                        help="required to backtest --split hidden (one-shot 2025 holdout).")
    args = parser.parse_args()

    if args.split == "hidden" and not args.confirm_holdout:
        sys.exit("--split hidden requires --confirm-holdout")

    events = prepare_events(load_events_all())
    eval_df = split_events(events, args.split)
    print(f"backtest split={args.split}  n_events={len(eval_df)}")

    feature_cols = _cfg["model"]["feature_cols"]
    positions    = _cfg["backtest"]["positions"]
    symbols      = _cfg["symbols"]

    # Versions are pinned in config.yaml::registry.versions. Normalize YAML
    # keys (which may load as either int or str depending on quoting) into
    # {sym: {int(pos): int(version)}} before passing to the loader.
    versions_raw = _cfg.get("registry", {}).get("versions", {}) or {}
    versions = {sym: {int(p): int(v) for p, v in (cell or {}).items()}
                for sym, cell in versions_raw.items()}

    # Load from MLflow Registry first; retrain in-process if requested or any head missing.
    models = ({} if args.retrain
              else _load_models_mlflow(positions, symbols, versions))
    if args.retrain or any(models.get((s, p)) is None
                           for s in symbols for p in positions):
        print("[backtest] training models in-process (--retrain or MLflow load failed)")
        train = split_events(events, "train")
        models = {}
        for s in symbols:
            train_sym = train[train["symbol"] == s]
            for p in positions:
                models[(s, p)] = train_position(train_sym, p, feature_cols)

    # Evaluate per (symbol, position): pre-filter the eval df by symbol so
    # evaluate() only sees rows for the model trained on that symbol. evaluate()
    # itself is unchanged — it still filters by position internally.
    evals = {}
    for s in symbols:
        eval_sym = eval_df[eval_df["symbol"] == s]
        for p in positions:
            trio = models.get((s, p))
            if trio is None:
                evals[(s, p)] = None
                continue
            evals[(s, p)] = evaluate(eval_sym, p, feature_cols, {p: trio})

    logs_dir = ROOT / "data" / "backtest_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== backtest (split={args.split}) ===")
    all_trades = []
    for s in symbols:
        for p, label in zip(positions, _cfg["backtest"]["position_labels"]):
            log_path = logs_dir / f"trades_{s}_position_{p}_{label}_{args.split}.csv"
            t = backtest(evals.get((s, p)),
                         f"sym={s} position={p} ({label})",
                         trade_log_path=log_path)
            if t is not None:
                all_trades.append(t)

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True).sort_values("det_timestamp")
        plot_equity_curve(combined)
