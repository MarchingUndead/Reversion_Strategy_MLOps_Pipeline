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


if __name__ == "__main__":
    from model import (load_events_all, prepare_events, split_events,
                       train_position, evaluate, load_models)

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "dev_test", "hidden"], default="val",
                        help="which split to backtest on (default: val)")
    parser.add_argument("--confirm-holdout", action="store_true",
                        help="required to run --split hidden")
    parser.add_argument("--retrain", action="store_true",
                        help="force retraining instead of loading ./models/*.pkl")
    args = parser.parse_args()

    if args.split == "hidden" and not args.confirm_holdout:
        print("ERROR: --split hidden requires --confirm-holdout "
              "(2025 is the one-shot holdout; see config.yaml::model.hidden_range).",
              file=sys.stderr)
        sys.exit(2)

    events = prepare_events(load_events_all())
    eval_df = split_events(events, args.split)
    print(f"backtest split={args.split}  n_events={len(eval_df)}")

    feature_cols = _cfg["model"]["feature_cols"]
    positions    = _cfg["backtest"]["positions"]

    # Load pickled models first; retrain only if missing or --retrain given.
    models = {} if args.retrain else load_models(positions)
    if args.retrain or any(models.get(p) is None for p in positions):
        print("[backtest] training models in-process (no pickles found or --retrain)")
        train = split_events(events, "train")
        models = {p: train_position(train, p, feature_cols) for p in positions}

    evals = {p: evaluate(eval_df, p, feature_cols, models) for p in positions}

    logs_dir = ROOT / "data" / "backtest_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== backtest (split={args.split}) ===")
    all_trades = []
    for p, label in zip(_cfg["backtest"]["positions"], _cfg["backtest"]["position_labels"]):
        log_path = logs_dir / f"trades_position_{p}_{label}_{args.split}.csv"
        t = backtest(evals.get(p), f"position={p} ({label})", trade_log_path=log_path)
        if t is not None:
            all_trades.append(t)

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True).sort_values("det_timestamp")
        plot_equity_curve(combined)
