# ---------- load events ----------
# One event row per detection: det/ext/res snapshots + session metadata.
# Columns described in src/events.ipynb (extract_events cell).

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, mean_absolute_error, confusion_matrix,
    f1_score, accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load(open(ROOT / "config.yaml"))

events_path = ROOT / _cfg["paths"]["events"]
MODELS_DIR  = ROOT / "models"
MONTHS      = _cfg["months"]


def load_events_all():
    frames = []
    for f in sorted(events_path.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["det_timestamp", "ext_timestamp", "res_timestamp"])
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------- features, targets, contract-position ----------
# Contract position relative to the session month:
#   0 = near  (contract expires in the session's own month)
#   1 = mid   (next month)
#   2 = far   (month after that)
# Rows with position > 2 shouldn't exist in a 3-contract ladder; dropped if any.
#
# Targets (per spec):
#   klass          - categorical {reversion, divergence, continuation}
#   duration_sec   - res - det (seconds)
#   revert_delta   - |det_z| - |res_z|  (>0 means z came back toward zero)
#
# Features (all known AT detection, no forward look):
#   det_z_score, det_spread, side, dte, bucket
#   det_dist_std, det_dist_count, det_fut_ltq, det_oi_fut, det_ltq
#   det_fut_ba = det_fut_askprice - det_fut_bidprice  (fut bid-ask width)
#   det_eq_ba  = det_eq_askprice  - det_eq_bidprice   (eq  bid-ask width)

def _session_month_from_day(d):
    return pd.Timestamp(str(int(d))).month - 1                          # 0..11


# ---------- train per contract position ----------
# 3 models per position: classifier for klass, regressors for duration & revert.
# "Train separately on all 3 contracts" -> one triplet per position (0=near,
# 1=mid, 2=far) as specified in the notebook cell at the top.

def _clean(df, cols):
    tgt = ["klass", "duration_sec", "revert_delta"]
    return df.dropna(subset=cols + tgt)


def train_position(train_df, position, feature_cols):
    sub = _clean(train_df[train_df["position"] == position], feature_cols)
    if len(sub) < 30:
        print(f"position={position}: only {len(sub)} rows, skipping")
        return None
    X      = sub[feature_cols].to_numpy()
    y_klas = sub["klass"].to_numpy()
    y_dur  = sub["duration_sec"].to_numpy()
    y_rev  = sub["revert_delta"].to_numpy()

    clf     = RandomForestClassifier(n_estimators=_cfg["model"]["rf_n_estimators"], max_depth=_cfg["model"]["rf_max_depth"],
                                     n_jobs=-1, random_state=_cfg["model"]["random_state"]).fit(X, y_klas)
    reg_dur = RandomForestRegressor (n_estimators=_cfg["model"]["rf_n_estimators"], max_depth=_cfg["model"]["rf_max_depth"],
                                     n_jobs=-1, random_state=_cfg["model"]["random_state"]).fit(X, y_dur)
    reg_rev = RandomForestRegressor (n_estimators=_cfg["model"]["rf_n_estimators"], max_depth=_cfg["model"]["rf_max_depth"],
                                     n_jobs=-1, random_state=_cfg["model"]["random_state"]).fit(X, y_rev)
    klass_cnt = dict(pd.Series(y_klas).value_counts())
    print(f"position={position}: trained on {len(sub)} events  klass={klass_cnt}")
    return clf, reg_dur, reg_rev


# ---------- evaluate ----------
# Classification report for klass + MAE for duration and revert_delta on the
# held-out (test) slice, per contract position.

def evaluate(test_df, position, feature_cols, models):
    sub = _clean(test_df[test_df["position"] == position], feature_cols)
    if sub.empty:
        print(f"position={position}: no test rows"); return None
    clf, reg_dur, reg_rev = models[position]
    X = sub[feature_cols].to_numpy()
    pred_k   = clf.predict(X)
    pred_dur = reg_dur.predict(X)
    pred_rev = reg_rev.predict(X)
    print(f"\n=== position={position}  (n={len(sub)}) ===")
    print(classification_report(sub["klass"], pred_k, zero_division=0))
    print(f"duration_sec   MAE = {mean_absolute_error(sub['duration_sec'], pred_dur):8.1f}")
    print(f"revert_delta   MAE = {mean_absolute_error(sub['revert_delta'], pred_rev):8.3f}")
    sub = sub.copy()
    sub["pred_klass"] = pred_k
    sub["pred_dur"]   = pred_dur
    sub["pred_rev"]   = pred_rev
    return sub


# ==================================================================
# Additive helpers used by backtest.py and streamlit_app.py.
# The functions above are untouched.
# ==================================================================

def _slice_by_range(events, range_pair):
    """Filter events whose det_timestamp falls in [start_ym, end_ym] inclusive.
    range_pair is like ['2024-01', '2024-06']."""
    start = pd.Timestamp(range_pair[0] + "-01")
    end   = pd.Timestamp(range_pair[1] + "-01") + pd.offsets.MonthEnd(0)
    end   = end.replace(hour=23, minute=59, second=59)
    mask  = (events["det_timestamp"] >= start) & (events["det_timestamp"] <= end)
    return events.loc[mask].reset_index(drop=True)


def prepare_events(events):
    """Add session_month, contract_month, position, targets, derived features."""
    events = events.copy()
    events["session_month"]  = events["day_fut"].apply(_session_month_from_day)
    events["contract_month"] = events["contract"].map({m: i for i, m in enumerate(MONTHS)})
    events["position"]       = (events["contract_month"] - events["session_month"]) % 12
    events = events[events["position"].isin([0, 1, 2])].reset_index(drop=True)

    events["duration_sec"]  = (events["res_timestamp"] - events["det_timestamp"]).dt.total_seconds()
    events["revert_delta"]  = events["det_z_score"].abs() - events["res_z_score"].abs()
    events["spread_change"] = events["res_spread"] - events["det_spread"]
    events["det_fut_ba"]    = events["det_fut_askprice"] - events["det_fut_bidprice"]
    events["det_eq_ba"]     = events["det_eq_askprice"]  - events["det_eq_bidprice"]
    return events


def split_events(events, split):
    """split in {'train', 'val', 'dev_test', 'hidden'} -> config range pair."""
    key = {"train":    "train_range",
           "val":      "val_range",
           "dev_test": "dev_test_range",
           "hidden":   "hidden_range"}[split]
    return _slice_by_range(events, _cfg["model"][key])


def save_models(models, out_dir=MODELS_DIR):
    """Persist the {position: (clf, reg_dur, reg_rev)} dict to pickle files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, trio in models.items():
        if trio is None:
            continue
        clf, reg_dur, reg_rev = trio
        with open(out_dir / f"classifier_pos{p}.pkl", "wb") as f: pickle.dump(clf, f)
        with open(out_dir / f"duration_pos{p}.pkl",   "wb") as f: pickle.dump(reg_dur, f)
        with open(out_dir / f"revert_pos{p}.pkl",     "wb") as f: pickle.dump(reg_rev, f)
    print(f"saved models -> {out_dir}")


def eval_metrics(eval_df):
    """Extract flat metrics from the DataFrame returned by `evaluate()`.
    Reads the `pred_klass` / `pred_dur` / `pred_rev` columns evaluate() adds —
    does not recompute predictions or touch evaluate()."""
    if eval_df is None or eval_df.empty:
        return {}
    return {
        "n_test":       int(len(eval_df)),
        "f1_macro":     float(f1_score(eval_df["klass"], eval_df["pred_klass"],
                                       average="macro", zero_division=0)),
        "accuracy":     float(accuracy_score(eval_df["klass"], eval_df["pred_klass"])),
        "mae_duration": float(mean_absolute_error(eval_df["duration_sec"], eval_df["pred_dur"])),
        "mae_revert":   float(mean_absolute_error(eval_df["revert_delta"], eval_df["pred_rev"])),
    }


def load_models(positions=(0, 1, 2), in_dir=MODELS_DIR):
    """Inverse of save_models — returns {position: (clf, reg_dur, reg_rev)}.
    Missing files produce None for that position."""
    in_dir = Path(in_dir)
    out = {}
    for p in positions:
        try:
            with open(in_dir / f"classifier_pos{p}.pkl", "rb") as f: clf     = pickle.load(f)
            with open(in_dir / f"duration_pos{p}.pkl",   "rb") as f: reg_dur = pickle.load(f)
            with open(in_dir / f"revert_pos{p}.pkl",     "rb") as f: reg_rev = pickle.load(f)
            out[p] = (clf, reg_dur, reg_rev)
        except FileNotFoundError:
            out[p] = None
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-n-estimators", type=int, default=_cfg["model"]["rf_n_estimators"])
    parser.add_argument("--rf-max-depth",    type=int, default=_cfg["model"]["rf_max_depth"])
    args = parser.parse_args()

    _cfg["model"]["rf_n_estimators"] = args.rf_n_estimators
    _cfg["model"]["rf_max_depth"]    = args.rf_max_depth

    feature_cols = _cfg["model"]["feature_cols"]
    positions    = _cfg["backtest"]["positions"]

    events = prepare_events(load_events_all())
    train  = split_events(events, "train")
    val    = split_events(events, "val")
    print(f"train={len(train)}  val={len(val)}")

    models = {p: train_position(train, p, feature_cols) for p in positions}
    evals  = {p: evaluate(val, p, feature_cols, models)  for p in positions}
    save_models(models)

    for p in positions:
        m = eval_metrics(evals.get(p))
        if m:
            print(f"pos{p}: {m}")
