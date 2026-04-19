# ---------- load events ----------
# One event row per detection: det/ext/res snapshots + session metadata.
# Columns described in src/events.ipynb (extract_events cell).

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load(open(ROOT / "config.yaml"))

events_path = ROOT / _cfg["paths"]["events"]
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


if __name__ == "__main__":
    events = load_events_all()
    print(f"events: {len(events)} rows across symbols {sorted(events['symbol'].unique())}")
    print(events.groupby(["symbol", "klass"]).size().unstack(fill_value=0))

    events["session_month"]  = events["day_fut"].apply(_session_month_from_day)
    events["contract_month"] = events["contract"].map({m: i for i, m in enumerate(MONTHS)})
    events["position"]       = (events["contract_month"] - events["session_month"]) % 12

    before = len(events)
    events = events[events["position"].isin([0, 1, 2])].reset_index(drop=True)
    print(f"kept {len(events)} / {before} events in positions 0..2")

    # Targets
    events["duration_sec"] = (events["res_timestamp"] - events["det_timestamp"]).dt.total_seconds()
    events["revert_delta"] = events["det_z_score"].abs() - events["res_z_score"].abs()
    events["spread_change"] = events["res_spread"] - events["det_spread"]

    # Derived features
    events["det_fut_ba"] = events["det_fut_askprice"] - events["det_fut_bidprice"]
    events["det_eq_ba"]  = events["det_eq_askprice"]  - events["det_eq_bidprice"]

    feature_cols = _cfg["model"]["feature_cols"]
    print("feature cols:", feature_cols)
    print(events.groupby(["position", "klass"]).size().unstack(fill_value=0))

    # ---------- time-based train / test split ----------
    # All available events span 2022-2024 (processed data upper bound).
    # Train on events through 2023, held-out on 2024. No random shuffle -- strictly
    # causal split.

    events["year"] = events["det_timestamp"].dt.year
    train = events[events["year"] <= _cfg["model"]["train_year_cutoff"]].reset_index(drop=True)
    test  = events[events["year"] == _cfg["model"]["test_year"]].reset_index(drop=True)
    print(f"train: {len(train)} events (<={_cfg['model']['train_year_cutoff']})   test: {len(test)} events ({_cfg['model']['test_year']})")
    print("train klass:", dict(train['klass'].value_counts()))
    print("test  klass:", dict(test['klass'].value_counts()))

    models = {p: train_position(train, p, feature_cols) for p in (0, 1, 2)}
    evals  = {p: evaluate(test, p, feature_cols, models) for p in (0, 1, 2)}
