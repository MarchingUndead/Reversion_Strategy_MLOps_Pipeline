"""
features.py — Snapshot feature engineering at detection tick.

Pure function: attach_features(session, det_idx, dte_bucket_n_obs, vix_at_det)
returns a dict of features computed strictly from ticks at or before det_idx.
No leakage. Called once per detected outlier event by preprocess.py.
"""

import numpy as np
import pandas as pd

SESSION_OPEN_MIN = 9 * 60 + 15


def _safe_window(arr, end_idx, n):
    """Return arr[max(0, end_idx-n+1) : end_idx+1] without going negative."""
    start = max(0, end_idx - n + 1)
    return arr[start:end_idx + 1]


def attach_features(session, det_idx, dte_bucket_n_obs=np.nan, vix_at_det=np.nan):
    """Compute snapshot features at detection tick.

    session:    per-session DataFrame with columns spread, volume, oi
                (and bid_price/ask_price if available). Indexed by timestamp.
    det_idx:    integer position of the detection tick within session.
    Returns:    dict of feature_name -> float.
    """
    spreads = session["spread"].values
    volumes = session["volume"].values if "volume" in session.columns else None
    ois     = session["oi"].values     if "oi"     in session.columns else None
    ts      = session.index

    feats = {}

    # --- Spread velocity & acceleration (5-tick) ---
    win5 = _safe_window(spreads, det_idx, 6)  # need 6 points for 5-step diff
    if len(win5) >= 2:
        feats["spread_velocity_5tick"] = float(win5[-1] - win5[0]) / max(1, len(win5) - 1)
    else:
        feats["spread_velocity_5tick"] = 0.0

    if len(win5) >= 3:
        diffs = np.diff(win5)
        feats["spread_accel_5tick"] = float(diffs[-1] - diffs[0]) / max(1, len(diffs) - 1)
    else:
        feats["spread_accel_5tick"] = 0.0

    # --- Volume z-score over trailing 20 ticks ---
    if volumes is not None:
        vwin = _safe_window(volumes, det_idx, 20)
        if len(vwin) >= 5:
            v_mu  = float(np.nanmean(vwin[:-1])) if len(vwin) > 1 else float(vwin[-1])
            v_sd  = float(np.nanstd(vwin[:-1]))  if len(vwin) > 1 else 0.0
            feats["volume_zscore_20tick"] = (
                float((vwin[-1] - v_mu) / v_sd) if v_sd > 1e-9 else 0.0
            )
        else:
            feats["volume_zscore_20tick"] = 0.0
    else:
        feats["volume_zscore_20tick"] = 0.0

    # --- OI change over trailing 20 ticks (RELATIVE — scale-invariant) ---
    # Was: raw delta owin[-1] - owin[0]. That's an absolute OI count and
    # scales with contract size + year, which made 2025 values land 80×
    # outside the 2022-24 train envelope and Ridge extrapolated wildly.
    # Relative change is bounded in practice and comparable across years.
    if ois is not None:
        owin = _safe_window(ois, det_idx, 20)
        if len(owin) >= 2:
            base = float(owin[0])
            delta = float(owin[-1] - owin[0])
            feats["oi_change_20tick"] = delta / base if base > 1.0 else 0.0
        else:
            feats["oi_change_20tick"] = 0.0
    else:
        feats["oi_change_20tick"] = 0.0

    # --- Bid-ask features (UNCOMMENT when 2025 bid-ask data is available) ---
    # if "bid_price" in session.columns and "ask_price" in session.columns:
    #     bid = float(session["bid_price"].values[det_idx])
    #     ask = float(session["ask_price"].values[det_idx])
    #     if bid > 0 and ask > 0:
    #         mid = (bid + ask) / 2.0
    #         feats["quoted_spread"]     = ask - bid
    #         feats["rel_quoted_spread"] = (ask - bid) / mid
    #         feats["mid_distance"]      = float(spreads[det_idx]) - mid
    #     else:
    #         feats["quoted_spread"] = np.nan
    #         feats["rel_quoted_spread"] = np.nan
    #         feats["mid_distance"] = np.nan
    # else:
    #     feats["quoted_spread"] = np.nan
    #     feats["rel_quoted_spread"] = np.nan
    #     feats["mid_distance"] = np.nan

    # --- Time since session open (minutes) ---
    det_ts = pd.Timestamp(ts[det_idx])
    det_min = det_ts.hour * 60 + det_ts.minute
    feats["time_since_session_open_min"] = float(max(0, det_min - SESSION_OPEN_MIN))

    # --- Prior spread IQR over trailing 30 ticks (vol proxy) ---
    swin = _safe_window(spreads, det_idx, 30)
    if len(swin) >= 4:
        q75, q25 = np.nanpercentile(swin[:-1], [75, 25])
        feats["prior_spread_iqr_30tick"] = float(q75 - q25)
    else:
        feats["prior_spread_iqr_30tick"] = 0.0

    # --- VIX level at detection (passed in, daily ffill from preprocess) ---
    feats["vix_level"] = float(vix_at_det) if vix_at_det is not None and not pd.isna(vix_at_det) else np.nan

    # --- Distribution-cell sample size (signal quality flag) ---
    feats["dte_bucket_n_obs"] = float(dte_bucket_n_obs) if not pd.isna(dte_bucket_n_obs) else np.nan

    return feats