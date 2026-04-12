"""
calendars.py — Expiry resolution and DTE computation. Pure functions.

Used by preprocess.py:
    from calendars import compute_dte, get_contract_expiry
"""

import re
import pandas as pd
import numpy as np

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def get_contract_expiry(contract_str, expiry_dates):
    """Resolve a contract code like '25MAR' to its expiry Timestamp.

    Looks up the entry in expiry_dates whose (year%100, month) matches.
    Returns a pandas Timestamp normalized to midnight, or None if not found.
    """
    m = re.match(r"^(\d{2})([A-Z]{3})$", contract_str.upper().strip())
    if not m:
        return None
    yy, mon = int(m.group(1)), MONTH_MAP.get(m.group(2))
    if mon is None:
        return None
    for ts in expiry_dates:
        t = pd.Timestamp(ts)
        if (t.year % 100) == yy and t.month == mon:
            return t.normalize()
    return None


def compute_dte(current_dates, expiry_date, trading_days):
    """Trading days remaining from each date in current_dates until expiry (inclusive).

    current_dates: array-like of dates/timestamps (one per row of the tick frame).
    expiry_date:   single expiry Timestamp.
    trading_days:  pd.DatetimeIndex of all valid trading sessions, sorted.

    Returns:
      numpy array of floats — integer DTE for valid trading days,
      np.nan for holidays / weekends / dates after expiry. Caller is
      expected to dropna() on the result.

    Vectorized: O(N log T) via searchsorted, no Python loop over rows.
    """
    if not isinstance(trading_days, pd.DatetimeIndex):
        trading_days = pd.DatetimeIndex(sorted(trading_days))

    expiry_norm = pd.Timestamp(expiry_date).normalize()
    idx_exp = trading_days.searchsorted(expiry_norm, side="left")
    # Expiry must itself be a trading day; if not, snap to the last trading
    # day on or before expiry (cash-settled expiries always land on a session).
    if idx_exp >= len(trading_days) or trading_days[idx_exp] != expiry_norm:
        idx_exp = max(0, idx_exp - 1)

    # Normalize input to a DatetimeIndex of midnights
    cur = pd.to_datetime(pd.Series(list(current_dates))).dt.normalize()
    cur_idx = pd.DatetimeIndex(cur.values)

    # Position of each current_date in the trading calendar
    positions = trading_days.searchsorted(cur_idx, side="left")

    out = np.full(len(cur_idx), np.nan, dtype=float)
    for i, (pos, d) in enumerate(zip(positions, cur_idx)):
        if pos >= len(trading_days):
            continue                          # past end of calendar
        if trading_days[pos] != d:
            continue                          # not a trading day -> NaN
        if pos > idx_exp:
            continue                          # after expiry -> NaN
        out[i] = float(idx_exp - pos + 1)     # inclusive count
    return out


def compute_dte_single(date_val, expiry_dates, trading_days):
    """Trading days from date_val to the next expiry on or after date_val.

    Convenience wrapper used by live/inference code paths.
    """
    ts = pd.Timestamp(date_val).normalize()
    future = [pd.Timestamp(d) for d in expiry_dates
              if pd.Timestamp(d).normalize() >= ts]
    if not future:
        return None
    expiry = future[0]
    result = compute_dte([ts], expiry, trading_days)
    val = result[0]
    return None if np.isnan(val) else int(val)

from datetime import time as dtime


def get_time_bucket(ts, time_buckets):
    """Return bucket label for a timestamp, or None if outside all buckets.

    time_buckets: list of [label, "HH:MM", "HH:MM"] triples from config.
                  Buckets are half-open: [start, end).
    """
    t = ts.time() if hasattr(ts, "time") else ts
    for label, s, e in time_buckets:
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        if dtime(sh, sm) <= t < dtime(eh, em):
            return label
    return None


def load_calendars(cfg):
    """Load expiry dates and trading days from CSVs in config.

    Accepts cfg as either a plain dict (cfg["paths"]["expiry_csv"]) or an
    attribute-style object (cfg.paths.expiry_csv) — handles both so this
    works with the old build_tables.py and the new preprocess.py.
    """
    def _get(obj, key):
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    paths = _get(cfg, "paths")
    expiry_path  = _get(paths, "expiry_csv")
    trading_path = _get(paths, "trading_csv")

    expiry_df = pd.read_csv(expiry_path)
    col = next((c for c in expiry_df.columns
                if "expiry" in c.lower() or "date" in c.lower()), None)
    if col is None:
        raise ValueError(f"{expiry_path}: no expiry/date column found")
    expiry_dates = sorted(pd.to_datetime(expiry_df[col]).unique())

    trading_df = pd.read_csv(trading_path, parse_dates=["date"])
    trading_days = pd.DatetimeIndex(sorted(trading_df["date"]))
    return expiry_dates, trading_days