import heapq
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from plots import plot

ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load(open(ROOT / "config.yaml"))

processed_path = ROOT / _cfg["paths"]["processed"]
events_dir     = ROOT / _cfg["paths"]["events"]
months         = _cfg["months"]


# ---------- read_processed ----------
# Load the full per-symbol processed frame (all dte/bucket bins concatenated).
# Every row already carries timestamp, spread, dte, bucket, contract,
# dist_mean, dist_std, ltp (cash) and the bid/ask columns -- no raw I/O or
# re-merge needed downstream. Cached so re-calls are instant.

_processed_cache = {}

def read_processed(symbol):
    if symbol in _processed_cache:
        return _processed_cache[symbol]
    frames = []
    for f in sorted(processed_path.glob(f"{symbol}_*_*.csv")):
        frames.append(pd.read_csv(f, parse_dates=["timestamp"]))
    full = (pd.concat(frames, ignore_index=True)
              .sort_values("timestamp")
              .reset_index(drop=True))
    _processed_cache[symbol] = full
    print(f"read_processed({symbol}): {len(full):,} rows from {len(frames)} bins")
    return full


# ---------- contract helpers + processed slicer ----------
# No raw-file I/O. Filter the per-symbol processed frame to one contract in one
# (year, month) observation window -- or one single session when `day` is given.

def get_contracts_for(year, month_str):
    """(2025, 'JAN') -> ['25JAN', '25FEB', '25MAR']. Wraps across year-end."""
    mi = months.index(month_str)
    y2 = year % 100
    return [f"{(y2 + (mi + i) // 12):02d}{months[(mi + i) % 12]}" for i in range(3)]


def _resolve_day(day, year, m_num):
    """Accept day-of-month (23) or full YYYYMMDD (20240123); return YYYYMMDD int."""
    d = int(day)
    return d if d >= 10000 else year * 10000 + m_num * 100 + d


def slice_processed(full, contract_tok, year, month_str, day=None):
    """Return rows for one contract in (year, month_str); narrowed to `day`
    if given. day_fut in processed is an int YYYYMMDD."""
    month_tag = contract_tok[2:]                         # '25FEB' -> 'FEB'
    m_num     = months.index(month_str) + 1
    mask = ((full["contract"] == month_tag)
            & (full["timestamp"].dt.year  == year)
            & (full["timestamp"].dt.month == m_num))
    if day is not None:
        mask &= full["day_fut"].astype(int) == _resolve_day(day, year, m_num)
    sel = full[mask].copy()
    if sel.empty:
        return sel

    sel["z_score"] = (sel["spread"] - sel["dist_mean"]) / sel["dist_std"]
    sel["day_fut"] = sel["day_fut"].astype(int)
    sel["dte"]     = sel["dte"].astype(int)
    sel["bucket"]  = sel["bucket"].astype(int)
    return sel.reset_index(drop=True)


# ---------- track ----------
# Task points 5, 6, 7, 9. One session at a time.
#
# Detection  : first run of >= min_ticks consecutive ticks with |z| > out_thresh.
# Resolution : first tick that CROSSES the divergence or reversion line.
#              A "crossing" requires a transition -- z was on the non-triggered
#              side at tick k-1 and on the triggered side at tick k. If z
#              starts already past the divergence line at detection (sustained
#              extreme spread from open), there is no crossing -> continuation.
# Re-entry   : after a resolution, next scan starts only once |z| <= out_thresh.

def _first_run_start(is_out, lo, hi, k):
    seg = is_out[lo:hi + 1].astype(np.int32)
    if seg.size < k:
        return None
    cs  = np.concatenate(([0], np.cumsum(seg)))
    win = cs[k:] - cs[:-k]
    hits = np.where(win == k)[0]
    if hits.size == 0:
        return None
    return lo + int(hits[0])


def _find_resolution(z, start, hi, out_thresh, rev_thresh, side):
    """First true CROSSING of the div or rev line after `start`. Returns
    (idx, klass) where klass in {'divergence', 'reversion', 'continuation'}.
    Continuation -> idx = hi (end of session)."""
    if start >= hi:
        return hi, "continuation"
    div = out_thresh + rev_thresh
    rev = out_thresh - rev_thresh
    sub = z[start:hi + 1]                         # include z[start] to detect a
                                                  # crossing between start and start+1
    if side == 1:
        div_in = sub >= div
        rev_in = sub <= rev
    else:
        div_in = sub <= -div
        rev_in = sub >= -rev

    # A crossing at offset k (in sub) means sub[k-1] was outside the zone and
    # sub[k] is inside. Indexed by k in [1, len(sub)-1] -> array of len(sub)-1.
    cross_div = (~div_in[:-1]) & div_in[1:]
    cross_rev = (~rev_in[:-1]) & rev_in[1:]

    d_hit = int(np.argmax(cross_div)) if cross_div.any() else -1
    r_hit = int(np.argmax(cross_rev)) if cross_rev.any() else -1

    if d_hit == -1 and r_hit == -1:
        return hi, "continuation"
    if d_hit != -1 and (r_hit == -1 or d_hit <= r_hit):
        return start + 1 + d_hit, "divergence"
    return start + 1 + r_hit, "reversion"


def _find_extremum(spread, lo, hi, side):
    sub = spread[lo:hi + 1]
    return lo + (int(np.argmax(sub)) if side == 1 else int(np.argmin(sub)))


def track(session_df, out_thresh, rev_thresh, min_ticks=3):
    z      = session_df["z_score"].to_numpy()
    spread = session_df["spread"].to_numpy()
    ts     = session_df["timestamp"].to_numpy()
    n = len(z)
    if n == 0:
        return []

    pos_out = z >  out_thresh
    neg_out = z < -out_thresh
    outlier = pos_out | neg_out

    events = []
    pq = [(0, n - 1)]
    heapq.heapify(pq)

    while pq:
        lo, hi = heapq.heappop(pq)
        if lo >= hi:
            continue

        pos_start = _first_run_start(pos_out, lo, hi, min_ticks)
        neg_start = _first_run_start(neg_out, lo, hi, min_ticks)
        if pos_start is None and neg_start is None:
            continue
        if pos_start is None:            det_idx, side = neg_start, -1
        elif neg_start is None:          det_idx, side = pos_start,  1
        elif pos_start <= neg_start:     det_idx, side = pos_start,  1
        else:                            det_idx, side = neg_start, -1

        res_idx, klass = _find_resolution(z, det_idx, hi, out_thresh, rev_thresh, side)
        ext_idx        = _find_extremum(spread, det_idx, res_idx, side)

        events.append({
            "det_idx": det_idx, "ext_idx": ext_idx, "res_idx": res_idx,
            "side": side, "klass": klass,
            "det_time":   ts[det_idx],     "ext_time":   ts[ext_idx],     "res_time":   ts[res_idx],
            "det_spread": spread[det_idx], "ext_spread": spread[ext_idx], "res_spread": spread[res_idx],
            "det_z":      z[det_idx],      "ext_z":      z[ext_idx],      "res_z":      z[res_idx],
            "det_cash":   session_df["ltp"].iloc[det_idx],
            "bucket":     int(session_df["bucket"].iloc[det_idx]),
            "dte":        int(session_df["dte"].iloc[det_idx]),
        })

        # Re-entry gate: next scan can only begin once z has returned inside
        # the outlier band (|z| <= out_thresh).
        if res_idx < hi:
            sub = outlier[res_idx + 1:hi + 1]
            inside = np.where(~sub)[0]
            if inside.size:
                heapq.heappush(pq, (res_idx + 1 + int(inside[0]), hi))

    return events


# ---------- driver ----------
# run(symbol, year, month_str)            -> full month, 3 contracts.
# run(symbol, year, month_str, day=...)   -> single session, 3 contracts.
# day accepts day-of-month (23) or full YYYYMMDD (20240123).

def run(symbol, year, month_str, day=None,
        out_thresh=2.0, rev_thresh=1.0, min_ticks=3):

    full   = read_processed(symbol)
    tokens = get_contracts_for(year, month_str)
    scope  = f"{year}-{month_str}" + (f"  day={day}" if day is not None else "")
    print(f"{symbol} {scope}  contracts -> {tokens}")

    all_events = []
    for tok in tokens:
        df = slice_processed(full, tok, year, month_str, day=day)
        if df.empty:
            print(f"  {tok}: no rows in processed")
            continue

        events = []
        for _, session in df.groupby("day_fut", sort=True):
            session = session.reset_index(drop=True)
            events.extend(track(session, out_thresh, rev_thresh, min_ticks))

        plot(df, events, symbol, tok, out_thresh, rev_thresh)
        plt.show()

        for e in events: e["contract"] = tok
        all_events.extend(events)
        klass_cnt = dict(Counter(e["klass"] for e in events))
        print(f"  {tok}: ticks={len(df)}  events={len(events)}  classes={klass_cnt}")

    return pd.DataFrame(all_events)


# ---------- extract_events ----------
# Run track() over every (symbol, contract, session) and persist one events
# CSV per symbol to data/processed/events/. Each row carries:
#   - session metadata (symbol, contract, day_fut, bucket, dte, session_ticks)
#   - the event classification (side, klass) and session-local indices
#   - FULL per-tick snapshot of every processed column at det / ext / res
#     (prefixed det_, ext_, res_). That's fut/eq LTP/LTQ/OI/bid/ask + spread +
#     dist_mean/std/count + bucket/dte/contract at each of the three points.

def _snap(session, idx, prefix):
    row = session.iloc[idx]
    return {f"{prefix}_{col}": row[col] for col in session.columns}


def _event_row(session, symbol, contract, day, e):
    base = {
        "symbol":   symbol,
        "contract": contract,
        "day_fut":  day,
        "side":     e["side"],
        "klass":    e["klass"],
        "det_idx":  e["det_idx"],
        "ext_idx":  e["ext_idx"],
        "res_idx":  e["res_idx"],
        "session_ticks": len(session),
        "bucket":   int(session["bucket"].iloc[e["det_idx"]]),
        "dte":      int(session["dte"].iloc[e["det_idx"]]),
    }
    base.update(_snap(session, e["det_idx"], "det"))
    base.update(_snap(session, e["ext_idx"], "ext"))
    base.update(_snap(session, e["res_idx"], "res"))
    return base


def extract_events_symbol(symbol, out_thresh=2.0, rev_thresh=1.0, min_ticks=3):
    """Run track on every session for symbol, persist events CSV."""
    full = read_processed(symbol).copy()
    full["z_score"] = (full["spread"] - full["dist_mean"]) / full["dist_std"]

    rows = []
    for (contract, day), session in full.groupby(["contract", "day_fut"], sort=True):
        session = session.reset_index(drop=True)
        events  = track(session, out_thresh, rev_thresh, min_ticks)
        for e in events:
            rows.append(_event_row(session, symbol, contract, int(day), e))

    events_dir.mkdir(parents=True, exist_ok=True)
    out_file = events_dir / f"{symbol}.csv"
    if not rows:
        print(f"{symbol}: no events")
        # still write an empty file so downstream can see the run happened
        pd.DataFrame().to_csv(out_file, index=False)
        return None
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    klass_cnt = dict(df["klass"].value_counts())
    print(f"{symbol}: {len(df)} events -> {out_file}  classes={klass_cnt}")
    return df


def extract_all_events(out_thresh=2.0, rev_thresh=1.0, min_ticks=3):
    """Run extract_events_symbol for every symbol present in data/processed/."""
    symbols = sorted({f.stem.split("_")[0]
                      for f in processed_path.glob("*_*_*.csv")})
    print(f"extract_all_events: symbols = {symbols}")
    out = {}
    for sym in symbols:
        out[sym] = extract_events_symbol(sym, out_thresh, rev_thresh, min_ticks)
    return out


if __name__ == "__main__":
    extract_all_events(**_cfg["events"])
