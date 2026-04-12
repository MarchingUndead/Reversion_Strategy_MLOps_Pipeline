"""
preprocess.py — Tick CSVs -> distribution lookup + RAW events table.

Pipeline:
  1. Build (dte, bucket) spread distribution per symbol (weighted moments).
  2. Per session: z-score each tick against its (dte, bucket) cell.
  3. Detect outliers (|z| >= z_thresh). Multi-event per session via cooldown
     that walks past the entire elevated regime before re-arming.
  4. Forward-scan each event up to reversion_window_min, tracking BOTH:
       - most_reverted: min |z| during window  (for reversion classification)
       - most_extended: max |z| during window  (for continuation classification)
  5. Attach features (features.attach_features) AT DETECTION ONLY.
  6. Save raw events WITHOUT classification — label_events.py applies labels.

Output:
  data/processed/distributions/{SYM}_dist.parquet
  data/processed/reversion/{SYM}_reversion.parquet  (raw, unlabeled)
"""
import re, argparse, logging, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from calendars import compute_dte, get_contract_expiry, get_time_bucket
from features import attach_features

warnings.filterwarnings("ignore")

FUTURES_RE = re.compile(r"^([A-Z&]+)(\d{2}[A-Z]{3})FUT\.csv$", re.IGNORECASE)
CHUNKSIZE = 500_000


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_raw_csv(path):
    probe = pd.read_csv(path, sep=None, engine="python", header=None, nrows=1)
    n = len(probe.columns)
    if n < 5:
        raise ValueError(f"{path}: expected >=5 cols, got {n}")
    base  = ["date", "time", "price", "volume", "oi"]
    extra = (["bid_price","bid_volume","ask_price","ask_volume"]
             if n == 9 else [f"_x{i}" for i in range(n-5)])
    parts = []
    for chunk in pd.read_csv(path, sep=",", engine="c", header=None,
                             chunksize=CHUNKSIZE, low_memory=True):
        chunk.columns = base + extra
        chunk["timestamp"] = pd.to_datetime(
            pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
            + " " + chunk["time"].astype(str))
        keep = ["timestamp","price","volume","oi"]
        if n >= 9: keep += ["bid_price","ask_price"]
        parts.append(chunk[keep])
    df = pd.concat(parts, ignore_index=True); del parts
    return df.sort_values("timestamp").reset_index(drop=True)


def load_equity(paths):
    parts = []
    for p in paths:
        try:
            parts.append(read_raw_csv(p)[["timestamp","price"]]
                         .rename(columns={"price":"cash_price"}))
        except Exception: continue
    if not parts: return None
    return (pd.concat(parts, ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True))


def load_vix_daily(vix_paths, rolling_window):
    paths = [Path(p) for p in (vix_paths if isinstance(vix_paths, list) else [vix_paths])]
    parts = []
    for p in paths:
        if not p.exists(): continue
        for chunk in pd.read_csv(p, sep=",", engine="c", header=None,
                                 names=["date","time","vix","volume","oi"],
                                 chunksize=CHUNKSIZE):
            chunk["date"] = pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d")
            chunk["vix"]  = pd.to_numeric(chunk["vix"], errors="coerce")
            parts.append(chunk[["date","time","vix"]].dropna(subset=["vix"]))
    if not parts: return None
    df = pd.concat(parts, ignore_index=True)
    daily = (df.sort_values(["date","time"]).groupby("date")["vix"].last()
             .rename("vix_close").to_frame())
    daily["vix_rolling"] = daily["vix_close"].rolling(rolling_window, min_periods=5).mean().shift(1)
    return daily


def scan_symbol_files(cfg, symbol):
    fut_dirs = [Path(p) for p in cfg["paths"]["raw_futures"] if Path(p).exists()]
    eq_dirs  = [Path(p) for p in cfg["paths"]["raw_equity"]  if Path(p).exists()]
    eq_paths = [str(f) for d in eq_dirs for f in d.glob(f"{symbol}.csv")]
    if not eq_paths: raise FileNotFoundError(f"No equity CSV for {symbol}")
    contracts, seen = [], set()
    for d in fut_dirs:
        for f in sorted(d.glob("*.csv")):
            m = FUTURES_RE.match(f.name)
            if not m or m.group(1).upper() != symbol: continue
            c = m.group(2).upper()
            if c not in seen:
                seen.add(c); contracts.append((c, str(f)))
    if not contracts: raise FileNotFoundError(f"No futures CSVs for {symbol}")
    return sorted(contracts), eq_paths


def merge_contract(fut_path, eq_df, expiry, trading_days, time_buckets, contract):
    fut = read_raw_csv(fut_path).rename(columns={"price":"futures_price"})
    d0, d1 = fut["timestamp"].min()-pd.Timedelta(days=1), fut["timestamp"].max()+pd.Timedelta(days=1)
    eq = eq_df[(eq_df["timestamp"]>=d0)&(eq_df["timestamp"]<=d1)]
    base = pd.merge_asof(fut.sort_values("timestamp"), eq.sort_values("timestamp"),
                         on="timestamp", direction="backward").dropna(subset=["cash_price"])
    if base.empty: return None
    base["spread"] = (base["futures_price"]-base["cash_price"])/base["cash_price"].replace(0,np.nan)*100
    base["date"] = base["timestamp"].dt.normalize()
    base["dte"]  = compute_dte(base["date"], expiry, trading_days)
    base["contract"] = contract
    base["bucket"] = base["timestamp"].map(lambda t: get_time_bucket(t, time_buckets))
    base = base.set_index("timestamp").sort_index()
    base = base[~base.index.duplicated(keep="last")]
    return base.dropna(subset=["dte","spread"])


def build_distributions(symbol, contracts, eq_df, expiry_dates, trading_days, cfg, log):
    min_obs = cfg["distributions"]["min_obs"]
    tb = cfg["time_buckets"]
    # LEAKAGE FIX: build (dte, bucket) μ/σ from TRAIN years only. The
    # holdout year's ticks must never inform what "normal" looks like,
    # otherwise z-scores (and therefore detection, reversion_frac, and
    # dte_bucket_n_obs) are softly contaminated by the future.
    holdout_yy = int(cfg["training"]["holdout_year"])
    groups = {}
    n_dropped = 0
    for contract, fp in contracts:
        exp = get_contract_expiry(contract, expiry_dates)
        if exp is None: continue
        try: df = merge_contract(fp, eq_df, exp, trading_days, tb, contract)
        except Exception as e: log.warning(f"  [{contract}] {e}"); continue
        if df is None: continue
        before = len(df)
        df = df[df.index.year % 100 != holdout_yy]
        n_dropped += before - len(df)
        if df.empty: continue
        for (dte,bk), grp in df.groupby(["dte","bucket"]):
            if bk is None: continue
            k = (int(dte), bk)
            if k not in groups: groups[k] = {"s":[], "v":[]}
            groups[k]["s"].extend(grp["spread"].dropna().values.tolist())
            groups[k]["v"].extend(grp["volume"].clip(lower=0).fillna(1).values.tolist())
        del df
    rows = []
    for (dte,bk),g in groups.items():
        s = np.array(g["s"]); w = np.array(g["v"])
        if len(s) < min_obs: continue
        if w.sum() <= 0: w = np.ones(len(s))
        sw = float(w.sum()); mu = float((w*s).sum()/sw)
        sigma = float(np.sqrt(max((w*s*s).sum()/sw - mu*mu, 0)))
        rows.append({"dte":dte, "bucket":bk, "n":len(s), "mu":round(mu,6), "sigma":round(sigma,6)})
    log.info(f"  distributions: dropped {n_dropped:,} holdout-year (yy={holdout_yy}) ticks before fitting")
    return pd.DataFrame(rows)


def scan_session_for_events(session, lookup, cfg, daily_vix, date_val):
    z_thresh    = cfg["detection"]["z_thresh"]
    rev_window  = pd.Timedelta(minutes=cfg["detection"]["reversion_window_min"])
    cooldown_s  = cfg["detection"].get("cooldown_seconds", 60)
    sigma_floor = cfg["detection"]["sigma_floor"]

    dte = int(session["dte"].iloc[0])
    spreads = session["spread"].values
    ts = session.index
    n = len(spreads)

    z = np.full(n, np.nan); n_obs = np.full(n, np.nan)
    for i in range(n):
        bk = session.iloc[i]["bucket"]
        rec = lookup.get((dte, bk))
        if rec is None: continue
        mu, sg, no = rec
        if sg < sigma_floor: continue
        z[i] = (spreads[i] - mu) / sg
        n_obs[i] = no

    vix_at_det = np.nan
    if daily_vix is not None:
        prior = daily_vix[daily_vix.index < date_val]["vix_close"]
        if len(prior): vix_at_det = float(prior.iloc[-1])

    events = []
    i = 0
    while i < n:
        if np.isnan(z[i]) or abs(z[i]) < z_thresh:
            i += 1; continue

        det_idx, det_ts, z_det = i, ts[i], z[i]
        positive = z_det > 0
        spread_t = spreads[det_idx]

        # Track BOTH extrema within the reversion window:
        rev_idx, rev_z, rev_spread = det_idx, z_det, spread_t          # most reverted (toward 0)
        ext_idx, ext_z = det_idx, z_det                                 # most extended (away from 0)
        deadline = det_ts + rev_window
        last_in_window_idx = det_idx

        for j in range(det_idx + 1, n):
            if ts[j] > deadline: break
            last_in_window_idx = j
            zj = z[j]
            if np.isnan(zj): continue
            # Most reverted (closer to zero)
            if positive and zj < rev_z:
                rev_z, rev_idx, rev_spread = zj, j, spreads[j]
            elif (not positive) and zj > rev_z:
                rev_z, rev_idx, rev_spread = zj, j, spreads[j]
            # Most extended (further from zero)
            if positive and zj > ext_z:
                ext_z, ext_idx = zj, j
            elif (not positive) and zj < ext_z:
                ext_z, ext_idx = zj, j

        feats = attach_features(session, det_idx,
                                dte_bucket_n_obs=n_obs[det_idx],
                                vix_at_det=vix_at_det)

        max_rev_z = (z_det - rev_z) if positive else (rev_z - z_det)
        max_ext_z = (ext_z - z_det) if positive else (z_det - ext_z)

        row = {
            "date": pd.Timestamp(date_val).normalize(),
            "contract": session["contract"].iloc[0],
            "dte": dte, "bucket": session.iloc[det_idx]["bucket"],
            "direction": "above" if positive else "below",
            "detection_ts": pd.Timestamp(det_ts),
            "z_t_detection": round(float(z_det), 4),
            "spread_t": round(float(spread_t), 6),
            "futures_t": round(float(session["futures_price"].iloc[det_idx]), 4),
            "cash_t":    round(float(session["cash_price"].iloc[det_idx]), 4),

            # Most-reverted extremum (used for reversion classification)
            "extremum_z":           round(float(rev_z), 4),
            "extremum_spread":      round(float(rev_spread), 6),
            "extremum_ts":          pd.Timestamp(ts[rev_idx]),
            "max_reversion_z":      round(float(max_rev_z), 4),
            "max_reversion_spread": round(float(spread_t - rev_spread) if positive else float(rev_spread - spread_t), 6),
            "time_to_extremum_min": round((pd.Timestamp(ts[rev_idx]) - pd.Timestamp(det_ts)).total_seconds()/60, 2),

            # Most-extended extremum (used for continuation classification)
            "most_extended_z":      round(float(ext_z), 4),
            "most_extended_ts":     pd.Timestamp(ts[ext_idx]),
            "max_extended_z":       round(float(max_ext_z), 4),
            "time_to_extended_min": round((pd.Timestamp(ts[ext_idx]) - pd.Timestamp(det_ts)).total_seconds()/60, 2),

            # Window-end snapshot
            "window_end_idx_z":     round(float(z[last_in_window_idx]), 4) if not np.isnan(z[last_in_window_idx]) else np.nan,
            "spread_at_window_end": round(float(spreads[last_in_window_idx]), 6),
            "time_to_window_end":   round((pd.Timestamp(ts[last_in_window_idx]) - pd.Timestamp(det_ts)).total_seconds()/60, 2),

            # Reversion fraction (label-independent target)
            "reversion_frac": round(float(max_rev_z / abs(z_det)) if abs(z_det) > 1e-9 else 0.0, 4),
        }
        row.update(feats)
        events.append(row)

        # Cooldown: walk past the elevated regime, then sleep cooldown_s.
        k = last_in_window_idx + 1
        while k < n and (np.isnan(z[k]) or abs(z[k]) >= z_thresh):
            k += 1
        if k < n:
            cool_until = pd.Timestamp(ts[k]) + pd.Timedelta(seconds=cooldown_s)
            while k < n and pd.Timestamp(ts[k]) < cool_until:
                k += 1
        i = k

    return events


def scan_reversion_events(symbol, contracts, eq_df, dist_df, expiry_dates,
                          trading_days, cfg, daily_vix, log):
    lookup = {(int(r["dte"]), r["bucket"]): (r["mu"], r["sigma"], r["n"])
              for _, r in dist_df.iterrows()}
    vix_gate = cfg["vix"]["gate_max"]
    tb = cfg["time_buckets"]
    all_events = []
    for contract, fp in contracts:
        exp = get_contract_expiry(contract, expiry_dates)
        if exp is None: continue
        try: df = merge_contract(fp, eq_df, exp, trading_days, tb, contract)
        except Exception as e: log.warning(f"  [{contract}] {e}"); continue
        if df is None: continue
        df["_d"] = df.index.normalize()
        for date_val, session in df.groupby("_d"):
            if daily_vix is not None:
                prior = daily_vix[daily_vix.index < date_val]
                if len(prior):
                    # Spot VIX gate (symmetric across train/test). The rolling
                    # mean is still computed in load_vix_daily for diagnostics
                    # / future use, but the gate fires on prior-day spot only.
                    spot = prior["vix_close"].iloc[-1]
                    if not pd.isna(spot) and spot > vix_gate: continue
            session = session.sort_index().dropna(subset=["spread"])
            if session.empty: continue
            rows = scan_session_for_events(session, lookup, cfg, daily_vix, date_val)
            for r in rows: r["symbol"] = symbol
            all_events.extend(rows)
        del df

    if all_events:
        tmp = pd.DataFrame(all_events)
        log.info(f"  detected {len(tmp)} raw events; "
                 f"avg_t_reverted={tmp['time_to_extremum_min'].mean():.1f}min "
                 f"avg_t_extended={tmp['time_to_extended_min'].mean():.1f}min "
                 f"median_t_reverted={tmp['time_to_extremum_min'].median():.1f}min")

    return pd.DataFrame(all_events)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("preprocess")
    cfg = load_config(args.config)

    expiry_df = pd.read_csv(cfg["paths"]["expiry_csv"])
    col = next(c for c in expiry_df.columns if "expiry" in c.lower() or "date" in c.lower())
    expiry_dates = sorted(pd.to_datetime(expiry_df[col]).unique())
    trading_df = pd.read_csv(cfg["paths"]["trading_csv"], parse_dates=["date"])
    trading_days = pd.DatetimeIndex(sorted(trading_df["date"]))
    log.info(f"Loaded {len(expiry_dates)} expiries, {len(trading_days)} trading days")

    daily_vix = load_vix_daily(cfg["paths"]["raw_vix"], cfg["vix"]["rolling_1m"])
    log.info("VIX loaded" if daily_vix is not None else "VIX disabled")

    dist_out = Path(cfg["paths"]["processed"]) / "distributions"
    rev_out  = Path(cfg["paths"]["processed"]) / "reversion"
    dist_out.mkdir(parents=True, exist_ok=True); rev_out.mkdir(parents=True, exist_ok=True)

    for symbol in (cfg.get("symbols") or []):
        log.info(f"[{symbol}] scanning...")
        contracts, eq_paths = scan_symbol_files(cfg, symbol)
        log.info(f"[{symbol}] {len(contracts)} contracts")
        eq_df = load_equity(eq_paths)
        if eq_df is None: log.warning(f"[{symbol}] no equity"); continue
        log.info(f"[{symbol}] equity ticks: {len(eq_df):,}")

        log.info(f"[{symbol}] building distributions...")
        dist_df = build_distributions(symbol, contracts, eq_df, expiry_dates, trading_days, cfg, log)
        dist_df.to_parquet(dist_out / f"{symbol}_dist.parquet", index=False)
        log.info(f"[{symbol}] dist cells: {len(dist_df)}")

        log.info(f"[{symbol}] scanning events...")
        rev_df = scan_reversion_events(symbol, contracts, eq_df, dist_df,
                                       expiry_dates, trading_days, cfg, daily_vix, log)
        out_path = rev_out / f"{symbol}_reversion.parquet"
        rev_df.to_parquet(out_path, index=False)
        log.info(f"[{symbol}] {len(rev_df)} raw events written -> {out_path}")
        log.info(f"[{symbol}] run label_events.py to apply classification + split")
        del eq_df, dist_df, rev_df

    log.info("Done.")


if __name__ == "__main__":
    main()