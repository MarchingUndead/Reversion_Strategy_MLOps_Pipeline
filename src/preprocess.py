"""
preprocess.py — Builds distribution and reversion tables from raw tick CSVs.

Processes one contract at a time to stay within memory budget.

Pass 1: accumulate (dte, bucket) spread groups across contracts → fit distributions
Pass 2: scan each contract's sessions with two pointers for outlier events

Memory-optimised CSV reading:
  read_raw_csv() uses chunked reading (default 200k rows per chunk) so that
  large tick files (often >10M rows) never materialise fully in RAM.
  load_equity() likewise concatenates chunks lazily.

Outputs:
  data/processed/distributions/{SYMBOL}_dist.parquet
  data/processed/reversion/{SYMBOL}_reversion.parquet
"""

import re
import json
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, t as t_dist, skewnorm, lognorm, kstest

from calendars import compute_dte, get_contract_expiry, get_time_bucket

warnings.filterwarnings("ignore")

DIST_MAP    = {"normal": norm, "t": t_dist, "skewnorm": skewnorm, "lognorm": lognorm}
FUTURES_RE  = re.compile(r"^([A-Z&]+)(\d{2}[A-Z]{3})FUT\.csv$", re.IGNORECASE)
SESSION_START = 9 * 60 + 15
SESSION_LEN   = 375

# Default chunk size for CSV reading — tune based on available RAM.
# 200k rows ≈ 30–50 MB per chunk for a 5-column tick file.
CSV_CHUNK_SIZE = 200_000


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_calendars(cfg):
    expiry_df = pd.read_csv(cfg["paths"]["expiry_csv"])
    col = next((c for c in expiry_df.columns if "expiry" in c.lower() or "date" in c.lower()), None)
    if col is None:
        raise ValueError("expiry_csv: no date/expiry column found")
    expiry_dates = sorted(pd.to_datetime(expiry_df[col]).unique())
    trading_df   = pd.read_csv(cfg["paths"]["trading_csv"], parse_dates=["date"])
    trading_days = pd.DatetimeIndex(sorted(trading_df["date"]))
    return expiry_dates, trading_days


def _detect_csv_columns(path):
    """Peek at the first line to determine column count without loading the file."""
    with open(path, "r") as f:
        first_line = f.readline()
    # Try common separators
    for sep in [",", "\t", ";", "|"]:
        if sep in first_line:
            return len(first_line.split(sep)), sep
    return len(first_line.split(",")), ","


def read_raw_csv(path, chunksize=CSV_CHUNK_SIZE):
    """Read a headerless tick CSV in chunks and return a single DataFrame.

    Chunked reading ensures that a 2 GB tick file never sits entirely in
    RAM — each chunk is parsed, trimmed to the columns we need, and
    appended to a list.  The final concat is on the slim (4–5 column)
    frames, not the raw text.
    """
    ncols, sep = _detect_csv_columns(path)

    if ncols < 5:
        raise ValueError(f"{path}: expected >=5 columns, got {ncols}")

    base = ["date", "time", "price", "volume", "oi"]
    if ncols == 9:
        extra = ["bid_price", "bid_volume", "ask_price", "ask_volume"]
    else:
        extra = [f"_x{i}" for i in range(ncols - 5)]
    col_names = base + extra

    keep = ["timestamp", "price", "volume", "oi"]
    if ncols >= 9:
        keep += ["bid_price", "ask_price"]

    parts = []
    reader = pd.read_csv(
        path, sep=sep, engine="c", header=None,
        names=col_names, chunksize=chunksize,
    )

    for chunk in reader:
        chunk["timestamp"] = pd.to_datetime(
            pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
            + " " + chunk["time"].astype(str)
        )
        parts.append(chunk[keep].copy())

    if not parts:
        raise ValueError(f"{path}: file is empty")

    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def load_equity(equity_paths, chunksize=CSV_CHUNK_SIZE):
    """Load and concatenate equity tick files, reading each in chunks."""
    parts = []
    for p in equity_paths:
        try:
            df = read_raw_csv(p, chunksize=chunksize)[["timestamp", "price"]]
            df = df.rename(columns={"price": "cash_price"})
            parts.append(df)
        except Exception:
            continue

    if not parts:
        raise RuntimeError("No readable equity files")

    return (pd.concat(parts, ignore_index=True)
              .sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last"))


def load_vix_daily(vix_paths, rolling_window):
    paths = [Path(p) for p in (vix_paths if isinstance(vix_paths, list) else [vix_paths])]
    parts = []
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_csv(p, sep=None, engine="python", header=None,
                         names=["date", "time", "vix", "volume", "oi"])
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df["vix"]  = pd.to_numeric(df["vix"], errors="coerce")
        parts.append(df.dropna(subset=["vix"]))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    daily = (df.sort_values(["date", "time"])
               .groupby("date")["vix"].last()
               .rename("vix_close").to_frame())
    # shift(1): rolling mean for day D uses only days < D
    daily["vix_rolling"] = daily["vix_close"].rolling(rolling_window, min_periods=5).mean().shift(1)
    return daily


def scan_symbol_files(cfg, symbol):
    futures_dirs = [Path(p) for p in cfg["paths"]["raw_futures"] if Path(p).exists()]
    equity_dirs  = [Path(p) for p in cfg["paths"]["raw_equity"]  if Path(p).exists()]

    equity_paths = [str(f) for d in equity_dirs for f in d.glob(f"{symbol}.csv")]
    if not equity_paths:
        raise FileNotFoundError(f"No equity CSV for {symbol}")

    contracts, seen = [], set()
    for d in futures_dirs:
        for f in sorted(d.glob("*.csv")):
            m = FUTURES_RE.match(f.name)
            if not m or m.group(1).upper() != symbol:
                continue
            c = m.group(2).upper()
            if c not in seen:
                seen.add(c)
                contracts.append((c, str(f)))

    if not contracts:
        raise FileNotFoundError(f"No futures CSVs for {symbol}")
    return sorted(contracts), equity_paths


def _merge_contract(fut_path, eq_df, expiry, trading_days, time_buckets, contract):
    """Load one futures file, merge with equity slice, return slim tick DataFrame."""
    fut = read_raw_csv(fut_path).rename(columns={"price": "futures_price"})

    # slice equity to contract date range to minimise merge_asof work
    date_min = fut["timestamp"].min() - pd.Timedelta(days=1)
    date_max = fut["timestamp"].max() + pd.Timedelta(days=1)
    eq_slice = eq_df[(eq_df["timestamp"] >= date_min) & (eq_df["timestamp"] <= date_max)]

    base = pd.merge_asof(
        fut.sort_values("timestamp"),
        eq_slice.sort_values("timestamp"),
        on="timestamp", direction="backward"
    ).dropna(subset=["cash_price"])

    if base.empty:
        return None

    base["spread"]   = (base["futures_price"] - base["cash_price"]) / base["cash_price"].replace(0, np.nan) * 100
    base["date"]     = base["timestamp"].dt.normalize()
    base["dte"]      = compute_dte(base["date"], expiry, trading_days)
    base["contract"] = contract
    base["bucket"]   = base["timestamp"].map(lambda ts: get_time_bucket(ts, time_buckets))
    base = base.set_index("timestamp").sort_index()
    base = base[~base.index.duplicated(keep="last")]
    return base.dropna(subset=["dte", "spread"])


def build_distributions(symbol, contracts, eq_df, expiry_dates, trading_days, cfg, log):
    """Accumulate (dte, bucket) groups one contract at a time, then fit distributions."""
    min_obs      = cfg["distributions"]["min_obs"]
    families     = cfg["distributions"]["families"]
    time_buckets = cfg["time_buckets"]

    # groups[(dte, bucket)] = {"spreads": [...], "volumes": [...]}
    groups = {}

    for contract, fut_path in contracts:
        expiry = get_contract_expiry(contract, expiry_dates)
        if expiry is None:
            continue
        try:
            df = _merge_contract(fut_path, eq_df, expiry, trading_days, time_buckets, contract)
        except Exception as e:
            log.warning(f"  [{contract}] skipped: {e}")
            continue
        if df is None:
            continue

        for (dte, bucket), grp in df.groupby(["dte", "bucket"]):
            if bucket is None:
                continue
            key = (int(dte), bucket)
            spreads = grp["spread"].dropna().values
            volumes = grp["volume"].clip(lower=0).fillna(1).values
            if key not in groups:
                groups[key] = {"spreads": [], "volumes": []}
            groups[key]["spreads"].extend(spreads.tolist())
            groups[key]["volumes"].extend(volumes.tolist())

        del df

    rows = []
    for (dte, bucket), g in groups.items():
        spreads = np.array(g["spreads"])
        if len(spreads) < min_obs:
            continue
        weights = np.array(g["volumes"])
        if weights.sum() <= 0:
            weights = np.ones(len(spreads))
        sum_w   = float(weights.sum())
        sum_wx  = float((weights * spreads).sum())
        sum_wx2 = float((weights * spreads**2).sum())
        mu      = sum_wx / sum_w
        sigma   = float(np.sqrt(max(sum_wx2 / sum_w - mu**2, 0)))

        best_name, best_ks, best_params = "normal", np.inf, None
        for name in families:
            try:
                if name == "lognorm":
                    shift  = max(0.0, -spreads.min() + 1e-6)
                    params = lognorm.fit(spreads + shift, floc=0)
                    ks, _  = kstest(spreads + shift, lognorm.cdf, args=params)
                    params = params + (shift,)
                else:
                    params = DIST_MAP[name].fit(spreads)
                    ks, _  = kstest(spreads, DIST_MAP[name].cdf, args=params)
                if ks < best_ks:
                    best_ks, best_name, best_params = ks, name, params
            except Exception:
                continue

        rows.append({
            "symbol": symbol, "dte": dte, "bucket": bucket,
            "n": len(spreads), "mu": round(mu, 6), "sigma": round(sigma, 6),
            "sum_w": sum_w, "sum_wx": sum_wx, "sum_wx2": sum_wx2,
            "dist_name": best_name,
            "dist_params": json.dumps(list(best_params)) if best_params else "[]",
            "p5":  round(float(np.percentile(spreads,  5)), 5),
            "p25": round(float(np.percentile(spreads, 25)), 5),
            "p50": round(float(np.percentile(spreads, 50)), 5),
            "p75": round(float(np.percentile(spreads, 75)), 5),
            "p95": round(float(np.percentile(spreads, 95)), 5),
        })

    return pd.DataFrame(rows)


def scan_reversion_events(symbol, contracts, eq_df, dist_df, expiry_dates, trading_days, cfg, daily_vix, log):
    """Two-pointer session scan, one contract at a time.  Multiple events per session."""
    z_thresh    = cfg["detection"]["z_thresh"]
    exit_t      = cfg["detection"]["event_exit_threshold"]
    sigma_floor = cfg["detection"]["sigma_floor"]
    vix_gate    = cfg["vix"]["gate_max"]
    vix_window  = cfg["vix"]["rolling_1m"]
    time_buckets = cfg["time_buckets"]

    lookup = {
        (int(r["dte"]), r["bucket"]): (r["mu"], r["sigma"])
        for _, r in dist_df.iterrows()
        if r["sigma"] >= sigma_floor
    }

    all_events = []

    for contract, fut_path in contracts:
        expiry = get_contract_expiry(contract, expiry_dates)
        if expiry is None:
            continue
        try:
            df = _merge_contract(fut_path, eq_df, expiry, trading_days, time_buckets, contract)
        except Exception as e:
            log.warning(f"  [{contract}] skipped: {e}")
            continue
        if df is None:
            continue

        df["_date"] = df.index.normalize()

        for date_val, session in df.groupby("_date"):

            # VIX filter: use only strictly prior days
            if daily_vix is not None:
                prior = daily_vix[daily_vix.index < date_val]["vix_close"]
                if len(prior) >= 5:
                    rv = prior.rolling(vix_window, min_periods=5).mean().iloc[-1]
                    if not np.isnan(rv) and rv > vix_gate:
                        continue

            session = session.sort_index().dropna(subset=["spread"])
            if session.empty:
                continue

            dte     = int(session["dte"].iloc[0])
            ticks   = session.index
            spreads = session["spread"].values
            n       = len(ticks)

            z    = np.full(n, np.nan)
            mu_s = np.full(n, np.nan)
            sg_s = np.full(n, np.nan)
            for i in range(n):
                key = (dte, session.iloc[i]["bucket"])
                if key in lookup:
                    mu, sigma = lookup[key]
                    z[i], mu_s[i], sg_s[i] = (spreads[i] - mu) / sigma, mu, sigma

            eod_spread = spreads[-1]

            # two pointers — multiple events per session
            left = 0
            while left < n:
                if np.isnan(z[left]) or abs(z[left]) < z_thresh:
                    left += 1
                    continue

                z_det    = z[left]
                positive = z_det > 0
                rev_thr  =  (z_thresh - exit_t)           if positive else -(z_thresh - exit_t)
                cont_thr =  max(z_thresh + exit_t, z_det) if positive else min(-(z_thresh + exit_t), z_det)

                exit_type = exit_right = None
                right = left + 1
                while right < n:
                    if np.isnan(z[right]):
                        right += 1
                        continue
                    if positive:
                        if z[right] < rev_thr:
                            exit_type, exit_right = "reversion", right; break
                        if z[right] > cont_thr:
                            exit_type, exit_right = "continuation", right; break
                    else:
                        if z[right] > rev_thr:
                            exit_type, exit_right = "reversion", right; break
                        if z[right] < cont_thr:
                            exit_type, exit_right = "continuation", right; break
                    right += 1

                mu_det, sg_det = mu_s[left], sg_s[left]
                z_eod = (eod_spread - mu_det) / (sg_det + 1e-12) if not np.isnan(mu_det) else np.nan

                if exit_type is not None:
                    pz  = z[exit_right:];       valid = ~np.isnan(pz)
                    ps  = spreads[exit_right:]
                    pt  = ticks[exit_right:]
                    if valid.sum() == 0:
                        left = exit_right + 1
                        continue
                    pz, ps, pt = pz[valid], ps[valid], pt[valid]
                    ext_i = (int(np.argmin(pz)) if positive else int(np.argmax(pz))) if exit_type == "reversion" \
                             else (int(np.argmax(pz)) if positive else int(np.argmin(pz)))
                    ext_z, ext_sp, ext_ts = float(pz[ext_i]), float(ps[ext_i]), pt[ext_i]
                    exit_ts = ticks[exit_right]
                    resume_i = exit_right + 1
                else:
                    if not ((z_eod > z_det) if positive else (z_eod < z_det)):
                        # non-event: z stayed put without crossing back — no signal
                        left = n
                        continue
                    exit_type = "divergence"
                    ext_z, ext_sp, ext_ts = float(z_eod), float(eod_spread), ticks[-1]
                    exit_ts = pd.NaT
                    resume_i = n

                det_ts   = ticks[left]
                t_min    = det_ts.hour * 60 + det_ts.minute
                all_events.append({
                    "symbol":               symbol,
                    "date":                 date_val,
                    "contract":             contract,
                    "dte":                  dte,
                    "bucket":               session.iloc[left]["bucket"],
                    "direction":            "above" if positive else "below",
                    "event_type":           exit_type,
                    "detection_ts":         det_ts,
                    "z_t":                  round(float(z_det), 4),
                    "spread_t":             round(float(spreads[left]), 5),
                    "exit_ts":              exit_ts,
                    "extremum_z":           round(ext_z, 4),
                    "extremum_spread":      round(ext_sp, 5),
                    "extremum_ts":          ext_ts,
                    "time_to_extremum_min": round((ext_ts - det_ts).total_seconds() / 60, 1),
                    "max_reversion_spread": round(float(spreads[left] - ext_sp), 5),
                    "spread_eod":           round(float(eod_spread), 5),
                    "z_eod":                round(float(z_eod), 4) if not np.isnan(z_eod) else np.nan,
                    "session_elapsed_frac": round(max(0.0, min(1.0, (t_min - SESSION_START) / SESSION_LEN)), 4),
                    "reversion_frac":       round((z_det - ext_z) / (abs(z_det) + 1e-12), 4),
                    "fully_reverted":       bool(abs(ext_z) < 0.5),
                })
                left = resume_i  # advance past this event, keep scanning

        del df

    return pd.DataFrame(all_events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("preprocess")

    cfg = load_config(args.config)
    expiry_dates, trading_days = load_calendars(cfg)
    log.info(f"Loaded {len(expiry_dates)} expiries, {len(trading_days)} trading days")

    daily_vix = load_vix_daily(cfg["paths"]["raw_vix"], cfg["vix"]["rolling_1m"])
    log.info("VIX loaded" if daily_vix is not None else "No VIX — gate filter disabled")

    dist_out = Path(cfg["paths"]["processed"]) / "distributions"
    rev_out  = Path(cfg["paths"]["processed"]) / "reversion"
    dist_out.mkdir(parents=True, exist_ok=True)
    rev_out.mkdir(parents=True, exist_ok=True)

    for symbol in (cfg.get("symbols") or []):
        log.info(f"[{symbol}] scanning files...")
        contracts, equity_paths = scan_symbol_files(cfg, symbol)
        log.info(f"[{symbol}] {len(contracts)} contracts found")

        log.info(f"[{symbol}] loading equity...")
        eq_df = load_equity(equity_paths)
        log.info(f"[{symbol}] equity: {len(eq_df):,} ticks")

        log.info(f"[{symbol}] building distributions...")
        dist_df = build_distributions(symbol, contracts, eq_df, expiry_dates, trading_days, cfg, log)
        dist_df.to_parquet(dist_out / f"{symbol}_dist.parquet", index=False)
        log.info(f"[{symbol}] dist: {len(dist_df)} (dte, bucket) rows saved")

        log.info(f"[{symbol}] scanning reversion events...")
        rev_df = scan_reversion_events(symbol, contracts, eq_df, dist_df, expiry_dates, trading_days, cfg, daily_vix, log)
        rev_df.to_parquet(rev_out / f"{symbol}_reversion.parquet", index=False)
        n_rev  = (rev_df["event_type"] == "reversion").sum()    if len(rev_df) else 0
        n_cont = (rev_df["event_type"] == "continuation").sum() if len(rev_df) else 0
        n_div  = (rev_df["event_type"] == "divergence").sum()   if len(rev_df) else 0
        log.info(f"[{symbol}] {len(rev_df)} events — rev={n_rev} cont={n_cont} div={n_div}")

        del eq_df, dist_df, rev_df

    log.info("Done.")


if __name__ == "__main__":
    main()