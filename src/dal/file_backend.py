"""
dal/file_backend.py — File/SSD implementation of the DAL.

Handles the directory layout where futures, equity, and VIX coexist
in the same directories, differentiated by filename pattern:
  Futures: {SYMBOL}{YY}{MMM}FUT.csv
  Equity:  {SYMBOL}.csv
  VIX:     INDIAVIX.csv
"""

import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.dal.base import (
    TickSource, CandleStore, TableStore, CalendarSource, DataBackend,
)

logger = logging.getLogger(__name__)

_FUTURES_PATTERN = re.compile(r"^([A-Z&]+)(\d{2}[A-Z]{3})FUT\.csv$", re.IGNORECASE)


# ── shared CSV reader ────────────────────────────────────────────────

def read_raw_csv(path):
    """Read headerless tick CSV. Auto-detects 5-col vs 9-col (BidAsk)."""
    df = pd.read_csv(path, sep=None, engine="python", header=None)
    ncols = len(df.columns)

    if ncols == 5:
        df.columns = ["date", "time", "price", "volume", "oi"]
    elif ncols == 9:
        df.columns = ["date", "time", "price", "volume", "oi",
                       "bid_price", "bid_volume", "ask_price", "ask_volume"]
    elif ncols > 5:
        base = ["date", "time", "price", "volume", "oi"]
        df.columns = base + [f"_extra_{i}" for i in range(ncols - 5)]
    else:
        raise ValueError(f"{path}: expected >=5 columns, got {ncols}")

    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["timestamp"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"].astype(str))

    keep = ["timestamp", "price", "volume", "oi"]
    if ncols >= 9:
        keep += ["bid_price", "bid_volume", "ask_price", "ask_volume"]
    return df[keep].sort_values("timestamp")


def _normalize_paths(cfg_value):
    if cfg_value is None: return []
    if isinstance(cfg_value, str): cfg_value = [cfg_value]
    if not isinstance(cfg_value, list): return []
    return [Path(str(p)) for p in cfg_value if Path(str(p)).exists()]


def _normalize_file_paths(cfg_value):
    if cfg_value is None: return []
    if isinstance(cfg_value, str): cfg_value = [cfg_value]
    if not isinstance(cfg_value, list): return []
    return [Path(str(p)) for p in cfg_value if Path(str(p)).exists()]


# ── TickSource ───────────────────────────────────────────────────────

class FileTickSource(TickSource):

    def __init__(self, cfg):
        self._futures_dirs = _normalize_paths(cfg.paths.raw_futures)
        self._equity_dirs = _normalize_paths(cfg.paths.raw_equity)
        self._vix_paths = _normalize_file_paths(cfg.paths.raw_vix)

    def list_contracts(self, symbols=None, year_filter=None):
        # Build equity lookup: symbol → [path, ...]
        equity_map = {}
        for eq_dir in self._equity_dirs:
            for f in sorted(eq_dir.glob("*.csv")):
                if _FUTURES_PATTERN.match(f.name): continue
                if f.stem.upper() == "INDIAVIX": continue
                equity_map.setdefault(f.stem.upper(), []).append(str(f))

        results, seen = [], set()
        for fut_dir in self._futures_dirs:
            if not fut_dir.exists(): continue
            for fpath in sorted(fut_dir.glob("*.csv")):
                m = _FUTURES_PATTERN.match(fpath.name)
                if not m: continue
                symbol, contract = m.group(1).upper(), m.group(2).upper()
                key = (symbol, contract)
                if key in seen: continue
                if year_filter and contract[:2] not in year_filter: continue
                if symbols and symbol not in [s.upper() for s in symbols]: continue
                eq_files = equity_map.get(symbol)
                if not eq_files: continue
                seen.add(key)
                results.append({"symbol": symbol, "contract": contract,
                                "futures_path": str(fpath), "equity_paths": eq_files})

        results.sort(key=lambda r: (r["symbol"], r["contract"]))
        return results

    def read_futures_ticks(self, futures_path):
        return read_raw_csv(futures_path)

    def read_equity_ticks(self, equity_paths):
        if isinstance(equity_paths, str): equity_paths = [equity_paths]
        parts = []
        for p in equity_paths:
            try:
                part = read_raw_csv(p)[["timestamp", "price"]].rename(
                    columns={"price": "cash_price"})
                parts.append(part)
            except Exception as e:
                logger.warning(f"Failed to read equity {p}: {e}")
        if not parts:
            raise RuntimeError(f"No readable equity files from: {equity_paths}")
        eq = pd.concat(parts, ignore_index=True)
        return eq.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    def read_vix_ticks(self):
        if not self._vix_paths: return None
        parts = []
        for vp in self._vix_paths:
            try:
                df = pd.read_csv(vp, sep=None, engine="python", header=None,
                                 names=["date", "time", "vix", "volume", "oi"])
                df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
                df["timestamp"] = pd.to_datetime(
                    df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"].astype(str))
                parts.append(df[["timestamp", "vix"]])
            except Exception as e:
                logger.warning(f"Failed to read VIX {vp}: {e}")
        if not parts: return None
        return (pd.concat(parts, ignore_index=True)
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True))


# ── CandleStore ──────────────────────────────────────────────────────

class FileCandleStore(CandleStore):
    """Layout: {processed}/{SYMBOL}/{SYMBOL}{YY}{MMM}FUT/candles_{N}min.parquet"""

    def __init__(self, cfg):
        self._dir = Path(cfg.paths.processed)
        self._fmt = cfg.get("save", {}).get("format", "parquet")

    def _path(self, symbol, contract, interval):
        return self._dir / symbol / f"{symbol}{contract}FUT" / f"candles_{interval}min.{self._fmt}"

    def read_candles(self, symbol, contract, interval):
        p = self._path(symbol, contract, interval)
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
        df = pd.read_parquet(p) if self._fmt == "parquet" else pd.read_csv(p, index_col="timestamp", parse_dates=True)
        if "timestamp" in df.columns: df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"
        return df.sort_index()

    def write_candles(self, df, symbol, contract, interval):
        p = self._path(symbol, contract, interval)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self._fmt == "parquet": df.to_parquet(p, engine="pyarrow")
        else: df.to_csv(p)

    def exists(self, symbol, contract, interval):
        return self._path(symbol, contract, interval).exists()

    def list_symbols(self):
        if not self._dir.exists(): return []
        return sorted(d.name for d in self._dir.iterdir()
                      if d.is_dir() and not d.name.startswith("_"))

    def list_contracts_for_symbol(self, symbol):
        sym_dir = self._dir / symbol
        if not sym_dir.exists(): return []
        pat = re.compile(rf"^{re.escape(symbol)}(\d{{2}}[A-Z]{{3}})FUT$", re.IGNORECASE)
        return sorted(m.group(1).upper() for d in sym_dir.iterdir()
                      if d.is_dir() and (m := pat.match(d.name)))


# ── TableStore ───────────────────────────────────────────────────────

class FileTableStore(TableStore):

    def __init__(self, cfg):
        self._dir = Path(cfg.paths.dist_tables)

    def _ensure(self): self._dir.mkdir(parents=True, exist_ok=True)

    def read_dist_table(self, symbol):
        p = self._dir / f"{symbol}_distr.parquet"
        if not p.exists(): raise FileNotFoundError(p)
        return pd.read_parquet(p)

    def write_dist_table(self, df, symbol):
        self._ensure()
        df.to_parquet(self._dir / f"{symbol}_distr.parquet", index=False)

    def read_reversion_table(self, symbol):
        p = self._dir / f"{symbol}_reversion.parquet"
        if not p.exists(): raise FileNotFoundError(p)
        return pd.read_parquet(p)

    def write_reversion_table(self, df, symbol):
        self._ensure()
        df.to_parquet(self._dir / f"{symbol}_reversion.parquet", index=False)

    def dist_table_exists(self, symbol):
        return (self._dir / f"{symbol}_distr.parquet").exists()

    def reversion_table_exists(self, symbol):
        return (self._dir / f"{symbol}_reversion.parquet").exists()


# ── CalendarSource ───────────────────────────────────────────────────

class FileCalendarSource(CalendarSource):

    def __init__(self, cfg):
        self._expiry_csv = Path(cfg.paths.expiry_csv)
        self._trading_csv = Path(cfg.paths.trading_csv)

    def get_expiry_dates(self):
        df = pd.read_csv(self._expiry_csv)
        for col in df.columns:
            if "date" in col.lower() or "expiry" in col.lower():
                df.rename(columns={col: "expiry_date"}, inplace=True)
                break
        df["expiry_date"] = pd.to_datetime(df["expiry_date"])
        return sorted(df["expiry_date"].unique())

    def get_trading_days(self):
        df = pd.read_csv(self._trading_csv, parse_dates=["date"])
        return pd.DatetimeIndex(sorted(df["date"]))


# ── Factory ──────────────────────────────────────────────────────────

def create_file_backend(cfg):
    return DataBackend(
        ticks=FileTickSource(cfg),
        candles=FileCandleStore(cfg),
        tables=FileTableStore(cfg),
        calendars=FileCalendarSource(cfg),
    )