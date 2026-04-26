import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load(open(ROOT / "config.yaml"))

raw_path       = ROOT / _cfg["paths"]["raw_root"]
train_path     = ROOT / _cfg["paths"]["train_root"]
# test_path holds evaluation-only ticks (2025 + rolling early-2026 near-month
# contracts). Ingested here so the expanding distribution stats continue across
# 2022→2025 in the same shards. Holdout discipline is enforced at evaluation
# time by `split_events("train")` in src/model.py and the --confirm-holdout
# flag on backtest.py.
test_path      = ROOT / _cfg["paths"]["test_root"]
dates_path     = ROOT / _cfg["paths"]["dates"]
processed_path = ROOT / _cfg["paths"]["processed"]

symbols              = _cfg["symbols"]
months               = _cfg["months"]
train_years          = _cfg["train_years"]
test_years           = _cfg.get("test_years", [])
test_contracts_extra = _cfg.get("test_contracts_extra", [])

columns_fut = _cfg["schemas"]["fut"]
#sample column : 20220128,09:17:34,6994.60,125,500,6986.30,125,7030.70,125

columns_eq = _cfg["schemas"]["eq"]
#sample column : 20220103 09:15:00 2364.00 	57496	0	2363.05	 235	 2364.00 	199

columns_vix = _cfg["schemas"]["vix"]
#sample column :20250101,09:15:05,14.39,0,0

# create dataframes corresponding to dte x time bucket : expected 90 x 7 = 630 approximate
trading_cal = pd.read_csv(dates_path / "trading_calendar.csv", parse_dates=["date"])
expiry_dates = pd.read_csv(dates_path / "expiry_dates.csv", parse_dates=["expiry_date"])
trading_days_set = set(trading_cal["date"])

#make 1 hour buckets. the time corresponds to bucket starts
# 915-10,10-11,11-12,12-13,13-14,14-15,15-1530
buckets = _cfg["buckets"]

# sorted trading calendar and expiry list for lookup
trading_days_sorted = sorted(trading_days_set)
expiry_list = sorted(expiry_dates["expiry_date"])


def find_first_gte(arr, target):
    """Binary search: return index of the first element >= target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def find_exact(arr, target):
    """Binary search for exact match. Returns index or None."""
    idx = find_first_gte(arr, target)
    if idx < len(arr) and arr[idx] == target:
        return idx
    return None


def get_dte(day_str):
    """Calculate DTE = trading days from cur to the nearest expiry >= cur.

    1. Parse day to Timestamp.
    2. Binary search expiry_list for nearest expiry >= cur.
    3. Binary search trading_days_sorted for index of cur.
    4. Binary search trading_days_sorted for index of expiry.
       DTE = index(expiry) - index(cur).
       (cur is on expiry day itself -> DTE = 0.)
    """
    cur = pd.Timestamp(str(day_str))

    # nearest expiry >= cur
    ei = find_first_gte(expiry_list, cur)
    if ei >= len(expiry_list):
        return np.nan
    expiry = expiry_list[ei]

    # binary search trading calendar for both dates
    idx_cur = find_exact(trading_days_sorted, cur)
    idx_exp = find_exact(trading_days_sorted, expiry)

    if idx_cur is None or idx_exp is None:
        return np.nan

    return idx_exp - idx_cur


def get_bucket(time_str):
    """Assign a bucket index (0, 1, 2, ...) based on bucket start times.

    Bucket i covers [buckets[i], buckets[i+1]).
    The last bucket covers [15:00:00, end of day).
    """
    t = pd.Timedelta(time_str)
    bucket_td = [pd.Timedelta(b) for b in buckets]
    for i in range(len(bucket_td) - 1, -1, -1):
        if t >= bucket_td[i]:
            return i
    return 0


# vix function
# read INDIAVIX.csv from train path (same headerless format: day, time, ltp, ...)
df_vix = pd.read_csv(train_path / _cfg["preprocess"]["vix_file"], header=None, names=columns_vix)

vix_thresh = _cfg["preprocess"]["vix_thresh"]  # set vix threshold


def get_high_vix_days(df_vix, threshold=vix_thresh):
    """Return a set of days where VIX crossed the threshold at any tick.

    For each unique day in the VIX data, check all ticks from that day.
    If the threshold is crossed even once, the whole day is flagged.
    """
    high_vix_days = set()
    for day, day_ticks in df_vix.groupby("day"):
        if (day_ticks["vix"] >= threshold).any():
            high_vix_days.add(day)
    return high_vix_days


high_vix_days = get_high_vix_days(df_vix)
# Union with test-period VIX so both passes apply the same exclusion set.
if test_years:
    test_vix_file = test_path / _cfg["preprocess"]["vix_file"]
    if test_vix_file.exists():
        df_vix_test = pd.read_csv(test_vix_file, header=None, names=columns_vix)
        high_vix_days = high_vix_days | get_high_vix_days(df_vix_test)


def store_group(df_group, symbol, dte, bucket):
    """Append a DataFrame chunk to the appropriate CSV.

    File naming: {symbol}_{dte}_{bucket}.csv
    Writes header on first write; appends without header thereafter.
    """
    csv_path = processed_path / f"{symbol}_{int(dte)}_{int(bucket)}.csv"
    header = not csv_path.exists()
    df_group.to_csv(csv_path, mode="a", header=header, index=False)


if __name__ == "__main__":
    print(f"{len(high_vix_days)} days excluded out of {df_vix['day'].nunique()} total (VIX >= {vix_thresh})")

    # processed data output directory
    processed_path.mkdir(parents=True, exist_ok=True)

    # clear old processed CSVs to avoid duplicates from re-runs
    for old_csv in processed_path.glob("*.csv"):
        old_csv.unlink()

    for sym in symbols:
        for yr in train_years:
            for mon in months:
                # locate files
                eq_file = train_path / f"{sym}.csv"
                fut_file = train_path / f"{sym}{yr}{mon}FUT.csv"
                if not fut_file.exists():
                    continue

                # read tick data (headerless CSVs)
                df_eq = pd.read_csv(eq_file, header=None, names=columns_eq)
                df_fut = pd.read_csv(fut_file, header=None, names=columns_fut)

                # create timestamp by merging day and time
                df_eq["timestamp"] = pd.to_datetime(df_eq["day"].astype(str) + " " + df_eq["time"])
                df_fut["timestamp"] = pd.to_datetime(df_fut["day"].astype(str) + " " + df_fut["time"])

                # sort by timestamp (required for merge_asof)
                #df_eq = df_eq.sort_values("timestamp")
                #df_fut = df_fut.sort_values("timestamp")

                # merge_asof: for each futures tick, grab the nearest prior equity tick
                df = pd.merge_asof(df_fut, df_eq, on="timestamp", direction="backward", suffixes=("_fut", "_eq"))

                # filter out high VIX days
                df = df[~df["day_fut"].isin(high_vix_days)]

                # compute DTE via binary search on trading calendar and expiry dates
                df["dte"] = df["day_fut"].apply(get_dte)

                # compute bucket per row
                df["bucket"] = df["time_fut"].apply(get_bucket)

                # tag each tick with its source contract month (e.g. "JAN")
                df["contract"] = mon

                # drop rows where DTE could not be computed
                df = df.dropna(subset=["dte"])

                # group by (dte, bucket) and append each group to its CSV
                for (dte, bucket), grp in df.groupby(["dte", "bucket"]):
                    store_group(grp, sym, dte, bucket)

                n_csvs = df.groupby(["dte", "bucket"]).ngroups
                print(f"{sym}{yr}{mon}FUT — {len(df)} rows saved across {n_csvs} CSVs")

    # Test pass: same body as the loop above, only the iteration arguments
    # change (test_years/test_path instead of train_years/train_path, plus the
    # rolling 26JAN/FEB/MAR contracts that live alongside the 2025 files).
    # Shards are unified — the expanding-stats pass below recomputes over the
    # combined per-shard series.
    for sym in symbols:
        for (yr, mon_list) in [(y, months)             for y in test_years] + \
                              [("26", test_contracts_extra)]:
            for mon in mon_list:
                # locate files
                eq_file = test_path / f"{sym}.csv"
                fut_file = test_path / f"{sym}{yr}{mon}FUT.csv"
                if not fut_file.exists():
                    continue

                # read tick data (headerless CSVs)
                df_eq = pd.read_csv(eq_file, header=None, names=columns_eq)
                df_fut = pd.read_csv(fut_file, header=None, names=columns_fut)

                # create timestamp by merging day and time
                df_eq["timestamp"] = pd.to_datetime(df_eq["day"].astype(str) + " " + df_eq["time"])
                df_fut["timestamp"] = pd.to_datetime(df_fut["day"].astype(str) + " " + df_fut["time"])

                # merge_asof: for each futures tick, grab the nearest prior equity tick
                df = pd.merge_asof(df_fut, df_eq, on="timestamp", direction="backward", suffixes=("_fut", "_eq"))

                # filter out high VIX days
                df = df[~df["day_fut"].isin(high_vix_days)]

                # compute DTE via binary search on trading calendar and expiry dates
                df["dte"] = df["day_fut"].apply(get_dte)

                # compute bucket per row
                df["bucket"] = df["time_fut"].apply(get_bucket)

                # tag each tick with its source contract month (e.g. "JAN")
                df["contract"] = mon

                # drop rows where DTE could not be computed
                df = df.dropna(subset=["dte"])

                # group by (dte, bucket) and append each group to its CSV
                for (dte, bucket), grp in df.groupby(["dte", "bucket"]):
                    store_group(grp, sym, dte, bucket)

                n_csvs = df.groupby(["dte", "bucket"]).ngroups
                print(f"{sym}{yr}{mon}FUT — {len(df)} rows saved across {n_csvs} CSVs")

    # ---- expanding distribution pass ----
    warmup_months = _cfg["preprocess"]["warmup_months"]

    # prepare VIX with timestamps for merge_asof
    df_vix["timestamp"] = pd.to_datetime(df_vix["day"].astype(str) + " " + df_vix["time"])
    df_vix_sorted = df_vix[["timestamp", "vix"]].sort_values("timestamp").reset_index(drop=True)

    # sort by (symbol, dte, bucket) — filename is {symbol}_{dte}_{bucket}.csv
    csv_files = sorted(
        processed_path.glob("*_*_*.csv"),
        key=lambda p: (p.stem.rsplit("_", 2)[0], int(p.stem.rsplit("_", 2)[1]), int(p.stem.rsplit("_", 2)[2]))
    )
    print(f"Processing {len(csv_files)} (symbol, dte, bucket) CSVs...")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # drop columns from partial prior runs to avoid duplicates
        df = df.drop(columns=["spread", "dist_mean", "dist_std", "dist_count", "vix", "vix_x", "vix_y"], errors="ignore")

        # drop 2021 rows and rows with missing equity data
        df = df[df["day_fut"] >= 20220101]
        df = df.dropna(subset=["ltp"])

        # re-apply VIX day filter (original run had broken VIX columns)
        df = df[~df["day_fut"].isin(high_vix_days)]

        if df.empty:
            csv_path.unlink()
            print(f"{csv_path.name}: deleted (empty after filtering)")
            continue

        # Sort by timestamp before the expanding pass so that .shift(1) really
        # means "rows strictly before me in time," not just "rows that landed
        # earlier in file-append order." Multiple contracts contribute to the
        # same shard and the train + test loops above don't guarantee strict
        # chronological append order.
        df = df.sort_values("timestamp").reset_index(drop=True)

        # normalized basis spread (%)
        df["spread"] = ((df["fut_ltp"] - df["ltp"]) / df["ltp"]) * 100

        # expanding stats shifted by 1: row i gets stats from rows 0..i-1
        expanding = df["spread"].expanding()
        df["dist_mean"] = expanding.mean().shift(1)
        df["dist_std"] = expanding.std().shift(1)
        df["dist_count"] = expanding.count().shift(1)

        # mask the 3-month warm-up period
        cutoff = df["timestamp"].iloc[0] + relativedelta(months=warmup_months)
        warmup_mask = df["timestamp"] < cutoff
        df.loc[warmup_mask, ["dist_mean", "dist_std", "dist_count"]] = np.nan

        # write back to the same processed CSV
        df.to_csv(csv_path, index=False)

        print(f"{csv_path.name}: {len(df)} rows, {(~warmup_mask).sum()} with distributions")
