"""
label_events.py — Apply (or re-apply) classification rules to the raw events
parquet produced by preprocess.py. Idempotent: existing event_type / split
columns are overwritten in place.
"""
import argparse, logging, shutil, yaml
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("label")


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify(df, cfg):
    """Vectorized classification.

    reversion    : extremum_z crossed event_exit_threshold toward zero AND
                   time_to_extremum_min >= min_dwell_min
    divergence   : most_extended_z went beyond z_det ± continuation_extra_z
    continuation : default — outlier persisted without committing in either
                   direction. Also catches fast snap-backs that crossed the
                   exit threshold but failed the dwell filter (these have no
                   separate label; they fall through to continuation).

    Priority: reversion > divergence > continuation. A snap-back that also
    tripped the divergence test is labeled continuation, not divergence —
    its defining feature is the rapid round-trip, not the other extreme of
    the window.
    """
    exit_thresh = cfg["detection"]["event_exit_threshold"]
    cont_extra  = cfg["detection"]["continuation_extra_z"]
    min_dwell   = float(cfg["detection"].get("min_dwell_min", 0))

    z_det = df["z_t_detection"].values
    rev_z = df["extremum_z"].values
    t_ext = df["time_to_extremum_min"].values
    positive = z_det > 0

    # Reversion test (raw — before dwell filter)
    rev_pos = positive & (rev_z <= exit_thresh)
    rev_neg = (~positive) & (rev_z >= -exit_thresh)
    is_rev_raw = rev_pos | rev_neg
    is_rev = is_rev_raw & (t_ext >= min_dwell)   # passed dwell → real reversion
    # Failed-dwell snap-backs get no explicit label; they fall through
    # to the "continuation" default below.

    # Divergence test
    if "most_extended_z" not in df.columns:
        log.warning("most_extended_z column missing — preprocess.py output is "
                    "stale. Divergence labels will be disabled. Re-run "
                    "preprocess.py to fix.")
        is_div = np.zeros(len(df), dtype=bool)
    else:
        ext_z = df["most_extended_z"].values
        div_pos = positive & (ext_z >= z_det + cont_extra)
        div_neg = (~positive) & (ext_z <= z_det - cont_extra)
        # A snap-back that also satisfied divergence stays continuation, not div.
        is_div = (div_pos | div_neg) & ~is_rev_raw

    labels = np.full(len(df), "continuation", dtype=object)  # default
    labels[is_div] = "divergence"
    labels[is_rev] = "reversion"   # real reversions override everything
    return labels


def label_symbol(cfg, symbol, dry_run=False):
    rev_path = (Path(cfg["paths"]["processed"])
                / "reversion" / f"{symbol}_reversion.parquet")
    if not rev_path.exists():
        log.error(f"[{symbol}] not found: {rev_path}")
        return

    df = pd.read_parquet(rev_path)
    if df.empty:
        log.warning(f"[{symbol}] empty parquet")
        return

    log.info(f"[{symbol}] {len(df)} raw events loaded")

    df["event_type"] = classify(df, cfg)
    holdout = cfg["training"]["holdout_year"]
    yr = pd.to_datetime(df["date"]).dt.year % 100
    df["split"] = np.where(yr == holdout, "test", "train")

    counts = df["event_type"].value_counts()
    n_rev  = int(counts.get("reversion", 0))
    n_cont = int(counts.get("continuation", 0))
    n_div  = int(counts.get("divergence", 0))
    n_tr   = int((df["split"] == "train").sum())
    n_te   = int((df["split"] == "test").sum())

    log.info(f"[{symbol}] rev={n_rev} ({n_rev/len(df):.1%}) "
             f"cont={n_cont} ({n_cont/len(df):.1%}) "
             f"div={n_div} ({n_div/len(df):.1%}) | "
             f"train={n_tr} test={n_te}")

    rev_only = df[df["event_type"] == "reversion"]
    if len(rev_only):
        log.info(f"[{symbol}] reversion timing: "
                 f"avg_t_extremum={rev_only['time_to_extremum_min'].mean():.1f}min "
                 f"median={rev_only['time_to_extremum_min'].median():.1f}min "
                 f"avg_max_rev_z={rev_only['max_reversion_z'].mean():.2f}")

    yr_breakdown = df.groupby([pd.to_datetime(df["date"]).dt.year,
                               "event_type"]).size().unstack(fill_value=0)
    log.info(f"[{symbol}] year x event:\n{yr_breakdown}")

    if dry_run:
        log.info(f"[{symbol}] DRY RUN — not writing")
        return

    backup = rev_path.with_suffix(".parquet.bak")
    shutil.copy2(rev_path, backup)
    df.to_parquet(rev_path, index=False)
    log.info(f"[{symbol}] wrote labels -> {rev_path} (backup: {backup.name})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--symbol", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    syms = [args.symbol] if args.symbol else (cfg.get("symbols") or [])
    for s in syms:
        try:
            label_symbol(cfg, s, dry_run=args.dry_run)
        except Exception as e:
            log.error(f"[{s}] FAILED: {type(e).__name__}: {e}", exc_info=True)


if __name__ == "__main__":
    main()