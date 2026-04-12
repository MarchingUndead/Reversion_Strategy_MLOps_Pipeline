"""
models/train.py — Ridge regression on the reversion events table.

Run without MLflow:  python models/train.py
Run with MLflow:     set mlflow.enabled: true in config.yaml, then
                     docker compose up  (or point tracking_uri at a live server)
"""
import argparse, logging, pickle, json, yaml
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

TARGETS    = ["reversion_frac", "time_to_extremum_min"]
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
CLIP_LO_Q  = 0.01
CLIP_HI_Q  = 0.99
N_CV_SPLITS = 5


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode_features(df, cols, bucket_categories):
    X = df[cols].copy()
    if "bucket" in X.columns:
        X["bucket"] = pd.Categorical(X["bucket"], categories=bucket_categories).codes
    if "direction" in X.columns:
        X["direction"] = (X["direction"] == "above").astype(int)
    for b in ("fully_reverted", "is_expiry_day"):
        if b in X.columns:
            X[b] = X[b].astype(int)
    return X.astype(float)


def baseline_r2(y_train, y_test):
    mu = y_train.mean()
    return {
        f"baseline_{t}_r2": round(float(r2_score(y_test[t], np.full(len(y_test), mu[t]))), 4)
        for t in TARGETS if t in y_test.columns and len(y_test)
    }


def load_events(cfg, symbol):
    rev_path = Path(cfg["paths"]["processed"]) / "reversion" / f"{symbol}_reversion.parquet"
    if not rev_path.exists():
        raise FileNotFoundError(f"Not found: {rev_path}")
    df = pd.read_parquet(rev_path)
    if "split" not in df.columns:
        raise RuntimeError(f"[{symbol}] no 'split' column — re-run label_events.py")
    df["symbol"] = symbol
    return df


def add_symbol_dummies(df, feature_cols):
    dummies = pd.get_dummies(df["symbol"], prefix="symbol", drop_first=True)
    for c in dummies.columns:
        df[c] = dummies[c].values
    return df, list(feature_cols) + list(dummies.columns)


def pick_alpha_walkforward(X_tr, y_tr, dates, alpha_grid, n_splits):
    order = np.argsort(dates.values)
    Xs = X_tr.iloc[order].reset_index(drop=True)
    ys = y_tr.iloc[order].reset_index(drop=True)
    splits = max(2, min(n_splits, len(Xs) // 50))
    tscv   = TimeSeriesSplit(n_splits=splits)
    scores = {}
    for a in alpha_grid:
        fold_r2 = []
        for tr_idx, te_idx in tscv.split(Xs):
            if len(tr_idx) < 10 or len(te_idx) < 5:
                continue
            pipe = Pipeline([("scaler", RobustScaler()),
                             ("ridge",  MultiOutputRegressor(Ridge(alpha=a)))])
            pipe.fit(Xs.iloc[tr_idx], ys.iloc[tr_idx])
            preds = pipe.predict(Xs.iloc[te_idx])
            fold_r2.append(float(np.mean(
                [r2_score(ys.iloc[te_idx, i], preds[:, i]) for i in range(len(TARGETS))]
            )))
        scores[a] = float(np.mean(fold_r2)) if fold_r2 else float("-inf")
    return max(scores, key=scores.get), scores


def compute_clip_bounds(y_tr):
    return {t: (float(np.nanquantile(y_tr[t], CLIP_LO_Q)),
                float(np.nanquantile(y_tr[t], CLIP_HI_Q)))
            for t in TARGETS}


def clip_preds(preds, bounds):
    out = preds.copy()
    for i, t in enumerate(TARGETS):
        out[:, i] = np.clip(out[:, i], *bounds[t])
    return out


def train(cfg, symbols, pooled=False):
    mcfg         = cfg["model"]
    feature_cols = list(mcfg["features"])
    save_dir     = Path(mcfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics").mkdir(exist_ok=True)

    frames = []
    for s in symbols:
        d = load_events(cfg, s)
        log.info(f"[{s}] {len(d)} events loaded")
        frames.append(d)
    df  = pd.concat(frames, ignore_index=True)
    tag = "POOLED" if pooled else symbols[0]
    log.info(f"[{tag}] {len(df)} total events; classes={df['event_type'].value_counts().to_dict()}")

    if pooled:
        df, feature_cols = add_symbol_dummies(df, feature_cols)

    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning(f"[{tag}] missing features dropped: {missing}")
    if not available:
        raise RuntimeError(f"[{tag}] no usable features")
    log.info(f"[{tag}] using {len(available)} features: {available}")

    bucket_cats = [b[0] for b in cfg["time_buckets"]]
    train_df    = df[df["split"] == "train"].copy()
    test_df     = df[df["split"] == "test"].copy()
    log.info(f"[{tag}] train={len(train_df)} test={len(test_df)}")
    if train_df.empty:
        raise RuntimeError(f"[{tag}] empty train split")

    X_tr = encode_features(train_df, available, bucket_cats)
    y_tr = train_df[TARGETS].astype(float)
    mask = X_tr.notna().all(axis=1) & y_tr.notna().all(axis=1)
    X_tr, y_tr = X_tr[mask], y_tr[mask]
    train_dates = pd.to_datetime(train_df.loc[mask, "date"])
    log.info(f"[{tag}] post-NaN train rows: {len(X_tr)}")
    if X_tr.empty:
        raise RuntimeError(f"[{tag}] all train rows dropped after NaN filter")

    log.info(f"[{tag}] walk-forward CV over alphas {ALPHA_GRID}")
    best_alpha, cv_scores = pick_alpha_walkforward(X_tr, y_tr, train_dates, ALPHA_GRID, N_CV_SPLITS)
    log.info(f"[{tag}] CV mean R² by alpha: { {a: round(s,4) for a,s in cv_scores.items()} }")
    log.info(f"[{tag}] selected alpha={best_alpha}")

    model = Pipeline([("scaler", RobustScaler()),
                      ("ridge",  MultiOutputRegressor(Ridge(alpha=best_alpha)))])
    model.fit(X_tr, y_tr)

    clip_bounds = compute_clip_bounds(y_tr)
    log.info(f"[{tag}] clip bounds: { {k: (round(v[0],3), round(v[1],3)) for k,v in clip_bounds.items()} }")

    metrics = {
        "label": tag, "symbols": symbols, "pooled": pooled,
        "n_train": int(len(X_tr)),
        "n_test":  int(len(test_df)),
        "features": available,
        "alpha": best_alpha,
        "cv_scores": {str(a): round(s, 4) for a, s in cv_scores.items()},
        "clip_bounds": {k: [round(v[0],4), round(v[1],4)] for k,v in clip_bounds.items()},
        "timestamp": datetime.utcnow().isoformat(),
    }

    X_te = encode_features(test_df, available, bucket_cats) if len(test_df) else None
    y_te = test_df[TARGETS].astype(float)                   if len(test_df) else None

    for split, X, y in [("train", X_tr, y_tr), ("test", X_te, y_te)]:
        if X is None or X.empty:
            continue
        mask_s = X.notna().all(axis=1) & y.notna().all(axis=1)
        X, y   = X[mask_s], y[mask_s]
        if X.empty:
            continue
        pr_raw = model.predict(X)
        pr_clp = clip_preds(pr_raw, clip_bounds)
        for i, t in enumerate(TARGETS):
            r2_raw = round(float(r2_score(y.iloc[:, i], pr_raw[:, i])), 4)
            r2_clp = round(float(r2_score(y.iloc[:, i], pr_clp[:, i])), 4)
            mae    = round(float(mean_absolute_error(y.iloc[:, i], pr_clp[:, i])), 4)
            metrics[f"{split}_{t}_r2_raw"] = r2_raw
            metrics[f"{split}_{t}_r2"]     = r2_clp
            metrics[f"{split}_{t}_mae"]    = mae
            log.info(f"  {split} {t}: R²(raw)={r2_raw}  R²(clipped)={r2_clp}  MAE={mae}")

    if y_te is not None and len(y_te):
        metrics.update(baseline_r2(y_tr, y_te))
        for k, v in metrics.items():
            if k.startswith("baseline_"):
                log.info(f"  {k}: {v}")

    fname = "POOLED_ridge.pkl" if pooled else f"{symbols[0]}_ridge.pkl"
    pkl   = save_dir / fname
    with open(pkl, "wb") as f:
        pickle.dump({"model": model, "features": available,
                     "clip_bounds": clip_bounds, "metrics": metrics}, f)
    log.info(f"[{tag}] model -> {pkl}")

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    mpath = save_dir / "metrics" / f"{tag}_{stamp}.json"
    mpath.write_text(json.dumps(metrics, indent=2))
    log.info(f"[{tag}] metrics -> {mpath}")

    return model, metrics, str(pkl)


def train_with_mlflow(cfg, symbols, pooled=False):
    import mlflow
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://mlflow:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment", "basis-reversion"))

    tag = "POOLED" if pooled else symbols[0]
    with mlflow.start_run(run_name=tag):
        mlflow.set_tag("symbol", tag)
        mcfg = cfg["model"]
        mlflow.log_params({
            "symbol":       tag,
            "features":     ",".join(mcfg["features"]),
            "alpha_grid":   str(ALPHA_GRID),
            "holdout_year": cfg["training"]["holdout_year"],
            "pooled":       pooled,
        })

        model, metrics, pkl_path = train(cfg, symbols, pooled=pooled)

        # log only scalar metrics
        mlflow.log_metrics({
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        })
        mlflow.log_artifact(pkl_path, artifact_path="model")

    return model, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--symbol", default=None)
    p.add_argument("--pooled", action="store_true")
    args = p.parse_args()
    cfg  = load_config(args.config)

    use_mlflow = cfg.get("mlflow", {}).get("enabled", False)
    runner     = train_with_mlflow if use_mlflow else lambda c, s, pooled: train(c, s, pooled)

    if args.pooled:
        syms = cfg.get("symbols") or []
        if not syms:
            log.error("--pooled requires symbols in config"); return
        try:
            runner(cfg, syms, pooled=True)
        except Exception as e:
            log.error(f"[POOLED] FAILED: {type(e).__name__}: {e}", exc_info=True)
    else:
        syms = [args.symbol] if args.symbol else (cfg.get("symbols") or [])
        for s in syms:
            try:
                runner(cfg, [s], pooled=False)
            except Exception as e:
                log.error(f"[{s}] FAILED: {type(e).__name__}: {e}", exc_info=True)


if __name__ == "__main__":
    main()