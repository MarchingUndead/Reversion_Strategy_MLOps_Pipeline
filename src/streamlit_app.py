"""Streamlit UI for the reversion pipeline.

Four tabs:
  1. Event explorer    — run events.run() for a (symbol, year, month, day)
                          and render the per-day overlay.
  2. Sample prediction — enter feature values by hand; load the trained
                          models from ./models/*.pkl; show predictions.
  3. Backtest viewer   — load persisted trade logs and render the equity
                          curve + per-position summary.
  4. MLflow Serving    — POST a feature row to a running `mlflow models serve`
                          endpoint and display the returned prediction.

Models are pickled by `python src/model.py` into ./models/. No registry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FEATURE_COLS    = _cfg["model"]["feature_cols"]
POSITIONS       = _cfg["backtest"]["positions"]
POSITION_LABELS = _cfg["backtest"]["position_labels"]
MONTHS          = _cfg["months"]
OUT_THRESH      = _cfg["events"]["out_thresh"]
LOT_SIZE        = _cfg["backtest"]["lot_size"]
ORDER_SIZE      = _cfg["backtest"]["order_size"]
TXN_COST_RATE   = _cfg["backtest"]["txn_cost_rate"]

st.set_page_config(page_title="Reversion Strategy", layout="wide")
st.title("Reversion Strategy")

tab1, tab2, tab3, tab4 = st.tabs([
    "Event explorer",
    "Live tick prediction",
    "Backtest viewer",
    "MLflow Serving",
])


# ------------------------------ tab 1 ------------------------------
with tab1:
    st.header("Event explorer")
    c1, c2, c3, c4 = st.columns(4)
    symbol    = c1.selectbox("symbol", _cfg["symbols"], key="t1_symbol")
    # 2025 is the hidden test holdout and is intentionally excluded here.
    year      = c2.number_input("year", min_value=2022, max_value=2024, value=2024, step=1, key="t1_year")
    month_str = c3.selectbox("month", MONTHS, index=7, key="t1_month")
    day_input = c4.text_input("day (1–31 or YYYYMMDD, blank = full month)", value="", key="t1_day")

    if st.button("Run events.run()", type="primary", key="t1_submit"):
        with st.spinner("loading processed data and detecting events..."):
            from events import run
            day = int(day_input) if day_input.strip() else None
            try:
                events_df = run(symbol, int(year), month_str, day=day)
                st.success(f"{len(events_df)} events detected")
                if not events_df.empty:
                    by_klass = events_df.groupby(["contract", "klass"]).size().unstack(fill_value=0)
                    st.dataframe(by_klass, use_container_width=True)
                    st.dataframe(events_df, use_container_width=True, height=400)
            except Exception as e:
                st.error(f"events.run failed: {e}")


# ------------------------------ tab 2 ------------------------------
with tab2:
    st.header("Live tick prediction")
    st.caption("Paste a live tick. If |z| ≤ out_thresh the tick is not an outlier → no action. "
               "Otherwise the frozen classifier runs and a position is suggested.")

    # ---- session context ----
    st.subheader("Session context")
    c1, c2, c3, c4 = st.columns(4)
    symbol   = c1.selectbox("symbol", _cfg["symbols"], key="t2_symbol")
    position = c2.selectbox(
        "contract position",
        list(zip(POSITIONS, POSITION_LABELS)),
        format_func=lambda t: f"{t[0]} ({t[1]})",
    )[0]
    dte      = c3.number_input("dte (days to expiry)", min_value=0, max_value=60, value=10, step=1)
    time_str = c4.text_input("time (HH:MM:SS)", value="10:30:00")

    # ---- futures leg ----
    st.subheader("Futures leg")
    c1, c2, c3, c4 = st.columns(4)
    fut_ltp      = c1.number_input("fut_ltp",      value=100.50, format="%.4f")
    fut_ltq      = c2.number_input("fut_ltq",      value=50,     step=1)
    oi_fut       = c3.number_input("oi_fut",       value=150000, step=100)
    fut_bidprice = c4.number_input("fut_bidprice", value=100.45, format="%.4f")
    c1, c2, c3, c4 = st.columns(4)
    fut_bidvol   = c1.number_input("fut_bidvol",   value=100,    step=1)
    fut_askprice = c2.number_input("fut_askprice", value=100.55, format="%.4f")
    fut_askvol   = c3.number_input("fut_askvol",   value=100,    step=1)

    # ---- cash leg ----
    st.subheader("Cash leg")
    c1, c2, c3, c4 = st.columns(4)
    eq_ltp      = c1.number_input("eq_ltp (cash)",  value=100.00, format="%.4f")
    eq_ltq      = c2.number_input("eq_ltq",         value=100,    step=1)
    oi_eq       = c3.number_input("oi (cash)",      value=0,      step=1)
    eq_bidprice = c4.number_input("eq_bidprice",    value=99.97,  format="%.4f")
    c1, c2, c3, c4 = st.columns(4)
    eq_bidvol   = c1.number_input("eq_bidvol",      value=200,    step=1)
    eq_askprice = c2.number_input("eq_askprice",    value=100.03, format="%.4f")
    eq_askvol   = c3.number_input("eq_askvol",      value=200,    step=1)

    # ---- distribution reference ----
    st.subheader("Distribution reference (expanding stats at this dte + bucket)")
    st.caption("In live use, these are maintained incrementally per (symbol, dte, bucket). "
               "For ad-hoc testing, copy the latest dist_mean/std/count from the matching "
               "`data/processed/{symbol}_{dte}_{bucket}.csv`.")
    c1, c2, c3 = st.columns(3)
    dist_mean  = c1.number_input("dist_mean",  value=0.20, format="%.6f")
    dist_std   = c2.number_input("dist_std",   value=0.15, format="%.6f")
    dist_count = c3.number_input("dist_count", value=500,  step=1)

    # ---- go ----
    if st.button("Compute features + predict", type="primary"):
        try:
            import numpy as np
            #from model import load_models
            from preprocess import get_bucket

            # ---- derive features ----
            spread   = ((fut_ltp - eq_ltp) / eq_ltp) * 100 if eq_ltp else 0.0
            z_score  = (spread - dist_mean) / dist_std if dist_std else 0.0
            side     = 1 if z_score > 0 else -1
            bucket   = get_bucket(time_str)
            fut_ba   = fut_askprice - fut_bidprice
            eq_ba    = eq_askprice - eq_bidprice

            st.subheader("Derived features")
            r1 = st.columns(4)
            r1[0].metric("spread (%)", f"{spread:.4f}")
            r1[1].metric("z_score",    f"{z_score:+.3f}")
            r1[2].metric("|z|",        f"{abs(z_score):.3f}")
            r1[3].metric("outlier?",   "YES" if abs(z_score) > OUT_THRESH else "no",
                         help=f"out_thresh = {OUT_THRESH}")
            r2 = st.columns(4)
            r2[0].metric("side",     f"+1 (spread > μ)" if side == 1 else "-1 (spread < μ)")
            r2[1].metric("bucket",   str(bucket))
            r2[2].metric("fut bid-ask", f"{fut_ba:.4f}")
            r2[3].metric("eq bid-ask",  f"{eq_ba:.4f}")

            # ---- outlier gate ----
            if abs(z_score) <= OUT_THRESH:
                st.info(
                    f"**Not an outlier** (|z| = {abs(z_score):.3f} ≤ out_thresh = {OUT_THRESH}). "
                    f"**Action: do nothing.**"
                )
            else:
                # ---- load models from mlflow+ predict ----
                #trio = load_models(positions=[position]).get(position)
                trio = mlflow.pyfunc.load_model
                if trio is None:
                    st.error(
                        f"No pickled models for position {position}. "
                        f"Run `python src/model.py` first to freeze a model."
                    )
                else:
                    clf, reg_dur, reg_rev = trio
                    feat = {
                        "det_z_score":    z_score,
                        "det_spread":     spread,
                        "side":           side,
                        "dte":            dte,
                        "bucket":         bucket,
                        "det_dist_std":   dist_std,
                        "det_dist_count": dist_count,
                        "det_fut_ltq":    fut_ltq,
                        "det_oi_fut":     oi_fut,
                        "det_ltq":        eq_ltq,
                        "det_fut_ba":     fut_ba,
                        "det_eq_ba":      eq_ba,
                    }
                    X = np.array([[feat[f] for f in FEATURE_COLS]], dtype=float)

                    pred_klass = str(clf.predict(X)[0])
                    pred_dur   = float(reg_dur.predict(X)[0])
                    pred_rev   = float(reg_rev.predict(X)[0])

                    direction  = {"reversion": 1, "divergence": -1, "continuation": 0}.get(pred_klass, 0)
                    sign       = direction * side

                    st.subheader("Model prediction")
                    p1 = st.columns(3)
                    p1[0].metric("pred_klass",          pred_klass)
                    p1[1].metric("duration_sec (est)",  f"{pred_dur:,.0f}")
                    p1[2].metric("revert_delta (est)",  f"{pred_rev:+.3f}")

                    # ---- suggested position ----
                    st.subheader("Suggested position")
                    V         = LOT_SIZE * ORDER_SIZE
                    notional  = V * (fut_ltp + eq_ltp)
                    est_cost  = TXN_COST_RATE * notional
                    if sign == 0:
                        st.warning(
                            f"pred_klass = `{pred_klass}` → **no trade** "
                            f"(continuation or unknown class)."
                        )
                    elif sign == 1:
                        st.success(
                            f"**SHORT {ORDER_SIZE} lot FUT + LONG {ORDER_SIZE} lot CASH**  \n"
                            f"thesis: spread is above mean and is expected to "
                            f"{'narrow' if direction==1 else 'widen'} "
                            f"(direction={direction:+d}, side={side:+d}, sign={sign:+d}).  \n"
                            f"entry notional ≈ ₹{notional:,.2f}, txn cost per leg ≈ ₹{est_cost:,.2f}"
                        )
                    else:   # sign == -1
                        st.success(
                            f"**LONG {ORDER_SIZE} lot FUT + SHORT {ORDER_SIZE} lot CASH**  \n"
                            f"thesis: spread is below mean and is expected to "
                            f"{'narrow' if direction==1 else 'widen'} "
                            f"(direction={direction:+d}, side={side:+d}, sign={sign:+d}).  \n"
                            f"entry notional ≈ ₹{notional:,.2f}, txn cost per leg ≈ ₹{est_cost:,.2f}"
                        )

                    # ---- class probabilities + feature importances ----
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(X)[0]
                        st.caption("class probabilities")
                        st.dataframe(pd.DataFrame({"class": clf.classes_, "p": proba}),
                                     use_container_width=True)
                    if hasattr(clf, "feature_importances_"):
                        fi = pd.DataFrame({
                            "feature":    FEATURE_COLS,
                            "importance": clf.feature_importances_,
                        }).sort_values("importance", ascending=False)
                        st.caption("classifier feature importances")
                        st.bar_chart(fi.set_index("feature"))
        except Exception as e:
            st.error(f"prediction failed: {e}")


# ------------------------------ tab 3 ------------------------------
with tab3:
    st.header("Backtest viewer")
    logs_dir = ROOT / "data" / "backtest_logs"
    if not logs_dir.exists():
        st.warning(f"No backtest_logs directory at {logs_dir}. Run `python src/backtest.py` first.")
    else:
        files = sorted(logs_dir.glob("trades_position_*.csv"))
        if not files:
            st.warning("No trade logs found.")
        else:
            choices = [f.name for f in files]
            picked  = st.multiselect("trade log files", choices, default=choices)
            if picked:
                frames = [pd.read_csv(logs_dir / n, parse_dates=["det_timestamp", "res_timestamp"])
                          for n in picked]
                combined = pd.concat(frames, ignore_index=True).sort_values("det_timestamp")

                rows = []
                for name, df in zip(picked, frames):
                    n = len(df)
                    if n == 0:
                        continue
                    wins = (df["pnl"] > 0).sum()
                    rows.append({
                        "file":      name,
                        "n_trades":  n,
                        "win_rate":  wins / n,
                        "pnl_gross": df["pnl_gross"].sum(),
                        "txn_costs": df["txn_cost"].sum(),
                        "pnl_net":   df["pnl"].sum(),
                    })
                if rows:
                    st.subheader("per-file summary")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if not combined.empty:
                    st.subheader("cumulative PnL")
                    equity = combined.set_index("det_timestamp")["pnl"].cumsum()
                    st.line_chart(equity)
                    st.subheader("raw trades")
                    st.dataframe(combined, use_container_width=True, height=400)


# ------------------------------ tab 4 ------------------------------
with tab4:
    st.header("MLflow Serving")
    st.caption(
        "POSTs a single feature row to a running `mlflow models serve` endpoint. "
        "Start it from the project root, e.g.: "
        "`mlflow models serve -m runs:/<run_id>/model_classifier --port 5001 --env-manager=local --no-conda`"
        "Three different models can be served : model_classifier, model_duration, model_revert"
    )

    endpoint = st.text_input(
        "Serving endpoint",
        value="http://127.0.0.1:5001/invocations",
        key="t4_endpoint",
    )

    st.subheader("Feature row")
    st.caption(f"Order matters — columns sent in config order: {FEATURE_COLS}")

    defaults = {
        "det_z_score":    2.5,
        "det_spread":     0.35,
        "side":           1,
        "dte":            10,
        "bucket":         2,
        "det_dist_std":   0.15,
        "det_dist_count": 500,
        "det_fut_ltq":    50,
        "det_oi_fut":     150000,
        "det_ltq":        100,
        "det_fut_ba":     0.10,
        "det_eq_ba":      0.06,
    }

    cols = st.columns(4)
    inputs: dict[str, float] = {}
    for i, feat in enumerate(FEATURE_COLS):
        inputs[feat] = cols[i % 4].number_input(
            feat, value=float(defaults.get(feat, 0.0)), format="%.6f", key=f"t4_{feat}",
        )

    if st.button("Predict via MLflow", type="primary", key="t4_submit"):
        import requests

        payload = {
            "dataframe_split": {
                "columns": FEATURE_COLS,
                "data":    [[inputs[f] for f in FEATURE_COLS]],
            }
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            resp.raise_for_status()
            body = resp.json()
            st.subheader("Response")
            st.json(body)
            preds = body.get("predictions", body)
            if isinstance(preds, list) and preds:
                st.metric("prediction", str(preds[0]))
        except Exception as e:
            st.error(f"request failed: {e}")


