"""Streamlit UI for the reversion pipeline.

Three tabs:
  1. Event explorer    - run events.run() for a (symbol, year, month, day)
                          and render the per-day overlay.
  2. Backtest viewer   - load persisted trade logs and render the equity
                          curve + per-position summary.
  3. MLflow Serving    - POST a feature row to a running `mlflow models serve`
                          endpoint and display the returned prediction.

Live prediction is intentionally delegated to the MLflow Serving tab; this
app does not load pickles itself.
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
FEATURE_COLS = _cfg["model"]["feature_cols"]
MONTHS       = _cfg["months"]

st.set_page_config(page_title="Reversion Strategy", layout="wide")
st.title("Reversion Strategy")

tab1, tab2, tab3 = st.tabs([
    "Event explorer",
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
    day_input = c4.text_input("day (1-31 or YYYYMMDD, blank = full month)", value="", key="t1_day")

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


# ------------------------------ tab 3 ------------------------------
with tab3:
    st.header("MLflow Serving")
    st.caption(
        "POSTs a single feature row to a running `mlflow models serve` endpoint. "
        "Start one from the project root, e.g.: "
        "`mlflow models serve -m runs:/<run_id>/model_classifier --port 5001 --env-manager=local --no-conda`. "
        "Three artefacts can be served independently: `model_classifier`, `model_duration`, `model_revert`."
    )

    endpoint = st.text_input(
        "Serving endpoint",
        value="http://127.0.0.1:5001/invocations",
        key="t3_endpoint",
    )

    st.subheader("Feature row")
    st.caption(f"Order matters - columns sent in config order: {FEATURE_COLS}")

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
            feat, value=float(defaults.get(feat, 0.0)), format="%.6f", key=f"t3_{feat}",
        )

    if st.button("Predict via MLflow", type="primary", key="t3_submit"):
        import requests

        payload = {
            "dataframe_split": {
                "columns": FEATURE_COLS,
                "data":    [[inputs[f] for f in FEATURE_COLS]],
            }
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot reach `{endpoint}`. Start the server with "
                f"`mlflow models serve -m runs:/<run_id>/model_classifier "
                f"--port 5001 --env-manager=local --no-conda` and retry."
            )
        except requests.exceptions.Timeout:
            st.error(f"Request to {endpoint} timed out after 60s. The server may still be loading the model.")
        except requests.exceptions.HTTPError as e:
            st.error(f"server returned {resp.status_code}: {resp.text[:500]}")
        except Exception as e:
            st.error(f"request failed: {e}")
        else:
            body = resp.json()
            st.subheader("Response")
            st.json(body)
            preds = body.get("predictions", body)
            if isinstance(preds, list) and preds:
                st.metric("prediction", str(preds[0]))
