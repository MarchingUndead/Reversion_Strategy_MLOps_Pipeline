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

import datetime as _dt
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
_SERVE_CFG   = _cfg.get("serve", {})
SERVE_HOST   = _SERVE_CFG.get("host", "serving")
SERVE_PORT   = int(_SERVE_CFG.get("port", 5002))
SERVE_URL    = f"http://{SERVE_HOST}:{SERVE_PORT}"

st.set_page_config(page_title="Reversion Strategy", layout="wide")
st.title("Reversion Strategy")


# ------------------------------ sidebar: registry / model swap ------------------------------
def _ensure_mlflow_tracking():
    """Set the tracking URI on the global mlflow module from config.yaml.
    Reads `serve.mlruns_path` (default ./mlruns); relative paths resolve
    against the repo root. Idempotent."""
    import mlflow
    mlruns = Path(_cfg.get("serve", {}).get("mlruns_path", "./mlruns"))
    if not mlruns.is_absolute():
        mlruns = (ROOT / mlruns).resolve()
    mlflow.set_tracking_uri(f"file:{mlruns.as_posix()}")
    return mlflow


def _registry_client():
    """Cached MLflow client. Returns (client, error_str). client is None on error."""
    try:
        _ensure_mlflow_tracking()
        from mlflow import MlflowClient
        return MlflowClient(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@st.cache_resource(show_spinner="loading model from MLflow registry...")
def _load_head_model(head: str, position: int, symbol: str,
                     stage: str = "Production"):
    """Resolve and load reversion-{head}-pos{position}-{symbol}.

    Returns (model, version_label, error). model is None on error. Falls back
    to the latest version of any stage if the requested stage has no version.
    Cached on (head, position, symbol, stage).
    """
    try:
        mlflow = _ensure_mlflow_tracking()
        import mlflow.sklearn
        from mlflow import MlflowClient
        name = f"reversion-{head}-pos{position}-{symbol}"
        client = MlflowClient()
        prod = client.get_latest_versions(name, stages=[stage])
        if prod:
            v = prod[0]
            label = f"v{v.version} ({stage})"
        else:
            all_v = list(client.search_model_versions(f"name='{name}'"))
            if not all_v:
                return None, None, f"no versions of {name} in the registry"
            all_v.sort(key=lambda x: int(x.version), reverse=True)
            v = all_v[0]
            label = f"v{v.version} ({v.current_stage})"
        # Use runs:/<run_id>/<artifact_path> instead of models:/<name>/<version>:
        # the models:/ resolver writes a registered_model_meta sidecar next to
        # the artifact, which fails on read-only mlruns mounts (the airflow
        # scheduler container mounts /opt/airflow/project/mlruns RO).
        run_uri = f"runs:/{v.run_id}/model_{head}"
        model = mlflow.sklearn.load_model(run_uri)
        return model, label, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


tab1, tab2, tab3 = st.tabs([
    "Event explorer",
    "Backtest viewer",
    "MLflow Serving",
])


# ------------------------------ tab 1 ------------------------------
with tab1:
    st.header("Event explorer")

    # ---- Registered models (moved from sidebar) ----
    with st.expander("Registered models — inspect, pin, promote", expanded=False):
        client, err = _registry_client()
        if err:
            st.error(f"MLflow client unavailable: {err}")
        else:
            models = list(client.search_registered_models())
            if not models:
                st.info("No registered models. Run the pipeline DAG to populate the registry.")
            else:
                names = [m.name for m in models]
                rm1, rm2 = st.columns([1, 1])
                picked_name = rm1.selectbox("Model name", names, key="rm_name")
                versions = list(client.search_model_versions(f"name='{picked_name}'"))
                versions.sort(key=lambda v: int(v.version), reverse=True)
                v_labels = [f"v{v.version} ({v.current_stage})" for v in versions]
                v_idx = rm2.selectbox("Version", range(len(versions)),
                                      format_func=lambda i: v_labels[i], key="rm_version")
                picked_version = versions[v_idx].version

                btn_pin, btn_prom = st.columns(2)
                if btn_pin.button("Pin this version (bounces serving)", key="rm_apply"):
                    import subprocess as _sp
                    cmd = ["python", str(ROOT / "scripts" / "swap_model.py"),
                           "--name", picked_name, "--version", str(picked_version)]
                    with st.spinner("swapping model and bouncing serving container..."):
                        proc = _sp.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
                    if proc.returncode == 0:
                        st.success(f"served model -> {picked_name}/v{picked_version}")
                    else:
                        st.error(f"swap failed (rc={proc.returncode})")
                    with st.expander("output"):
                        st.code((proc.stdout or "") + "\n" + (proc.stderr or ""))
                if btn_prom.button("Promote to Production", key="rm_promote"):
                    try:
                        client.transition_model_version_stage(
                            name=picked_name, version=picked_version,
                            stage="Production", archive_existing_versions=True,
                        )
                        st.success(f"{picked_name}/v{picked_version} -> Production")
                    except Exception as e:
                        st.error(f"promote failed: {type(e).__name__}: {e}")

                st.caption("Currently served:")
                try:
                    import requests as _rq
                    # FastAPI container exposes /health with the loaded model_uri.
                    # Plain `mlflow models serve` only exposes /ping + /invocations,
                    # so fall back to /ping to detect a host-launched CLI server.
                    h = _rq.get(f"{SERVE_URL}/health", timeout=3)
                    if h.status_code == 200:
                        st.code(h.json().get("model_uri", "?"))
                    else:
                        p = _rq.get(f"{SERVE_URL}/ping", timeout=3)
                        if p.status_code == 200:
                            st.code(f"alive at {SERVE_URL} "
                                    f"(mlflow CLI mode — model_uri not exposed)")
                        else:
                            st.caption("(serving endpoint unreachable)")
                except Exception:
                    st.caption("(serving endpoint unreachable)")

    st.divider()

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


# tab 3
with tab3:
    st.header("Direct prediction")
    st.caption(
        "Loads a head from the MLflow Registry and runs `.predict()` in this Streamlit "
        "process — bypassing the FastAPI serving container, which can only host one head "
        "at a time. Use this to switch between classifier / duration / revert without "
        "bouncing the server. For the production HTTP path, run "
        "`python scripts/smoke.py predict` against the running serving container."
    )

    st.subheader("Trade context")
    cc1, cc2, cc3, cc4 = st.columns(4)
    t3_symbol = cc1.selectbox("symbol", _cfg["symbols"], key="t3_symbol",
                              help="Models are per-symbol — picks the right "
                                   "(symbol, position) registry entry.")
    session_date = cc2.date_input("session date", value=_dt.date.today(), key="t3_session")
    expiry_idx = cc3.selectbox(
        "contract expiry month",
        options=list(range(12)),
        format_func=lambda i: MONTHS[i],
        index=session_date.month - 1,
        key="t3_expiry",
        help="Position is derived as (expiry_month - session_month) mod 12. "
             "Only positions 0/1/2 (near/mid/far) have models registered.",
    )
    session_month = session_date.month - 1
    position = (expiry_idx - session_month) % 12
    pos_label = ["near", "mid", "far"][position] if position in (0, 1, 2) else "out of range"
    cc4.metric("position", f"{position} ({pos_label})")

    if position not in (0, 1, 2):
        st.warning(
            f"Position={position} is outside the 3-contract ladder. Pick an expiry "
            f"within the 3 months following the session date."
        )

    st.subheader("Head")
    head = st.radio(
        "head to call",
        options=["classifier", "duration", "revert"],
        horizontal=True,
        captions=[
            "klass ∈ {reversion, divergence, continuation}",
            "duration_sec (regressor)",
            "revert_delta (regressor)",
        ],
        key="t3_head",
    )

    if position in (0, 1, 2):
        st.caption(f"Resolved model: `reversion-{head}-pos{position}-{t3_symbol}`")

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

    if st.button("Predict (direct load)", type="primary", key="t3_submit"):
        if position not in (0, 1, 2):
            st.error("Position out of range — adjust expiry or session date.")
        else:
            model, ver_label, err = _load_head_model(head, position, t3_symbol)
            if err:
                st.error(err)
            else:
                df = pd.DataFrame(
                    [[inputs[f] for f in FEATURE_COLS]], columns=FEATURE_COLS,
                )
                try:
                    preds = model.predict(df)
                    val = preds[0] if hasattr(preds, "__getitem__") else preds
                    st.success(f"loaded reversion-{head}-pos{position}-{t3_symbol} {ver_label}")
                    st.metric(head, str(val))
                    if hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(df)[0]
                            classes = list(getattr(model, "classes_", []))
                            if len(classes) == len(proba):
                                st.subheader("Class probabilities")
                                st.dataframe(
                                    pd.DataFrame({"class": classes, "proba": proba}),
                                    use_container_width=True, hide_index=True,
                                )
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"predict failed: {type(e).__name__}: {e}")
