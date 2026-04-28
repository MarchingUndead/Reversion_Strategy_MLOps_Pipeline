"""Unified smoke / load / alert-fire tool.

Replaces three legacy scripts (load_gen.py, fire_alerts.ps1, test_email.ps1)
with a single Python CLI dispatched on subcommand. Used by:

  - airflow/dags/smoke_tests.py (read-only subcommands every 30 min)
  - manual demos (load + fire-alerts + email-test)

Read-only subcommands (used by Airflow):
  health      check serving /health
  ready       check serving /ready
  predict     POST a synthetic feature row, assert 200 + has 'predictions'
  prom-up     query prometheus, assert up{job="reversion-serving"} == 1

Manual / demo subcommands:
  load        drive synthetic load (replaces load_gen.py)
  fire-alerts orchestrate alert-rule firing (replaces fire_alerts.ps1)
  email-test  POST a synthetic alert to Alertmanager (replaces test_email.ps1)

Endpoints come from env vars so the same script works in-container
(airflow→serving DNS) and on the host (localhost):

  SMOKE_SERVING_URL (default http://localhost:5002)
  SMOKE_PROM_URL    (default http://localhost:9090)
  SMOKE_ALERT_URL   (default http://localhost:9093)

Never reads data/raw/2025/ — sample rows come from data/events/ only.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT       = Path(__file__).resolve().parents[1]
EVENTS_DIR = ROOT / "data" / "events"

SERVING_URL = os.environ.get("SMOKE_SERVING_URL", "http://localhost:5002")
PROM_URL    = os.environ.get("SMOKE_PROM_URL",    "http://localhost:9090")
ALERT_URL   = os.environ.get("SMOKE_ALERT_URL",   "http://localhost:9093")

FEATURE_COLS = [
    "det_z_score", "det_spread", "side", "dte", "bucket",
    "det_dist_std", "det_dist_count", "det_fut_ltq", "det_oi_fut",
    "det_ltq", "det_fut_ba", "det_eq_ba",
]

SYNTHETIC_ROW = [1.5, 0.02, 1, 7, 0, 0.8, 50, 100.0, 25000.0, 150.0, 0.05, 0.04]


# ----------------------------------------------------------------------- helpers
def _payload_ok(rows):
    return {"dataframe_split": {"columns": FEATURE_COLS, "data": rows}}


def _load_sample_rows(n: int = 200):
    """Pull sample rows from data/events/. Falls back to SYNTHETIC_ROW."""
    try:
        import pandas as pd
        files = sorted(EVENTS_DIR.glob("*.parquet")) + sorted(EVENTS_DIR.glob("*.csv"))
        files = [f for f in files if "2025" not in f.name]
        if not files:
            raise FileNotFoundError("no usable events files")
        frames = []
        for f in files[:5]:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            cols = [c for c in FEATURE_COLS if c in df.columns]
            if len(cols) == len(FEATURE_COLS):
                frames.append(df[FEATURE_COLS].dropna().sample(min(n, len(df)), random_state=42))
        if not frames:
            raise ValueError("none of the events files had all feature_cols")
        sampled = pd.concat(frames).sample(n=min(n, sum(len(f) for f in frames)), random_state=1)
        return sampled.values.tolist()
    except Exception as exc:
        print(f"[smoke] events sample unavailable ({type(exc).__name__}: {exc}); using synthetic row")
        return [SYNTHETIC_ROW]


# ----------------------------------------------------------------------- read-only
def cmd_health(_) -> int:
    r = requests.get(f"{SERVING_URL}/health", timeout=5)
    print(f"GET {SERVING_URL}/health -> {r.status_code} {r.text[:200]}")
    return 0 if r.status_code == 200 else 1


def cmd_ready(_) -> int:
    r = requests.get(f"{SERVING_URL}/ready", timeout=5)
    print(f"GET {SERVING_URL}/ready -> {r.status_code} {r.text[:200]}")
    return 0 if r.status_code == 200 else 1


def cmd_predict(_) -> int:
    r = requests.post(f"{SERVING_URL}/invocations",
                      json=_payload_ok([SYNTHETIC_ROW]), timeout=30)
    print(f"POST {SERVING_URL}/invocations -> {r.status_code} {r.text[:200]}")
    if r.status_code != 200:
        return 1
    body = r.json()
    if "predictions" not in body or not body["predictions"]:
        print("ERROR: response has no 'predictions'")
        return 2
    return 0


def cmd_prom_up(_) -> int:
    # Job name must match scrape_configs[].job_name in monitoring/prometheus.yml
    # (currently 'model-server'). Don't rename — dashboards and alert rules
    # reference this job name too, so a rename has multi-file blast radius.
    q = 'up{job="model-server"}'
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": q}, timeout=5)
    if r.status_code != 200:
        print(f"prometheus query failed: {r.status_code} {r.text[:200]}")
        return 1
    result = r.json().get("data", {}).get("result", [])
    if not result:
        print(f"no result for {q!r} — is the serving target scraped?")
        return 2
    val = result[0]["value"][1]
    print(f'{q} = {val}')
    return 0 if val == "1" else 3


# ----------------------------------------------------------------------- load gen
def _payload_bad_json():
    return "this is not json {"

def _payload_bad_schema():
    return {"not_a_dataframe_split": [1, 2, 3]}

def _payload_exception(rows):
    # column count mismatch — predict() raises on shape mismatch.
    return {"dataframe_split": {"columns": FEATURE_COLS[:-3], "data": rows}}

def _payload_big(rows, n=1000):
    base = rows[0] if rows else SYNTHETIC_ROW
    return _payload_ok([base] * n)


def _send(url: str, mode: str, rows, session: requests.Session):
    headers = {"X-Client-ID": f"smoke-{mode}"}
    start = time.perf_counter()
    if mode == "bad-json":
        r = session.post(url, data=_payload_bad_json(),
                         headers={**headers, "Content-Type": "application/json"}, timeout=10)
    elif mode == "bad-schema":
        r = session.post(url, json=_payload_bad_schema(), headers=headers, timeout=10)
    elif mode == "exception":
        r = session.post(url, json=_payload_exception([random.choice(rows)]),
                         headers=headers, timeout=30)
    elif mode == "big":
        r = session.post(url, json=_payload_big(rows), headers=headers, timeout=60)
    else:  # ok / burst-as-ok
        r = session.post(url, json=_payload_ok([random.choice(rows)]),
                         headers=headers, timeout=10)
    return r.status_code, time.perf_counter() - start


def cmd_load(args) -> int:
    rows = _load_sample_rows()
    print(f"[smoke load] mode={args.mode} rps={args.rps} duration={args.duration}s "
          f"rows_sampled={len(rows)} url={args.url}")

    interval = 1.0 / args.rps if args.mode != "burst" else 1.0 / max(args.rps, 50.0)
    end_at   = time.perf_counter() + args.duration

    sess = requests.Session()
    sent = ok = bad = 0
    while time.perf_counter() < end_at:
        mode = (random.choice(["ok", "bad-json", "bad-schema", "exception", "big"])
                if args.mode == "mix" else args.mode)
        active_mode = "ok" if mode == "burst" else mode
        try:
            status, elapsed = _send(args.url, active_mode, rows, sess)
            sent += 1
            if 200 <= status < 300:
                ok += 1
            else:
                bad += 1
            if sent % 20 == 0:
                print(f"  sent={sent:5d} ok={ok:5d} bad={bad:5d} last={status} "
                      f"t={elapsed*1000:6.1f}ms")
        except requests.RequestException as exc:
            bad += 1
            sent += 1
            print(f"  request error: {type(exc).__name__}: {exc}")
        time.sleep(interval)

    print(f"[smoke load] done. sent={sent} ok={ok} bad={bad}")
    return 0 if bad == 0 or args.mode != "ok" else 1


# ----------------------------------------------------------------------- alerts
def _email_sent_count() -> int:
    """Read alertmanager_notifications_total{integration="email"} from /metrics."""
    try:
        body = requests.get(f"{ALERT_URL}/metrics", timeout=5).text
    except requests.RequestException:
        return 0
    for line in body.splitlines():
        if line.startswith('alertmanager_notifications_total{') and 'integration="email"' in line:
            try:
                return int(float(line.split()[-1]))
            except ValueError:
                pass
    return 0


def _show_firing() -> None:
    try:
        data = requests.get(f"{PROM_URL}/api/v1/alerts", timeout=5).json()
    except Exception as exc:
        print(f"  (could not list firing alerts: {exc})")
        return
    firing = [a for a in data.get("data", {}).get("alerts", []) if a.get("state") == "firing"]
    if firing:
        print("Currently FIRING in Prometheus:")
        for a in firing:
            print(f"  - {a['labels'].get('alertname','?'):25s} since {a.get('activeAt','?')}")


def cmd_fire_alerts(args) -> int:
    """Orchestrate alert demos (replaces fire_alerts.ps1)."""
    before = _email_sent_count()
    print(f"Email notifications sent so far: {before}")

    def _run_load(mode, rps, duration):
        load_args = argparse.Namespace(url=f"{SERVING_URL}/invocations",
                                       mode=mode, rps=rps, duration=duration)
        return cmd_load(load_args)

    if args.mode == "model-down":
        print("=== ModelServerDown demo ===")
        print("Stop the serving container: docker compose stop serving")
        print("ModelServerDown fires after 1m of scrape failures (severity=critical).")
        print("Expect email in 70-80s after stopping.")
    elif args.mode == "exceptions":
        print("=== HighExceptionRate / High5xxRate demo (~3 min) ===")
        _run_load("exception", 5, 200)
    elif args.mode == "four-xx":
        print("=== High4xxRate / ZeroRequestSuccess demo (~6 min) ===")
        _run_load("bad-schema", 5, 360)
    elif args.mode == "slow":
        print("=== HighLatencyP95 / HighLatencyP99 demo (~6 min, big payloads) ===")
        _run_load("big", 2, 360)
    elif args.mode == "spike":
        print("=== RequestRateSpike demo (~5 min: baseline + spike) ===")
        _run_load("ok", 1, 60)
        _run_load("burst", 50, 240)
    elif args.mode == "all":
        print("Running four-xx -> slow -> spike sequentially. ~17 min total.")
        for m in ("four-xx", "slow", "spike"):
            cmd_fire_alerts(argparse.Namespace(mode=m))

    _show_firing()
    after = _email_sent_count()
    delta = after - before
    print(f"Email notifications: {before} -> {after}  (delta = {delta})")
    if delta == 0:
        print("No emails yet. Either alert is still pending (for: window) or run was too short.")
    else:
        print(f"Check inbox + spam at {os.environ.get('ALERT_TO','<unset>')}")
    return 0


def cmd_email_test(_) -> int:
    """POST a synthetic alert to Alertmanager (replaces test_email.ps1)."""
    now  = datetime.now(timezone.utc)
    body = [{
        "labels":      {"alertname": "EmailPathTest", "severity": "critical", "area": "demo"},
        "annotations": {"summary": "Manual SMTP test from scripts/smoke.py",
                        "description": "If this email arrives, Gmail SMTP credentials are correct."},
        "startsAt":    now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "endsAt":      (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }]
    before_sent = _email_sent_count()
    print(f"Before: sent={before_sent}")
    print(f"POST {ALERT_URL}/api/v2/alerts")
    r = requests.post(f"{ALERT_URL}/api/v2/alerts", json=body, timeout=10)
    print(f"-> {r.status_code} {r.text[:200]}")
    if r.status_code >= 300:
        return 1

    print("Waiting up to 30s for Alertmanager to flush + send...")
    deadline = time.time() + 30
    while time.time() < deadline:
        if _email_sent_count() > before_sent:
            break
        time.sleep(0.5)
    after = _email_sent_count()
    print(f"After:  sent={after}")
    if after > before_sent:
        print(f"EMAIL SENT. Check inbox + spam at {os.environ.get('ALERT_TO','<unset>')}")
        return 0
    print("TIMED OUT waiting for delivery — alert may still be in group_wait.")
    return 2


# ----------------------------------------------------------------------- entry
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health",  help="serving /health")
    sub.add_parser("ready",   help="serving /ready")
    sub.add_parser("predict", help="POST synthetic row, assert 200")
    sub.add_parser("prom-up", help='prometheus up{job="reversion-serving"}')

    pl = sub.add_parser("load", help="drive load on serving")
    pl.add_argument("--url",      default=f"{SERVING_URL}/invocations")
    pl.add_argument("--mode",     default="ok",
                    choices=["ok", "burst", "bad-json", "bad-schema", "exception", "big", "mix"])
    pl.add_argument("--rps",      type=float, default=5.0)
    pl.add_argument("--duration", type=float, default=60.0)

    pf = sub.add_parser("fire-alerts", help="orchestrate alert-rule demos")
    pf.add_argument("--mode", required=True,
                    choices=["model-down", "exceptions", "four-xx", "slow", "spike", "all"])

    sub.add_parser("email-test", help="POST synthetic alert to Alertmanager")

    args = p.parse_args()

    handlers = {
        "health":      cmd_health,
        "ready":       cmd_ready,
        "predict":     cmd_predict,
        "prom-up":     cmd_prom_up,
        "load":        cmd_load,
        "fire-alerts": cmd_fire_alerts,
        "email-test":  cmd_email_test,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
