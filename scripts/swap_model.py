"""CLI for swapping the served model.

Updates the MODEL_URI line in .env and (optionally) bounces the serving
container so FastAPI reloads the new model. Same swap logic is also imported
by the Streamlit sidebar.

Usage:
    python scripts/swap_model.py --list
    python scripts/swap_model.py --name reversion-classifier-pos0 --version 5
    python scripts/swap_model.py --name reversion-classifier-pos0 --stage Production
    python scripts/swap_model.py --name reversion-classifier-pos0 --auto
    python scripts/swap_model.py --name reversion-classifier-pos0 --version 5 --no-restart

`--auto` shorthand mirrors serve.py's `auto:<name>` resolver: latest
Production, falling back to latest version of any stage.

The script never touches data/raw/2025/ (hidden holdout) and never edits any
src/*.py file (preserve-core-logic). It only writes one line in .env and
invokes `docker compose up -d --force-recreate serving`.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)


def _client():
    """Lazy MLflow client — avoid importing mlflow when --help is invoked."""
    import mlflow
    from mlflow import MlflowClient
    tracking_uri = (os.environ.get("MLFLOW_TRACKING_URI")
                    or f"file:{(ROOT / 'mlruns').as_posix()}")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def list_registered() -> int:
    """Print registered models with their versions and stages."""
    client = _client()
    models = list(client.search_registered_models())
    if not models:
        print("(no registered models)")
        return 0
    for m in models:
        print(f"\n{m.name}")
        for v in client.search_model_versions(f"name='{m.name}'"):
            print(f"  v{v.version}  stage={v.current_stage:10s}  run_id={v.run_id}")
    return 0


def resolve_uri(name: str, version: str | None, stage: str | None,
                auto: bool) -> str:
    """Compute a MODEL_URI. Validates the (name, version|stage) exists."""
    client = _client()

    if auto:
        prod = client.get_latest_versions(name, stages=["Production"])
        if prod:
            return f"models:/{name}/Production"
        none = client.get_latest_versions(name, stages=["None"])
        if not none:
            raise SystemExit(f"--auto: no versions for {name!r}")
        return f"models:/{name}/{none[0].version}"

    if stage:
        vs = client.get_latest_versions(name, stages=[stage])
        if not vs:
            raise SystemExit(f"no version of {name!r} in stage {stage!r}")
        return f"models:/{name}/{stage}"

    if version:
        # Validate the version exists.
        try:
            client.get_model_version(name, version)
        except Exception as e:
            raise SystemExit(f"version {version} of {name!r} not found: {e}")
        return f"models:/{name}/{version}"

    raise SystemExit("must pass one of --version, --stage, or --auto")


_MODEL_URI_RE = re.compile(r"^MODEL_URI\s*=.*$", flags=re.MULTILINE)


def write_env(uri: str) -> None:
    """Replace the MODEL_URI line in .env (creates it if missing)."""
    if not ENV_FILE.is_file():
        ENV_FILE.write_text(f"MODEL_URI={uri}\n")
        print(f"created {ENV_FILE} with MODEL_URI={uri}")
        return
    text = ENV_FILE.read_text()
    if _MODEL_URI_RE.search(text):
        new_text = _MODEL_URI_RE.sub(f"MODEL_URI={uri}", text)
    else:
        new_text = text.rstrip() + f"\nMODEL_URI={uri}\n"
    ENV_FILE.write_text(new_text)
    print(f"updated {ENV_FILE}: MODEL_URI={uri}")


def restart_serving() -> int:
    """Force-recreate the serving container so it picks up the new MODEL_URI."""
    cmd = ["docker", "compose", "up", "-d", "--force-recreate", "serving"]
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="List registered models and exit.")
    parser.add_argument("--name", help="Registered model name "
                                       "(e.g. reversion-classifier-pos0).")
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--version", help="Specific version number to pin.")
    sel.add_argument("--stage", choices=["None", "Staging", "Production", "Archived"],
                     help="Latest version in this stage.")
    sel.add_argument("--auto", action="store_true",
                     help="Latest Production, else latest version (any stage).")
    parser.add_argument("--no-restart", action="store_true",
                        help="Update .env but don't bounce the serving container.")
    args = parser.parse_args()

    if args.list:
        return list_registered()

    if not args.name:
        parser.error("--name is required (unless --list)")

    uri = resolve_uri(args.name, args.version, args.stage, args.auto)
    write_env(uri)

    if args.no_restart:
        print("--no-restart: skipping container recreate. New URI takes effect on next manual restart.")
        return 0
    return restart_serving()


if __name__ == "__main__":
    sys.exit(main())
