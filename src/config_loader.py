"""
config_loader.py — Config loader and logger. Replaces utils.py.

Changes from original:
  - DATA_ROOT env var for path resolution (no hardcoded SSD paths)
  - CONFIG_PATH env var support
  - No sys.path manipulation
"""

import os
import yaml
import logging
from pathlib import Path
from datetime import datetime


class Cfg(dict):
    """Dict with dot-access. cfg.paths.raw_futures instead of cfg['paths']['raw_futures']."""

    def __getattr__(self, key):
        try:
            v = self[key]
            return Cfg(v) if isinstance(v, dict) else v
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


def _resolve_paths(raw_cfg: dict, data_root: Path) -> dict:
    """Resolve relative paths in the 'paths' section against DATA_ROOT."""
    paths = raw_cfg.get("paths", {})
    resolved = {}
    for key, value in paths.items():
        if isinstance(value, str):
            p = Path(value)
            resolved[key] = str(data_root / value) if not p.is_absolute() else value
        elif isinstance(value, list):
            resolved[key] = [
                str(data_root / v) if not Path(v).is_absolute() else v
                for v in value
            ]
        else:
            resolved[key] = value
    raw_cfg["paths"] = resolved
    return raw_cfg


def load_config(path: str = None) -> Cfg:
    """Load config.yaml with env var support.

    Resolution: explicit path > CONFIG_PATH env var > ./config.yaml
    All relative paths resolved against DATA_ROOT (default: ./data).
    """
    if path is None:
        path = os.environ.get("CONFIG_PATH", "config.yaml")

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_root = Path(os.environ.get("DATA_ROOT", ".")).resolve()
    raw = _resolve_paths(raw, data_root)
    return Cfg(raw)


def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    """Create a logger with console + file handlers."""
    if log_dir is None:
        log_dir = os.environ.get("LOG_DIR", "logs")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(
        log_path / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger