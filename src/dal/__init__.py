"""
dal/__init__.py — Factory.

    from src.dal import get_backend
    backend = get_backend(cfg)
"""

from src.dal.base import DataBackend


def get_backend(cfg) -> DataBackend:
    backend_type = cfg.get("data_backend", "file").lower()
    if backend_type == "file":
        from src.dal.file_backend import create_file_backend
        return create_file_backend(cfg)
    elif backend_type == "kdb":
        from src.dal.kdb_backend import create_kdb_backend
        return create_kdb_backend(cfg)
    else:
        raise ValueError(f"Unknown data_backend: '{backend_type}'. Expected 'file' or 'kdb'.")