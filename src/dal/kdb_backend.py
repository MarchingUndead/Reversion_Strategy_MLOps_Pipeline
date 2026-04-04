"""
dal/kdb_backend.py — kdb+ stub. All tick methods raise NotImplementedError.
CandleStore and TableStore delegate to file backend (derived artifacts stay on disk).
"""

from src.dal.base import TickSource, CandleStore, TableStore, CalendarSource, DataBackend

_MSG = "kdb+ backend not implemented. Set data_backend: 'file' in config.yaml."


class KdbTickSource(TickSource):
    def __init__(self, cfg):
        self._host = cfg.kdb.host
        self._port = cfg.kdb.port

    def list_contracts(self, symbols=None, year_filter=None):
        raise NotImplementedError(_MSG)

    def read_futures_ticks(self, futures_path):
        raise NotImplementedError(_MSG)

    def read_equity_ticks(self, equity_paths):
        raise NotImplementedError(_MSG)

    def read_vix_ticks(self):
        raise NotImplementedError(_MSG)


class KdbCandleStore(CandleStore):
    def __init__(self, cfg):
        from src.dal.file_backend import FileCandleStore
        self._d = FileCandleStore(cfg)

    def read_candles(self, s, c, i): return self._d.read_candles(s, c, i)
    def write_candles(self, df, s, c, i): return self._d.write_candles(df, s, c, i)
    def exists(self, s, c, i): return self._d.exists(s, c, i)
    def list_symbols(self): return self._d.list_symbols()
    def list_contracts_for_symbol(self, s): return self._d.list_contracts_for_symbol(s)


class KdbTableStore(TableStore):
    def __init__(self, cfg):
        from src.dal.file_backend import FileTableStore
        self._d = FileTableStore(cfg)

    def read_dist_table(self, s): return self._d.read_dist_table(s)
    def write_dist_table(self, df, s): return self._d.write_dist_table(df, s)
    def read_reversion_table(self, s): return self._d.read_reversion_table(s)
    def write_reversion_table(self, df, s): return self._d.write_reversion_table(df, s)
    def dist_table_exists(self, s): return self._d.dist_table_exists(s)
    def reversion_table_exists(self, s): return self._d.reversion_table_exists(s)


class KdbCalendarSource(CalendarSource):
    def __init__(self, cfg): raise NotImplementedError(_MSG)
    def get_expiry_dates(self): raise NotImplementedError(_MSG)
    def get_trading_days(self): raise NotImplementedError(_MSG)


def create_kdb_backend(cfg):
    return DataBackend(
        ticks=KdbTickSource(cfg), candles=KdbCandleStore(cfg),
        tables=KdbTableStore(cfg), calendars=KdbCalendarSource(cfg),
    )