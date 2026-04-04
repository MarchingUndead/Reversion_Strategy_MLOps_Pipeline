"""
dal/base.py — Abstract interfaces for the Data Access Layer.

Four interfaces:
  TickSource     — raw tick data (CSV now, kdb+ later)
  CandleStore    — processed candle parquets
  TableStore     — distribution + reversion tables
  CalendarSource — expiry dates and trading days

Pipeline code ONLY calls these. Config key data_backend selects implementation.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class TickSource(ABC):
    @abstractmethod
    def list_contracts(self, symbols=None, year_filter=None) -> list[dict]:
        """Returns list of {symbol, contract, futures_path, equity_paths}."""
        ...

    @abstractmethod
    def read_futures_ticks(self, futures_path: str) -> pd.DataFrame:
        """Returns [timestamp, price, volume, oi] + optional bid/ask."""
        ...

    @abstractmethod
    def read_equity_ticks(self, equity_paths: list[str]) -> pd.DataFrame:
        """Returns [timestamp, cash_price], concatenated and deduped."""
        ...

    @abstractmethod
    def read_vix_ticks(self) -> Optional[pd.DataFrame]:
        """Returns [timestamp, vix] or None."""
        ...


class CandleStore(ABC):
    @abstractmethod
    def read_candles(self, symbol: str, contract: str, interval: int) -> pd.DataFrame:
        ...

    @abstractmethod
    def write_candles(self, df: pd.DataFrame, symbol: str, contract: str, interval: int):
        ...

    @abstractmethod
    def exists(self, symbol: str, contract: str, interval: int) -> bool:
        ...

    @abstractmethod
    def list_symbols(self) -> list[str]:
        ...

    @abstractmethod
    def list_contracts_for_symbol(self, symbol: str) -> list[str]:
        ...


class TableStore(ABC):
    @abstractmethod
    def read_dist_table(self, symbol: str) -> pd.DataFrame: ...

    @abstractmethod
    def write_dist_table(self, df: pd.DataFrame, symbol: str): ...

    @abstractmethod
    def read_reversion_table(self, symbol: str) -> pd.DataFrame: ...

    @abstractmethod
    def write_reversion_table(self, df: pd.DataFrame, symbol: str): ...

    @abstractmethod
    def dist_table_exists(self, symbol: str) -> bool: ...

    @abstractmethod
    def reversion_table_exists(self, symbol: str) -> bool: ...


class CalendarSource(ABC):
    @abstractmethod
    def get_expiry_dates(self) -> list: ...

    @abstractmethod
    def get_trading_days(self) -> pd.DatetimeIndex: ...


class DataBackend:
    """Composite: holds all four interfaces.

        backend = get_backend(cfg)
        backend.ticks.list_contracts()
        backend.candles.read_candles(...)
        backend.tables.read_dist_table(...)
        backend.calendars.get_expiry_dates()
    """
    def __init__(self, ticks: TickSource, candles: CandleStore,
                 tables: TableStore, calendars: CalendarSource):
        self.ticks = ticks
        self.candles = candles
        self.tables = tables
        self.calendars = calendars