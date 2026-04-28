"""Smoke tests for src/backtest.py — the no-fixture-needed cases.

The PnL-arithmetic case (single-row eval_df with hand-computed gross / cost /
net) lives in Phase 2 and needs `tests/fixtures/sample_events.csv`. Until
then, this file covers the early-return paths that don't touch any feature
columns: empty input and all-continuation predictions.
"""
import pandas as pd

from backtest import backtest


def test_backtest_returns_none_on_empty_dataframe():
    assert backtest(pd.DataFrame(), label="empty") is None


def test_backtest_returns_none_on_none_input():
    assert backtest(None, label="none") is None


def test_backtest_returns_none_when_all_predictions_are_continuation():
    eval_df = pd.DataFrame({"pred_klass": ["continuation"] * 5})
    assert backtest(eval_df, label="all-cont") is None
