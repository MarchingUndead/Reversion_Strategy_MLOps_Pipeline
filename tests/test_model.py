"""Pure-function tests for src/model.py.

Covers `_clean`, `_slice_by_range`, and the column-derivation path of
`prepare_events`. All inputs are inline pandas DataFrames — no disk I/O.

`train_position` and `evaluate` are NOT covered here: they fit real RFs and
belong at the integration layer, not in unit tests.
"""
import numpy as np
import pandas as pd

from model import _clean, _slice_by_range, prepare_events


# ---------------------------------------------------------------- _clean
def test_clean_drops_rows_with_nan_target():
    df = pd.DataFrame({
        "x":             [1.0, 2.0, 3.0],
        "klass":         ["reversion", None, "continuation"],
        "duration_sec":  [60.0, 120.0, 180.0],
        "revert_delta":  [0.5, 0.4, 0.3],
    })
    out = _clean(df, ["x"])
    assert len(out) == 2
    assert "klass" not in out["klass"].isna().tolist()


def test_clean_drops_rows_with_nan_feature():
    df = pd.DataFrame({
        "x":             [1.0, np.nan, 3.0],
        "klass":         ["reversion", "divergence", "continuation"],
        "duration_sec":  [60.0, 120.0, 180.0],
        "revert_delta":  [0.5, 0.4, 0.3],
    })
    out = _clean(df, ["x"])
    assert len(out) == 2
    assert not out["x"].isna().any()


def test_clean_returns_all_rows_when_clean():
    df = pd.DataFrame({
        "x":             [1.0, 2.0],
        "klass":         ["reversion", "divergence"],
        "duration_sec":  [60.0, 120.0],
        "revert_delta":  [0.5, 0.4],
    })
    out = _clean(df, ["x"])
    assert len(out) == 2


# --------------------------------------------------------- _slice_by_range
def _events_one_per_month(year: int) -> pd.DataFrame:
    return pd.DataFrame({
        "det_timestamp": [pd.Timestamp(year, m, 15) for m in range(1, 13)],
    })


def test_slice_by_range_inclusive_both_ends():
    df = _events_one_per_month(2024)
    out = _slice_by_range(df, ["2024-01", "2024-06"])
    assert len(out) == 6
    # Boundary check: first day of January and a mid-June timestamp should both be in.
    assert out["det_timestamp"].min() == pd.Timestamp(2024, 1, 15)
    assert out["det_timestamp"].max() == pd.Timestamp(2024, 6, 15)


def test_slice_by_range_excludes_outside_months():
    df = _events_one_per_month(2024)
    out = _slice_by_range(df, ["2024-06", "2024-08"])
    assert len(out) == 3
    months = sorted(out["det_timestamp"].dt.month.tolist())
    assert months == [6, 7, 8]


def test_slice_by_range_includes_last_day_of_end_month():
    # Ensure the end boundary really extends to the end of the month, not to its 1st.
    df = pd.DataFrame({
        "det_timestamp": [pd.Timestamp("2024-06-30 23:59:00"),
                          pd.Timestamp("2024-07-01 00:00:00")],
    })
    out = _slice_by_range(df, ["2024-01", "2024-06"])
    assert len(out) == 1
    assert out["det_timestamp"].iloc[0] == pd.Timestamp("2024-06-30 23:59:00")


# -------------------------------------------------------- prepare_events
def _minimal_event_row(*, day_fut: int, contract: str,
                       det_z: float = 2.5, res_z: float = 0.5) -> dict:
    """Just enough columns for prepare_events to derive its outputs."""
    det_ts = pd.Timestamp(str(day_fut)) + pd.Timedelta(hours=10)
    res_ts = det_ts + pd.Timedelta(seconds=300)
    return {
        "day_fut":           day_fut,
        "contract":          contract,
        "det_timestamp":     det_ts,
        "res_timestamp":     res_ts,
        "det_z_score":       det_z,
        "res_z_score":       res_z,
        "det_spread":        0.30,
        "res_spread":        0.05,
        "det_fut_askprice":  100.55,
        "det_fut_bidprice":  100.45,
        "det_eq_askprice":   100.03,
        "det_eq_bidprice":   99.97,
    }


def test_prepare_events_assigns_positions_0_1_2():
    # Session = JUL 2024 (day_fut 20240715). JUL=6 in MONTHS.
    # Expect: contract=JUL -> position 0, AUG -> 1, SEP -> 2.
    rows = [
        _minimal_event_row(day_fut=20240715, contract="JUL"),
        _minimal_event_row(day_fut=20240715, contract="AUG"),
        _minimal_event_row(day_fut=20240715, contract="SEP"),
    ]
    out = prepare_events(pd.DataFrame(rows))
    assert sorted(out["position"].tolist()) == [0, 1, 2]


def test_prepare_events_drops_position_above_2():
    # Session = JAN 2024 (day_fut 20240115). Contract DEC of the SAME year
    # has contract_month=11, session_month=0 -> position = 11, dropped.
    rows = [
        _minimal_event_row(day_fut=20240115, contract="JAN"),  # position 0, kept
        _minimal_event_row(day_fut=20240115, contract="DEC"),  # position 11, dropped
    ]
    out = prepare_events(pd.DataFrame(rows))
    assert len(out) == 1
    assert out["position"].iloc[0] == 0


def test_prepare_events_derives_expected_columns():
    rows = [_minimal_event_row(day_fut=20240715, contract="JUL")]
    out = prepare_events(pd.DataFrame(rows))
    row = out.iloc[0]
    # duration_sec = 300 by construction
    assert row["duration_sec"] == 300.0
    # revert_delta = |det_z| - |res_z| = 2.5 - 0.5 = 2.0
    assert row["revert_delta"] == 2.0
    # spread_change = res_spread - det_spread = 0.05 - 0.30 = -0.25
    assert np.isclose(row["spread_change"], -0.25)
    # fut bid-ask = 100.55 - 100.45 = 0.10
    assert np.isclose(row["det_fut_ba"], 0.10)
    # eq bid-ask = 100.03 - 99.97 = 0.06
    assert np.isclose(row["det_eq_ba"], 0.06)
