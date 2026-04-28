"""Pure-function tests for src/preprocess.py.

Only `get_bucket` is exercised here — it's a pure function over a string and
the config-loaded `buckets` list. `get_dte` depends on the trading calendar
read at module import; it's covered at the integration layer.
"""
from preprocess import get_bucket


def test_get_bucket_at_first_anchor():
    assert get_bucket("09:15:00") == 0


def test_get_bucket_at_each_anchor():
    # buckets in config.yaml: 09:15, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00
    assert get_bucket("10:00:00") == 1
    assert get_bucket("11:00:00") == 2
    assert get_bucket("12:00:00") == 3
    assert get_bucket("13:00:00") == 4
    assert get_bucket("14:00:00") == 5
    assert get_bucket("15:00:00") == 6


def test_get_bucket_between_anchors():
    assert get_bucket("09:30:00") == 0
    assert get_bucket("10:30:00") == 1
    assert get_bucket("14:59:59") == 5


def test_get_bucket_pre_market_falls_through_to_zero():
    # Anything before the first anchor is mapped to bucket 0 (no separate
    # pre-market bucket; matches the `return 0` fallthrough at the end).
    assert get_bucket("08:00:00") == 0
