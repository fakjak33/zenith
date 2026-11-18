import pytest

from zenith.backtest.data.providers import clean_ticker, _cache_path
from zenith.backtest.config import DATA_CACHE


def test_clean_ticker_normalizes():
    assert clean_ticker("spy") == "SPY"
    assert clean_ticker(" aapl ") == "AAPL"
    assert clean_ticker("BRK-B") == "BRK-B"
    assert clean_ticker("^GSPC") == "^GSPC"


def test_clean_ticker_blocks_path_traversal():
    # traversal/garbage chars are stripped, not honored
    assert "/" not in clean_ticker("../../etc/passwd")
    assert "\\" not in clean_ticker("..\\..\\windows")
    assert ".." not in clean_ticker("..\\..\\x")


def test_clean_ticker_rejects_empty():
    with pytest.raises(ValueError):
        clean_ticker("///")
    with pytest.raises(ValueError):
        clean_ticker("")


def test_cache_path_stays_in_cache_dir():
    p = _cache_path("SPY")
    assert DATA_CACHE.resolve() in p.parents
    # even a hostile ticker resolves inside the cache dir
    p2 = _cache_path("..\\..\\evil")
    assert DATA_CACHE.resolve() in p2.parents
