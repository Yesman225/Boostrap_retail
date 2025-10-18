"""Download market data."""
from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd
import yfinance as yf


def download_price_history(
    tickers: Sequence[str],
    *,
    start: date | None = None,
    end: date | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices for the provided tickers."""
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        progress=False,
        auto_adjust=auto_adjust,
        group_by="ticker",
    )

    if isinstance(data.columns, pd.MultiIndex):
        closes = data.loc[:, pd.IndexSlice[:, "Close"]]
        closes.columns = closes.columns.droplevel(1)
    else:
        closes = data

    return closes.sort_index()
