"""Ticker filtering utilities."""
from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence

import yfinance as yf
from dateutil.relativedelta import relativedelta


def filter_by_history(tickers: Sequence[str], *, minimum_years: int, today: date | None = None) -> list[str]:
    """Return tickers available for at least ``minimum_years`` years of history."""
    if not tickers:
        return []

    reference_date = (today or date.today()) - relativedelta(years=minimum_years)
    eligible: list[str] = []

    for symbol in tickers:
        data = yf.download(symbol, period="max", progress=False, auto_adjust=True)
        if data.empty:
            continue
        first_date = data.index.min().date()
        if first_date <= reference_date:
            eligible.append(symbol)

    return eligible


def filter_by_currency(tickers: Iterable[str], currency: str = "EUR") -> list[str]:
    """Return tickers whose currency matches ``currency`` according to yfinance."""
    matches: list[str] = []
    for symbol in tickers:
        info = yf.Ticker(symbol).info
        if info.get("currency") == currency:
            matches.append(symbol)
    return matches
