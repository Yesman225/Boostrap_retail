"""Ticker filtering utilities with concurrency to speed up yfinance calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Iterable, Sequence

import yfinance as yf
from dateutil.relativedelta import relativedelta


def filter_by_history(
    tickers: Sequence[str],
    *,
    minimum_years: int,
    today: date | None = None,
    max_workers: int = 5,
) -> list[str]:
    """Return tickers that have at least ``minimum_years`` of price history."""
    if not tickers:
        return []

    reference_date = (today or date.today()) - relativedelta(years=minimum_years)

    def _meets_history(symbol: str) -> str | None:
        try:
            data = yf.download(
                symbol,
                period="max",
                interval="1mo",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:  # pragma: no cover - network issues
            return None
        if data.empty:
            return None
        first_date = data.index.min().date()
        return symbol if first_date <= reference_date else None

    eligible: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_meets_history, symbol): symbol for symbol in tickers}
        for future in as_completed(future_map):
            try:
                result = future.result()
            except Exception:  # pragma: no cover - defensive
                continue
            if result:
                eligible.append(result)

    return eligible


def filter_by_currency(
    tickers: Iterable[str],
    currency: str = "EUR",
    *,
    max_workers: int = 5,
) -> list[str]:
    """Return tickers whose trading currency matches ``currency``."""

    def _matches_currency(symbol: str) -> str | None:
        try:
            ticker = yf.Ticker(symbol)
            info = getattr(ticker, "fast_info", {}) or {}
            curr = info.get("currency")
            if curr is None:
                curr = ticker.info.get("currency")
        except Exception:  # pragma: no cover - network issues
            return None
        return symbol if curr == currency else None

    matches: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_matches_currency, symbol): symbol for symbol in tickers}
        for future in as_completed(future_map):
            try:
                result = future.result()
            except Exception:  # pragma: no cover
                continue
            if result:
                matches.append(result)

    return matches
