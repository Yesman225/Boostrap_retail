"""Data source helpers for retrieving index constituents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd
import requests

DEFAULT_HEADERS: Mapping[str, str] = {
    "User-Agent": "RetailPortfolioApp/1.0 (contact@example.com)",
}


@dataclass(frozen=True)
class IndexConstituent:
    index_name: str
    company: str
    ticker: str | None


def fetch_index_components(index_name: str, url: str, *, headers: Mapping[str, str] | None = None) -> list[IndexConstituent]:
    """Return constituents for a given index scraped from Wikipedia."""
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}
    response = requests.get(url, headers=merged_headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(response.text)

    constituents: list[IndexConstituent] = []
    candidate_name_columns = ("Company", "Security", "Name")
    candidate_ticker_columns = ("Ticker", "Symbol", "Ticker symbol")

    for table in tables:
        name_column = _first_existing_column(table.columns, candidate_name_columns)
        ticker_column = _first_existing_column(table.columns, candidate_ticker_columns)

        if name_column and ticker_column:
            for _, row in table.iterrows():
                company = str(row.get(name_column, "")).strip()
                ticker = row.get(ticker_column)
                if isinstance(ticker, float) and pd.isna(ticker):  # pragma: no cover - guard
                    ticker = None
                ticker_value = str(ticker).strip() if ticker else None
                if company:
                    constituents.append(IndexConstituent(index_name, company, ticker_value))
            if constituents:
                break

    return constituents


def load_indices(index_urls: Mapping[str, str]) -> pd.DataFrame:
    """Return dataframe of index constituents for all provided URLs."""
    records: list[IndexConstituent] = []
    for index_name, url in index_urls.items():
        records.extend(fetch_index_components(index_name, url))

    return pd.DataFrame([r.__dict__ for r in records])


def _first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None
