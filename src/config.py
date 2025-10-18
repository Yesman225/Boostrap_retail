"""Application configuration constants and defaults."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Sequence

INDEX_URLS: Final[dict[str, str]] = {
    "CAC 40": "https://en.wikipedia.org/wiki/CAC_40",
    "DAX 40": "https://en.wikipedia.org/wiki/DAX",
    "AEX 25": "https://en.wikipedia.org/wiki/AEX_index",
    "IBEX 35": "https://en.wikipedia.org/wiki/IBEX_35",
    "FTSE MIB": "https://en.wikipedia.org/wiki/FTSE_MIB",
}

HORIZONS: Final[Sequence[str]] = ("1M", "6M", "1Y", "5Y", "10Y", "20Y", "MAX")

CACHE_TTL: Final[int] = 60 * 60 * 24  # 24 hours in seconds
DEFAULT_HISTORY_YEARS: Final[int] = 20
DEFAULT_BOOTSTRAP_RUNS: Final[int] = 100
FRONTIER_POINTS: Final[int] = 20
TRADING_DAYS_PER_YEAR: Final[int] = 252


THEME_COLORS: Final[dict[str, str]] = {
    "primary": "#2563EB",  # indigo-500
    "accent": "#7C3AED",   # purple-500
    "bg_light": "#F8FAFC",  # slate-50
    "bg_dark": "#0F172A",   # slate-900
    "positive": "#16A34A",  # green-500
    "negative": "#DC2626",  # red-600
}


@dataclass(frozen=True)
class BootstrapSettings:
    runs: int = DEFAULT_BOOTSTRAP_RUNS
    frontier_points: int = FRONTIER_POINTS
    percentile: float = 95.0


BOOTSTRAP_SETTINGS: Final[BootstrapSettings] = BootstrapSettings()
