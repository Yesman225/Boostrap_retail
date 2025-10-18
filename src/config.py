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
    "primary": "#6366F1",
    "accent": "#8B5CF6",
    "badge_bg": "rgba(99,102,241,0.18)",
    "section_bg": "rgba(255,255,255,0.6)",
    "border": "rgba(148,163,184,0.35)",
    "metric_bg": "rgba(99,102,241,0.18)",
    "text_primary": "inherit",
    "text_secondary": "inherit",
    "tag_bg": "#EF4444",
    "tag_text": "#FFFFFF",
    "slider_active": "#6366F1",
    "slider_thumb": "#EF4444",
}


@dataclass(frozen=True)
class BootstrapSettings:
    runs: int = DEFAULT_BOOTSTRAP_RUNS
    frontier_points: int = FRONTIER_POINTS
    percentile: float = 95.0


BOOTSTRAP_SETTINGS: Final[BootstrapSettings] = BootstrapSettings()
