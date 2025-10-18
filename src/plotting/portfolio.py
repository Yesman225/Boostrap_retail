"""Plotting helpers for portfolio visuals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from src.config import THEME_COLORS
from src.portfolio.frontier import EfficientFrontier


@dataclass
class PortfolioPlotBundle:
    price_figure: plt.Figure
    weights_figure: plt.Figure
    allocation_table: pd.DataFrame


plt.rcParams.update(
    {
        "font.family": "Poppins",
        "axes.edgecolor": "#e2e8f0",
        "axes.linewidth": 1.0,
        "axes.titlesize": 12,
        "axes.titlecolor": "#0f172a",
        "axes.labelcolor": "#475569",
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "grid.color": "#e2e8f0",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def allocation_table(weights: Sequence[float], labels: Sequence[str]) -> pd.DataFrame:
    data = {"Repartition": [f"{w:.1%}" for w in weights]}
    df = pd.DataFrame(data, index=labels)
    df.index.name = "Stock"
    return df


def weights_pie(weights: Sequence[float], labels: Sequence[str]) -> plt.Figure:
    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(weights)))
    weights = np.asarray(weights)
    mask = weights > 0.002
    filtered_weights = weights[mask]
    filtered_labels = np.asarray(labels)[mask]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        filtered_weights,
        labels=None,
        startangle=110,
        colors=colors[: len(filtered_weights)],
        autopct="%1.1f%%",
        pctdistance=0.75,
        textprops={"color": "#0f172a", "fontsize": 9},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("600")

    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_artist(centre_circle)

    ax.legend(
        wedges,
        filtered_labels,
        title="Holdings",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.4, 1),
        frameon=False,
    )
    ax.set_title("Portfolio allocation", fontweight="600")
    return fig


def price_evolution(prices: pd.Series, *, title: str, subtitle: str) -> plt.Figure:
    colors = THEME_COLORS
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        prices.index,
        prices.values,
        color=colors["primary"],
        linewidth=2.5,
        alpha=0.9,
    )
    ax.fill_between(
        prices.index,
        prices.values,
        color=colors["primary"],
        alpha=0.12,
    )
    ax.set_title(title, loc="left", fontweight="600")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value (€)")
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.4)

    fig.text(0.02, 0.93, subtitle, fontsize=10, color="#475569")
    fig.tight_layout()
    return fig


def efficient_frontier_plot(
    frontier: EfficientFrontier,
    *,
    subset: Iterable[EfficientFrontier] | None = None,
    representative_colour: str = THEME_COLORS["accent"],
    highlight: tuple[float, float] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    if subset:
        for candidate in subset:
            data = candidate.as_dataframe()
            ax.scatter(
                data["volatility"],
                data["expected_return"],
                color="#cbd5f5",
                alpha=0.25,
                s=20,
            )

    df = frontier.as_dataframe()
    ax.plot(
        df["volatility"],
        df["expected_return"],
        color=representative_colour,
        linewidth=2.5,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.8,
    )
    if highlight is not None:
        ax.scatter(
            highlight[0],
            highlight[1],
            color=THEME_COLORS["primary"],
            s=120,
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )

    ax.set_xlabel("Volatility (risk)")
    ax.set_ylabel("Expected return")
    ax.set_title("Risk vs return map", fontweight="600")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    return fig
