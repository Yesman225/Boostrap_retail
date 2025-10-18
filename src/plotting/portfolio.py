"""Plotting helpers using Altair for Streamlit-friendly visuals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import altair as alt
import numpy as np
import pandas as pd

from src.config import THEME_COLORS
from src.portfolio.frontier import EfficientFrontier


@dataclass
class PortfolioPlotBundle:
    price_chart: alt.Chart
    allocation_chart: alt.Chart
    allocation_table: pd.DataFrame


def allocation_table(weights: Sequence[float], labels: Sequence[str]) -> pd.DataFrame:
    data = {"Repartition": [f"{w:.1%}" for w in weights]}
    df = pd.DataFrame(data, index=labels)
    df.index.name = "Stock"
    return df


def weights_pie(weights: Sequence[float], labels: Sequence[str]) -> alt.Chart:
    weights = np.asarray(weights)
    mask = weights > 0.002
    filtered_weights = weights[mask]
    filtered_labels = np.asarray(labels)[mask]

    chart_data = pd.DataFrame({"Stock": filtered_labels, "Weight": filtered_weights})

    chart = (
        alt.Chart(chart_data)
        .mark_arc(outerRadius=120, innerRadius=60)
        .encode(
            theta="Weight",
            color=alt.Color("Stock", legend=alt.Legend(title="Holdings"), scale=alt.Scale(scheme="purples")),
            tooltip=["Stock", alt.Tooltip("Weight", format=".2%")],
        )
    )

    return chart


def price_evolution(prices: pd.Series, *, title: str, subtitle: str) -> alt.Chart:
    series = prices.sort_index().dropna()
    if series.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "Value": []})).mark_line()

    if len(series) > 600:
        series = series.resample("M").last().dropna()

    base = series.iloc[0]
    rebased = series / base * 100 if base != 0 else series.copy()

    chart_data = pd.DataFrame({"Date": rebased.index, "Value": rebased.values})

    chart = (
        alt.Chart(chart_data)
        .mark_line(color=THEME_COLORS["primary"], interpolate="monotone")
        .encode(x="Date:T", y=alt.Y("Value:Q", title="Rebased value (start = 100)"), tooltip=["Date:T", alt.Tooltip("Value:Q", format=",.0f")])
        .properties(title=alt.TitleParams(text=title, subtitle=subtitle, anchor="start"))
    )

    area = chart.mark_area(opacity=0.15)
    return area + chart


def efficient_frontier_plot(
    frontier: EfficientFrontier,
    *,
    subset: Iterable[EfficientFrontier] | None = None,
    representative_colour: str | None = None,
    highlight: tuple[float, float] | None = None,
) -> alt.Chart:
    colors = THEME_COLORS
    curve_colour = representative_colour or colors["accent"]

    base = frontier.as_dataframe().assign(Source="Representative")
    chart_data = base.copy()

    layers: list[alt.Chart] = []

    if subset:
        subset_frames = []
        for candidate in subset:
            subset_frames.append(candidate.as_dataframe().assign(Source="Scenario"))
        if subset_frames:
            subset_df = pd.concat(subset_frames, ignore_index=True)
            scenarios = (
                alt.Chart(subset_df)
                .mark_point(color="lightgray", opacity=0.35)
                .encode(x="volatility", y="expected_return")
            )
            layers.append(scenarios)

    frontier_line = (
        alt.Chart(chart_data)
        .mark_line(color=curve_colour, point=alt.OverlayMarkDef(color="white", size=70, stroke=curve_colour))
        .encode(x=alt.X("volatility", title="Volatility (risk)"), y=alt.Y("expected_return", title="Expected return"), tooltip=[alt.Tooltip("volatility", format=".2%"), alt.Tooltip("expected_return", format=".2%")])
    )
    layers.append(frontier_line)

    if highlight is not None:
        highlight_df = pd.DataFrame({"volatility": [highlight[0]], "expected_return": [highlight[1]]})
        highlight_chart = (
            alt.Chart(highlight_df)
            .mark_point(color=colors["primary"], size=120, filled=True)
            .encode(x="volatility", y="expected_return")
        )
        layers.append(highlight_chart)

    return alt.layer(*layers).properties(title="Risk vs return map")
