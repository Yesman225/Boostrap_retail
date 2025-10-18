"""Streamlit entrypoint for the retail portfolio optimiser."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

from src import config
from src.data import fetch, filters, sources
from src.plotting import portfolio as portfolio_plots
from src.portfolio import bootstrap, frontier, metrics, optimizer
from src.ui import state, widgets

st.set_page_config(page_title="European Stocks Selector & Optimiser", layout="wide")


def main() -> None:
    st.title("European Stocks Selector & Optimiser")
    st.caption("Pick your universe, explore optimal portfolios, and visualise allocations.")

    indices_df = get_indices_dataframe()
    if indices_df.empty:
        st.error("No index constituents could be loaded. Check your network connection.")
        return

    with st.sidebar:
        st.header("Universe")
        selected_indices = widgets.multiselect_indices(sorted(config.INDEX_URLS.keys()))
        min_years = st.slider("Minimum price history (years)", min_value=1, max_value=30, value=config.DEFAULT_HISTORY_YEARS)
        require_eur = st.checkbox("Only keep EUR-denominated stocks", value=True)

    filtered = indices_df[indices_df["index_name"].isin(selected_indices)].copy()
    filtered["ticker"] = filtered.apply(_normalise_symbol, axis=1)
    filtered = filtered.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])

    tickers = filtered["ticker"].tolist()
    if not tickers:
        st.warning("No tickers available for the selected indices.")
        return

    with st.spinner("Filtering tickers..."):
        history_ok = filters.filter_by_history(tickers, minimum_years=min_years)
        if require_eur:
            history_ok = filters.filter_by_currency(history_ok, currency="EUR")

    filtered = filtered[filtered["ticker"].isin(history_ok)]

    st.subheader("Available securities")
    st.write(
        filtered[["index_name", "company", "ticker"]]
        .sort_values(["index_name", "company"])
        .reset_index(drop=True)
    )

    selection = st.multiselect(
        "Choose stocks for optimisation",
        options=filtered["ticker"],
        format_func=lambda t: f"{filtered.loc[filtered['ticker']==t, 'company'].iloc[0]} ({t})",
        default=filtered["ticker"].head(8).tolist(),
    )
    state.set_selected_tickers(selection)
    chosen = state.get_selected_tickers()

    if len(chosen) < 2:
        st.info("Select at least two securities to build a portfolio.")
        return

    prices = load_prices(chosen)
    if prices.empty:
        st.error("Unable to download price history for the selected tickers.")
        return

    returns = metrics.daily_returns(prices)
    mean_vector = metrics.expected_daily_return(returns)
    covariance_matrix = metrics.covariance_matrix(returns)

    st.subheader("Portfolio Optimisation")
    approach = st.selectbox(
        "Select optimisation approach",
        ("Minimum Risk", "Maximum Return", "Target Return", "Bootstrap"),
    )

    if approach == "Minimum Risk":
        weights = optimizer.min_volatility(mean_vector, covariance_matrix)
        label = "Minimum Risk"
    elif approach == "Maximum Return":
        weights = optimizer.max_return(mean_vector, covariance_matrix)
        label = "Maximum Return"
    elif approach == "Target Return":
        weights, label = optimise_for_target(mean_vector, covariance_matrix)
        if weights is None:
            return
    else:
        result = bootstrap.bootstrap_frontiers(
            returns.to_numpy(),
            runs=config.BOOTSTRAP_SETTINGS.runs,
            steps=config.BOOTSTRAP_SETTINGS.frontier_points,
        )
        weights = result.representative.points[0].weights
        label = "Bootstrap Clusteroid"

    render_portfolio_summary(label, weights, returns, prices, filtered)

    with st.expander("Efficient frontier", expanded=False):
        base_frontier = frontier.build_frontier(
            mean_vector,
            covariance_matrix,
            steps=config.FRONTIER_POINTS,
        )
        fig = portfolio_plots.efficient_frontier_plot(base_frontier)
        st.pyplot(fig)


def get_indices_dataframe() -> pd.DataFrame:
    @st.cache_data(ttl=config.CACHE_TTL)
    def _load() -> pd.DataFrame:
        df = sources.load_indices(config.INDEX_URLS)
        return df

    return _load()


def _normalise_symbol(row: pd.Series) -> str | None:
    symbol = row.get("ticker")
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    symbol = symbol.strip()
    index_name = row.get("index_name")
    if index_name == "AEX 25" and not symbol.endswith(".AS"):
        return f"{symbol}.AS"
    return symbol


def load_prices(tickers: list[str]) -> pd.DataFrame:
    start = date.today() - relativedelta(years=30)

    @st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
    def _download(symbols: tuple[str, ...]) -> pd.DataFrame:
        data = fetch.download_price_history(symbols, start=start, end=date.today())
        return data

    return _download(tuple(sorted(tickers)))


def optimise_for_target(mean_vector: np.ndarray, covariance_matrix: np.ndarray) -> tuple[np.ndarray | None, str]:
    base_frontier = frontier.build_frontier(mean_vector, covariance_matrix, steps=config.FRONTIER_POINTS)
    min_return = base_frontier.points[0].expected_return
    max_return = base_frontier.points[-1].expected_return

    min_annual = (1 + min_return) ** config.TRADING_DAYS_PER_YEAR - 1
    max_annual = (1 + max_return) ** config.TRADING_DAYS_PER_YEAR - 1

    target_annual = st.slider(
        "Target annual return (%)",
        min_value=float(min_annual * 100),
        max_value=float(max_annual * 100),
        value=float((min_annual + max_annual) / 2 * 100),
    )

    target_daily = (1 + target_annual / 100) ** (1 / config.TRADING_DAYS_PER_YEAR) - 1

    try:
        weights = optimizer.min_volatility_for_return(mean_vector, covariance_matrix, target_daily)
    except RuntimeError as exc:  # pragma: no cover - streamlit feedback
        st.error(f"Optimisation failed: {exc}")
        return None, "Target Return"

    return weights, f"Target Return ({target_annual:.2f}% annual)"


def render_portfolio_summary(
    label: str,
    weights: np.ndarray,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    meta: pd.DataFrame,
) -> None:
    state.set_weights(weights)
    asset_names = [meta.loc[meta["ticker"] == ticker, "company"].iloc[0] for ticker in prices.columns]

    portfolio_returns = returns.to_numpy().dot(weights)
    annual_return = (1 + portfolio_returns.mean()) ** config.TRADING_DAYS_PER_YEAR - 1
    annual_vol = portfolio_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)

    st.markdown(f"### {label} Portfolio")
    st.write(f"Annualised return: **{annual_return*100:.2f}%**")
    st.write(f"Annualised volatility: **{annual_vol*100:.2f}%**")

    table = portfolio_plots.allocation_table(weights, asset_names)
    st.dataframe(table)

    latest_prices = prices.dot(weights)
    subtitle = f"Latest value: {latest_prices.iloc[-1]:.2f}"
    fig_price = portfolio_plots.price_evolution(latest_prices, title=f"{label} Portfolio Value", subtitle=subtitle)
    fig_weights = portfolio_plots.weights_pie(weights, asset_names)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_price)
    with col2:
        st.pyplot(fig_weights)


if __name__ == "__main__":
    main()
