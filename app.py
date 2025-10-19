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
from src.portfolio import frontier, metrics, optimizer, bootstrap
from src.ui import state, widgets

st.set_page_config(
    page_title="European Portfolio Co-Pilot",
    layout="wide",
    page_icon="💡",
    initial_sidebar_state="collapsed",
)


def main() -> None:
    st.title("European Portfolio Co-Pilot")
    st.caption(
        """
        Explore leading European stocks, compare ready-made strategies, or craft your own mix.
        We crunch the numbers so you can focus on your goals."""
    )

    indices_df = get_indices_dataframe()
    if indices_df.empty:
        st.error("No index constituents could be loaded. Check your network connection.")
        return

    with st.container():
        st.subheader("1. Choose your universe")
        st.write("Pick your investment playground. We shortlist the stocks that match your filters.")

        top_cols = st.columns([2, 1])
        with top_cols[0]:
            selected_indices = widgets.multiselect_indices(sorted(config.INDEX_URLS.keys()))
        with top_cols[1]:
            min_years = st.slider(
                "Minimum listing age",
                min_value=1,
                max_value=30,
                value=config.DEFAULT_HISTORY_YEARS,
                help="Older listings give us enough history to understand how the stock behaves.",
            )
            require_eur = st.toggle("Only show EUR stocks", value=True)

        filtered = indices_df[indices_df["index_name"].isin(selected_indices)].copy()
        filtered["ticker"] = filtered.apply(_normalise_symbol, axis=1)
        filtered = filtered.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])

        tickers = filtered["ticker"].tolist()
        if not tickers:
            st.warning("No tickers available for the selected indices.")
            return

        history_ok = apply_ticker_filters(tickers, min_years, require_eur)
        filtered = filtered[filtered["ticker"].isin(history_ok)]

        with st.expander("Show company list", expanded=False):
            st.dataframe(
                filtered[["index_name", "company", "ticker"]]
                .sort_values(["index_name", "company"])
                .reset_index(drop=True),
                hide_index=True,
            )

        selection = st.multiselect(
            "Pick the companies you want to explore",
            options=filtered["ticker"],
            format_func=lambda t: f"{filtered.loc[filtered['ticker']==t, 'company'].iloc[0]} ({t})",
            default=filtered["ticker"].head(8).tolist(),
            help="You can simply keep the suggested list or build your own watchlist.",
        )
        state.set_selected_tickers(selection)
        chosen = state.get_selected_tickers()

    if len(chosen) < 2:
        st.info("Select at least two companies to build a portfolio.")
        return

    prices = load_prices(chosen)
    if prices.empty:
        st.error("Unable to download price history for the selected tickers.")
        return

    returns, mean_vector, covariance_matrix = compute_statistics(prices)

    # Build a bootstrapped efficient frontier (Font, 2016 style), cached
    bs = config.BOOTSTRAP_SETTINGS
    bs_result = get_bootstrap_result(returns, bs.runs, bs.frontier_points, bs.percentile)
    eff_frontier = bs_result.representative

    min_point = eff_frontier.points[0]
    max_point = eff_frontier.points[-1]
    mid_point = eff_frontier.points[len(eff_frontier.points) // 2]

    def _annualised(daily_return: float) -> float:
        return (1 + daily_return) ** config.TRADING_DAYS_PER_YEAR - 1

    with st.container():
        st.subheader("2. Shape your risk & return")
        st.write("Slide along the efficient frontier to balance stability and growth.")

        # Compute full-sample expected returns for displayed KPIs
        kpi_min_er = float(min_point.weights.dot(mean_vector))
        kpi_mid_er = float(mid_point.weights.dot(mean_vector))
        kpi_max_er = float(max_point.weights.dot(mean_vector))

        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric(
                "Safest mix",
                f"{_annualised(kpi_min_er):.2%}",
                help="Annualised return (full-sample) of the minimum-risk portfolio.",
            )
        with metrics_cols[1]:
            st.metric(
                "Balanced option",
                f"{_annualised(kpi_mid_er):.2%}",
                help="Annualised return (full-sample) halfway along the curve.",
            )
        with metrics_cols[2]:
            st.metric(
                "Max performance",
                f"{_annualised(kpi_max_er):.2%}",
                help="Annualised return (full-sample) of the highest-return mix.",
            )

        risk_level = st.slider(
            "Move the slider to tune your risk appetite",
            min_value=1,
            max_value=len(eff_frontier.points),
            value=1,
            help="1 = lowest risk, {} = highest risk".format(len(eff_frontier.points)),
        )
        idx = risk_level - 1
        selected_point = eff_frontier.points[idx]

        if risk_level == 1:
            focus = "Safest mix"
        elif risk_level == len(eff_frontier.points):
            focus = "Maximum performance"
        else:
            focus = "Custom balance"

        st.write(f"Focus: **{focus}**")

    # Pre-align prices once for faster re-renders
    aligned_prices = prices.ffill().dropna()

    render_portfolio_summary(
        f"Efficient frontier — step {idx + 1}/{len(eff_frontier.points)}",
        selected_point.weights,
        returns,
        aligned_prices,
        filtered,
    )

    with st.expander("Visualise the full risk/return curve", expanded=False):
        frontier_chart = portfolio_plots.efficient_frontier_plot(
            eff_frontier,
            subset=bs_result.subset,
            highlight=(selected_point.volatility, selected_point.expected_return),
        )
        st.altair_chart(frontier_chart, use_container_width=True)


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


@st.cache_data(ttl=config.CACHE_TTL)
def compute_statistics(prices: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    clean_prices = prices.ffill().dropna()
    returns = metrics.daily_returns(clean_prices).dropna(how="any")
    mean_vector = metrics.expected_daily_return(returns)
    covariance_matrix = metrics.covariance_matrix(returns)
    if covariance_matrix.size:
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6
    return returns, mean_vector, covariance_matrix


@st.cache_data(ttl=config.CACHE_TTL)
def get_bootstrap_result(
    returns_df: pd.DataFrame,
    runs: int,
    steps: int,
    percentile: float,
    *,
    seed: int | None = None,
):
    """Cached wrapper for bootstrapped frontier computation.

    Caches on the returns DataFrame and parameters to avoid recomputation
    when the user moves the risk slider.
    """
    return bootstrap.bootstrap_frontiers(
        returns_df.to_numpy(),
        runs=runs,
        steps=steps,
        percentile=percentile,
        random_state=seed,
    )


@st.cache_data(ttl=config.CACHE_TTL)
def filter_tickers(symbols: tuple[str, ...], min_years: int, require_eur: bool) -> list[str]:
    tickers = filters.filter_by_history(symbols, minimum_years=min_years)
    if require_eur:
        tickers = filters.filter_by_currency(tickers, currency="EUR")
    return tickers


def apply_ticker_filters(tickers: list[str], min_years: int, require_eur: bool) -> list[str]:
    cache_key = (min_years, require_eur)
    cache = st.session_state.setdefault("_ticker_filters", {})
    state: dict[str, bool] = cache.setdefault(cache_key, {})

    missing = [symbol for symbol in tickers if symbol not in state]
    if missing:
        with st.spinner(f"Filtering {len(missing)} new tickers..."):
            valid_symbols = set(filter_tickers(tuple(sorted(missing)), min_years, require_eur))
        for symbol in missing:
            state[symbol] = symbol in valid_symbols

    return [symbol for symbol in tickers if state.get(symbol, False)]


def render_portfolio_summary(
    label: str,
    weights: np.ndarray,
    returns: pd.DataFrame,
    aligned_prices: pd.DataFrame,
    meta: pd.DataFrame,
) -> None:
    state.set_weights(weights)

    # Fast label resolution without per-ticker DataFrame filtering
    _name_map = dict(zip(meta["ticker"].astype(str), meta["company"].astype(str)))
    asset_names = [_name_map.get(t, t) for t in aligned_prices.columns]
    portfolio_returns = returns.to_numpy().dot(weights)
    annual_return = (1 + portfolio_returns.mean()) ** config.TRADING_DAYS_PER_YEAR - 1
    annual_vol = portfolio_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = 0.0 if annual_vol == 0 else (annual_return / annual_vol)

    with st.container():
        st.subheader(f"{label} snapshot")

        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Expected annual return", f"{annual_return*100:.2f}%")
        with metric_cols[1]:
            st.metric("Expected annual volatility", f"{annual_vol*100:.2f}%")
        with metric_cols[2]:
            st.metric("Return / Risk score", f"{sharpe:.2f}")

        st.write("### Allocation overview")
        st.dataframe(
            portfolio_plots.allocation_table(weights, asset_names),
            hide_index=False,
            use_container_width=True,
        )

        portfolio_prices = aligned_prices.dot(weights)
        latest_val = portfolio_prices.iloc[-1]
        subtitle = f"Latest value: €{latest_val:,.2f} | Chart rebased to 100 at start"
        chart_price = portfolio_plots.price_evolution(
            portfolio_prices,
            title=f"{label} value (rebased)",
            subtitle=subtitle,
        )
        chart_weights = portfolio_plots.weights_pie(weights, asset_names)

        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(chart_price, use_container_width=True)
        with col2:
            st.altair_chart(chart_weights, use_container_width=True)

        st.write(
            """
            • **Return** is what you could earn on average each year if markets behave like the past.
            • **Return** is what you could earn on average each year if markets behave like the past.
            • **Volatility** shows how much your portfolio might wobble day to day. Lower means steadier.
            • The **Return / Risk score** helps compare strategies: higher values mean more return per unit of risk.
            """
        )
        )


if __name__ == "__main__":
    main()
