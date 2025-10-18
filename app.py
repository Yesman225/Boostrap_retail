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
from src.portfolio import frontier, metrics, optimizer
from src.ui import state, widgets

st.set_page_config(
    page_title="European Portfolio Co-Pilot",
    layout="wide",
    page_icon="💡",
    initial_sidebar_state="collapsed",
)


def inject_global_styles() -> None:
    colors = config.THEME_COLORS
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

            html, body, [class^="css"]  {{
                font-family: 'Poppins', sans-serif;
                background-color: {colors["bg_light"]};
            }}

            .stApp {{
                background-color: {colors["bg_light"]};
            }}

            .section-card {{
                background-color: white;
                padding: 1.5rem;
                border-radius: 18px;
                box-shadow: 0 10px 40px rgba(15, 23, 42, 0.08);
                margin-bottom: 1.5rem;
                border: 1px solid rgba(15, 23, 42, 0.04);
            }}

            .metric-card {{
                padding: 1rem 1.2rem;
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(124, 58, 237, 0.12));
                border: 1px solid rgba(37, 99, 235, 0.1);
                color: #0f172a;
            }}

            .badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 600;
                background-color: rgba(37, 99, 235, 0.1);
                color: {colors["primary"]};
            }}

            .stButton button {{
                border-radius: 999px;
                padding: 0.6rem 1.3rem;
                font-weight: 600;
                border: none;
                background: linear-gradient(135deg, {colors["primary"]}, {colors["accent"]});
                color: white;
                box-shadow: 0 12px 30px rgba(37, 99, 235, 0.25);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }}

            .stButton button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 18px 40px rgba(37, 99, 235, 0.35);
            }}

            .block-container {{
                padding-top: 1.5rem;
            }}

            .stRadio > div {{
                background-color: transparent;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_global_styles()

    st.markdown(
        """
        <div class="badge">Smart guidance</div>
        <h1 style="margin-top:0.6rem;">European Portfolio Co-Pilot</h1>
        <p style="font-size:0.95rem;color:#475569;max-width:680px;">
            Explore leading European stocks, compare ready-made strategies, or craft your own mix.
            We crunch the numbers so you can focus on your goals.
        </p>
        """,
        unsafe_allow_html=True,
    )

    indices_df = get_indices_dataframe()
    if indices_df.empty:
        st.error("No index constituents could be loaded. Check your network connection.")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>1. Choose your universe</h3>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)
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

    st.markdown("</div>", unsafe_allow_html=True)

    if len(chosen) < 2:
        st.info("Select at least two companies to build a portfolio.")
        return

    prices = load_prices(chosen)
    if prices.empty:
        st.error("Unable to download price history for the selected tickers.")
        return

    returns, mean_vector, covariance_matrix = compute_statistics(prices)

    eff_frontier = frontier.build_frontier(
        mean_vector,
        covariance_matrix,
        steps=config.FRONTIER_POINTS,
    )

    min_point = eff_frontier.points[0]
    max_point = eff_frontier.points[-1]
    mid_point = eff_frontier.points[len(eff_frontier.points) // 2]

    def _annualised(daily_return: float) -> float:
        return (1 + daily_return) ** config.TRADING_DAYS_PER_YEAR - 1

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>2. Shape your risk & return</h3>", unsafe_allow_html=True)
    st.write("Slide along the efficient frontier to balance confidence and ambition.")

    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric(
            "Safest mix",
            f"{_annualised(min_point.expected_return):.2%}",
            help="Annualised return of the minimum-risk portfolio.",
        )
    with metrics_cols[1]:
        st.metric(
            "Balanced option",
            f"{_annualised(mid_point.expected_return):.2%}",
            help="Return halfway along the curve.",
        )
    with metrics_cols[2]:
        st.metric(
            "Max performance",
            f"{_annualised(max_point.expected_return):.2%}",
            help="Annualised return of the highest-return mix.",
        )

    risk_level = st.slider(
        "Move the slider to tune your risk appetite",
        min_value=0,
        max_value=100,
        value=0,
        help="0% keeps risk at its minimum. 100% pursues the highest return on the curve.",
    )
    idx = int(round(risk_level / 100 * (len(eff_frontier.points) - 1)))
    selected_point = eff_frontier.points[idx]

    if risk_level == 0:
        focus = "Safest mix"
    elif risk_level == 100:
        focus = "Maximum performance"
    else:
        focus = "Custom balance"

    st.markdown(f"<p style='margin-top:0.6rem;color:#64748b;'>Focus: <strong>{focus}</strong></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    render_portfolio_summary(
        f"Efficient frontier — step {idx + 1}/{len(eff_frontier.points)}",
        selected_point.weights,
        returns,
        prices,
        filtered,
    )

    with st.expander("Visualise the full risk/return curve", expanded=False):
        fig = portfolio_plots.efficient_frontier_plot(
            eff_frontier,
            highlight=(selected_point.volatility, selected_point.expected_return),
        )
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
    prices: pd.DataFrame,
    meta: pd.DataFrame,
) -> None:
    state.set_weights(weights)

    asset_names = [meta.loc[meta["ticker"] == ticker, "company"].iloc[0] for ticker in prices.columns]
    portfolio_returns = returns.to_numpy().dot(weights)
    annual_return = (1 + portfolio_returns.mean()) ** config.TRADING_DAYS_PER_YEAR - 1
    annual_vol = portfolio_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = 0.0 if annual_vol == 0 else (annual_return / annual_vol)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-top:0;'>{label} portfolio snapshot</h3>", unsafe_allow_html=True)

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

    aligned_prices = prices.ffill().dropna()
    portfolio_prices = aligned_prices.dot(weights)
    subtitle = f"Latest value: {portfolio_prices.iloc[-1]:.2f}"
    fig_price = portfolio_plots.price_evolution(
        portfolio_prices,
        title=f"{label} portfolio value",
        subtitle=subtitle,
    )
    fig_weights = portfolio_plots.weights_pie(weights, asset_names)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_price)
    with col2:
        st.pyplot(fig_weights)

    st.markdown("### What this means for you")
    st.write(
        """
        • **Return** is what you could earn on average each year if markets behave like the past. \n
        • **Volatility** shows how much your portfolio might wobble day to day. Lower means steadier.\n
        • The **Return / Risk score** helps compare strategies: higher values mean more return per unit of risk.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
