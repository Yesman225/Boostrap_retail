"""Portfolio metric utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def expected_daily_return(returns: pd.DataFrame) -> np.ndarray:
    return returns.mean(axis=0).to_numpy()


def covariance_matrix(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().to_numpy()


def portfolio_return(weights: np.ndarray, returns: pd.DataFrame, *, annualised: bool = False) -> float:
    daily = float(returns.dot(weights).mean())
    if annualised:
        return (1 + daily) ** TRADING_DAYS_PER_YEAR - 1
    return daily


def portfolio_volatility(weights: np.ndarray, returns: pd.DataFrame, *, annualised: bool = False) -> float:
    daily_std = float(returns.dot(weights).std())
    if annualised:
        return daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return daily_std
