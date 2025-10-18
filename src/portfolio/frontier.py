"""Efficient frontier construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import optimizer


@dataclass
class FrontierPoint:
    weights: np.ndarray
    expected_return: float
    volatility: float


@dataclass
class EfficientFrontier:
    points: list[FrontierPoint]

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "expected_return": [p.expected_return for p in self.points],
                "volatility": [p.volatility for p in self.points],
            }
        )

    def weight_matrix(self) -> np.ndarray:
        return np.vstack([p.weights for p in self.points])


def build_frontier(mean: np.ndarray, covariance: np.ndarray, *, steps: int = 20) -> EfficientFrontier:
    min_weights = optimizer.min_volatility(mean, covariance)
    max_weights = optimizer.max_return(mean, covariance)

    min_return = float(min_weights.dot(mean))
    max_return = float(max_weights.dot(mean))

    returns = np.linspace(min_return, max_return, steps)

    points: list[FrontierPoint] = []
    for target in returns:
        weights = optimizer.min_volatility_for_return(mean, covariance, target)
        expected_return = float(weights.dot(mean))
        volatility = float(np.sqrt(weights.T @ covariance @ weights))
        points.append(FrontierPoint(weights, expected_return, volatility))

    return EfficientFrontier(points)


def annualise_frontier(frontier: EfficientFrontier, periods_per_year: int) -> EfficientFrontier:
    annualised: list[FrontierPoint] = []
    for point in frontier.points:
        annual_return = (1 + point.expected_return) ** periods_per_year - 1
        annual_vol = point.volatility * np.sqrt(periods_per_year)
        annualised.append(FrontierPoint(point.weights, annual_return, annual_vol))
    return EfficientFrontier(annualised)
