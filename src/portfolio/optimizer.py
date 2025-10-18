"""Modern portfolio optimisation helpers."""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.optimize as opt

Bounds = tuple[tuple[float, float], ...]
Constraint = dict[str, Callable[[np.ndarray], float]]


def equal_weight_vector(size: int) -> np.ndarray:
    return np.full(shape=size, fill_value=1.0 / size)


def min_volatility(mean: np.ndarray, covariance: np.ndarray, *, bounds: Bounds | None = None) -> np.ndarray:
    """Return weights minimising portfolio volatility subject to full investment."""
    return _optimise(
        mean,
        covariance,
        objective=_volatility_objective,
        bounds=bounds or _default_bounds(len(mean)),
        constraints=(
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ),
    )


def max_return(mean: np.ndarray, covariance: np.ndarray, *, bounds: Bounds | None = None) -> np.ndarray:
    """Return weights maximising portfolio return subject to full investment."""
    return _optimise(
        mean,
        covariance,
        objective=lambda w: -w.dot(mean),
        bounds=bounds or _default_bounds(len(mean)),
        constraints=(
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ),
    )


def min_volatility_for_return(
    mean: np.ndarray,
    covariance: np.ndarray,
    target_return: float,
    *,
    bounds: Bounds | None = None,
) -> np.ndarray:
    """Return weights with minimum volatility for a target daily return."""
    return _optimise(
        mean,
        covariance,
        objective=_volatility_objective,
        bounds=bounds or _default_bounds(len(mean)),
        constraints=(
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: float(w.dot(mean)) - target_return},
        ),
    )


def _optimise(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    objective: Callable[[np.ndarray], float],
    bounds: Bounds,
    constraints: tuple[Constraint, ...],
) -> np.ndarray:
    size = len(mean)
    initial = equal_weight_vector(size)

    result = opt.minimize(
        fun=lambda w: objective(w.astype(float)),
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        args=(),
        options={"maxiter": 200, "ftol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")
    return result.x


def _volatility_objective(weights: np.ndarray) -> float:
    return _portfolio_variance(weights) ** 0.5


def _portfolio_variance(weights: np.ndarray) -> float:
    raise NotImplementedError("Variance objective requires covariance matrix context")


def _default_bounds(size: int) -> Bounds:
    return tuple((0.0, 1.0) for _ in range(size))
