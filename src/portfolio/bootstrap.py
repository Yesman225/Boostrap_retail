"""Bootstrap simulation of efficient frontiers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .frontier import EfficientFrontier, build_frontier


@dataclass
class BootstrapFrontierResult:
    representative: EfficientFrontier
    subset: list[EfficientFrontier]
    representative_index: int
    distances: np.ndarray


def bootstrap_frontiers(
    returns: np.ndarray,
    *,
    runs: int,
    steps: int,
    percentile: float = 95.0,
    random_state: int | None = None,
) -> BootstrapFrontierResult:
    returns = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(random_state)
    n_obs, _ = returns.shape

    sample_indices = rng.integers(0, n_obs, size=(runs, n_obs))

    features = np.empty((runs, steps * 2), dtype=float)
    frontiers: list[EfficientFrontier] = []

    for i, indices in enumerate(sample_indices):
        sample = returns[indices]
        mean = sample.mean(axis=0)
        covariance = np.cov(sample, rowvar=False)
        frontier = build_frontier(mean, covariance, steps=steps)
        frontiers.append(frontier)
        er = frontier.expected_returns()
        vol = frontier.volatilities()
        features[i, 0::2] = er
        features[i, 1::2] = vol

    row_means = features.mean(axis=1, keepdims=True)
    row_stds = features.std(axis=1, keepdims=True)
    standardized = (features - row_means) / np.clip(row_stds, a_min=1e-12, a_max=None)

    distances = squareform(pdist(standardized, metric="euclidean")).sum(axis=1)
    representative_index = int(np.argmin(distances))
    representative = frontiers[representative_index]

    cov_features = np.cov(features, rowvar=False)
    inv_cov = np.linalg.pinv(cov_features)

    centred = features - features[representative_index]
    mahalanobis = np.sqrt(np.einsum("ij,jk,ik->i", centred, inv_cov, centred))

    threshold = np.percentile(mahalanobis, percentile)
    subset = [frontiers[i] for i in range(runs) if mahalanobis[i] <= threshold]

    return BootstrapFrontierResult(
        representative=representative,
        subset=subset,
        representative_index=representative_index,
        distances=mahalanobis,
    )
