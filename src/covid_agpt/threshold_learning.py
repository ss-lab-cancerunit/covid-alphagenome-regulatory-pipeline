"""One-dimensional K-means threshold learning.

This module contains the two implementations used for the formal analyses:

* ``deterministic``: deterministic multi-start Lloyd-style 1D K-means used
  for the protein-coding analysis.
* ``sklearn``: scikit-learn KMeans with a fixed seed and 10 initialisations,
  used for the lncRNA analysis.

Fisher-Jenks code from an earlier development branch is preserved only in
``legacy/thresholds`` and is not part of the formal pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

Backend = Literal["deterministic", "sklearn"]


@dataclass(frozen=True)
class KMeans1DResult:
    centers: np.ndarray
    cutoffs: np.ndarray
    labels: np.ndarray
    wcss: float


def clean_values(values: Iterable[float]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    array = series.to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _finalize(values: np.ndarray, centers: np.ndarray) -> KMeans1DResult:
    centers = np.sort(np.asarray(centers, dtype=float))
    cutoffs = (centers[:-1] + centers[1:]) / 2.0
    labels = np.searchsorted(cutoffs, values, side="right")
    wcss = 0.0
    for group in range(len(centers)):
        group_values = values[labels == group]
        if len(group_values):
            wcss += float(((group_values - group_values.mean()) ** 2).sum())
    return KMeans1DResult(centers, cutoffs, labels, wcss)


def deterministic_1d_kmeans(
    values: Iterable[float],
    k: int,
    n_init: int = 20,
    max_iter: int = 300,
) -> KMeans1DResult:
    """Reproduce the deterministic multi-start implementation used for CD."""
    x = clean_values(values)
    if len(x) == 0:
        raise ValueError("No finite values were supplied")
    if k < 2:
        raise ValueError("k must be at least 2")
    if len(np.unique(x)) < k:
        raise ValueError(f"Only {len(np.unique(x))} unique values for k={k}")

    base_quantiles = np.linspace(0, 1, k + 2)[1:-1]
    initializations = [
        np.quantile(x, np.clip(base_quantiles + shift, 0.01, 0.99))
        for shift in np.linspace(-0.08, 0.08, n_init)
    ]
    initializations.append(np.linspace(np.min(x), np.max(x), k))

    best: KMeans1DResult | None = None
    for initial in initializations:
        centers = np.asarray(initial, dtype=float)
        for _ in range(max_iter):
            labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
            updated = centers.copy()
            for group in range(k):
                if np.any(labels == group):
                    updated[group] = x[labels == group].mean()
            if np.allclose(updated, centers, rtol=1e-10, atol=1e-12):
                centers = updated
                break
            centers = updated

        result = _finalize(x, centers)
        if best is None or result.wcss < best.wcss:
            best = result

    if best is None:  # pragma: no cover - guarded by the validation above
        raise RuntimeError("K-means failed to produce a solution")
    return best


def sklearn_1d_kmeans(
    values: Iterable[float],
    k: int,
    random_state: int = 42,
    n_init: int = 10,
) -> KMeans1DResult:
    """Reproduce the scikit-learn implementation used for lncRNA."""
    from sklearn.cluster import KMeans

    x = clean_values(values)
    if len(x) == 0:
        raise ValueError("No finite values were supplied")
    if len(np.unique(x)) < k:
        raise ValueError(f"Only {len(np.unique(x))} unique values for k={k}")

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        algorithm="lloyd",
    )
    model.fit(x.reshape(-1, 1))
    return _finalize(x, model.cluster_centers_.ravel())


def fit_1d_kmeans(
    values: Iterable[float],
    k: int,
    backend: Backend,
) -> KMeans1DResult:
    if backend == "deterministic":
        return deterministic_1d_kmeans(values, k)
    if backend == "sklearn":
        return sklearn_1d_kmeans(values, k)
    raise ValueError(f"Unsupported backend: {backend}")


def group_statistics(
    values: Iterable[float], result: KMeans1DResult
) -> pd.DataFrame:
    x = clean_values(values)
    rows = []
    for group in range(len(result.centers)):
        group_values = x[result.labels == group]
        rows.append(
            {
                "group": group,
                "count": len(group_values),
                "min": float(np.min(group_values)),
                "median": float(np.median(group_values)),
                "mean": float(np.mean(group_values)),
                "max": float(np.max(group_values)),
            }
        )
    return pd.DataFrame(rows)
