"""Label-free global POI-grid selection for shuffled polynomial traces.

The selector is deliberately different from selecting an independently
maximally separated sample inside every coefficient window.  A physical
implementation has a common execution cadence, so the selector searches a
single global grid

    p_k = round(offset + cadence * k),

and scores the same offset across all 256 slots.  The score uses only the
waveform calibration traces: a two-cluster standardized separation at each
candidate POI, aggregated robustly over slots.  No coefficient labels,
secret, execution order, or public algebraic vector is accepted by this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class POIGridCandidate:
    cadence: float
    offset: float
    score: float
    median_score: float
    fold_scores: tuple[float, ...]
    fold_winners: tuple[tuple[float, float], ...]
    indices: np.ndarray


def _column_two_means(values: np.ndarray, iterations: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit independent two-means models to columns without labels."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 4:
        raise ValueError("values must have shape (calibration_traces, poi_count)")
    low = np.quantile(x, 0.25, axis=0)
    high = np.quantile(x, 0.75, axis=0)
    for _ in range(int(iterations)):
        assignment = np.abs(x - low[None, :]) > np.abs(x - high[None, :])
        low_count = np.maximum(np.sum(~assignment, axis=0), 1)
        high_count = np.maximum(np.sum(assignment, axis=0), 1)
        new_low = np.sum(np.where(assignment, 0.0, x), axis=0) / low_count
        new_high = np.sum(np.where(assignment, x, 0.0), axis=0) / high_count
        low, high = new_low, new_high
    assignment = np.abs(x - low[None, :]) > np.abs(x - high[None, :])
    low_count = np.maximum(np.sum(~assignment, axis=0), 1)
    high_count = np.maximum(np.sum(assignment, axis=0), 1)
    var_low = np.sum(np.where(assignment, 0.0, (x - low[None, :]) ** 2), axis=0) / low_count
    var_high = np.sum(np.where(assignment, (x - high[None, :]) ** 2, 0.0), axis=0) / high_count
    separation = np.abs(high - low) / np.sqrt(var_low + var_high + 1e-8)
    separation[(low_count < 2) | (high_count < 2)] = 0.0
    return low, high, separation


def _trimmed_mean(values: np.ndarray, trim_fraction: float = 0.1) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if len(x) == 0:
        return 0.0
    trim = int(np.floor(float(trim_fraction) * len(x)))
    if 2 * trim >= len(x):
        return float(np.mean(x))
    return float(np.mean(x[trim:len(x) - trim]))


def grid_indices(offset: float, cadence: float, n_coefficients: int, trace_length: int) -> np.ndarray:
    """Construct and validate one common physical POI grid."""

    if cadence <= 0 or n_coefficients < 1:
        raise ValueError("cadence and n_coefficients must be positive")
    indices = np.rint(float(offset) + float(cadence) * np.arange(int(n_coefficients))).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= int(trace_length)):
        raise ValueError("candidate POI grid exceeds trace bounds")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("candidate POI grid is not strictly increasing")
    return indices


def score_grid(
    feature: np.ndarray,
    offset: float,
    cadence: float,
    n_coefficients: int = 256,
    trim_fraction: float = 0.1,
) -> tuple[float, float, np.ndarray]:
    """Return robust waveform-only separation score for one global grid."""

    x = np.asarray(feature, dtype=np.float64)
    indices = grid_indices(offset, cadence, n_coefficients, x.shape[1])
    _, _, separation = _column_two_means(x[:, indices])
    return _trimmed_mean(separation, trim_fraction), float(np.median(separation)), indices


def select_global_poi_grid(
    feature: np.ndarray,
    *,
    cadence_candidates: Iterable[float],
    offset_candidates: Iterable[float],
    n_coefficients: int = 256,
    n_stability_folds: int = 5,
    trim_fraction: float = 0.1,
) -> tuple[POIGridCandidate, list[dict[str, object]]]:
    """Select one global grid using only waveform calibration traces.

    Stability folds are contiguous blocks within the calibration set.  They
    are used only to report whether the selected physical grid is reproducible
    across waveform subsets; the attack caller should fit its per-POI GMM on
    the selected calibration traces or use the same selection inside each
    cross-fitting training fold.
    """

    x = np.asarray(feature, dtype=np.float64)
    if x.ndim != 2 or len(x) < max(20, int(n_stability_folds) * 4):
        raise ValueError("feature must contain enough waveform calibration traces")
    cadence_values = tuple(float(v) for v in cadence_candidates)
    offset_values = tuple(float(v) for v in offset_candidates)
    if not cadence_values or not offset_values:
        raise ValueError("candidate grids must be non-empty")

    def evaluate(data: np.ndarray) -> list[tuple[float, float, float, float, np.ndarray]]:
        rows = []
        for cadence in cadence_values:
            for offset in offset_values:
                try:
                    score, median, indices = score_grid(
                        data, offset, cadence, n_coefficients, trim_fraction
                    )
                except ValueError:
                    continue
                rows.append((score, median, cadence, offset, indices))
        if not rows:
            raise ValueError("no candidate POI grid fits the trace")
        return rows

    full_rows = evaluate(x)
    full_best = max(full_rows, key=lambda row: (row[0], row[1]))
    fold_scores: list[float] = []
    fold_winners: list[tuple[float, float]] = []
    fold_records: list[dict[str, object]] = []
    for fold_id, indices in enumerate(np.array_split(np.arange(len(x)), int(n_stability_folds))):
        rows = evaluate(x[indices])
        best = max(rows, key=lambda row: (row[0], row[1]))
        fold_scores.append(float(best[0]))
        fold_winners.append((float(best[2]), float(best[3])))
        fold_records.append({
            "fold_id": int(fold_id),
            "count": int(len(indices)),
            "best_cadence": float(best[2]),
            "best_offset": float(best[3]),
            "best_score": float(best[0]),
            "best_median_score": float(best[1]),
        })

    candidate = POIGridCandidate(
        cadence=float(full_best[2]),
        offset=float(full_best[3]),
        score=float(full_best[0]),
        median_score=float(full_best[1]),
        fold_scores=tuple(fold_scores),
        fold_winners=tuple(fold_winners),
        indices=np.asarray(full_best[4], dtype=np.int64),
    )
    return candidate, fold_records

