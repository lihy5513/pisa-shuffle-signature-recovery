"""Label-free count channels for the Robust PISA branch."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.special import betaln, gammaln


N = 256

DEFAULT_COUNT_SPREAD_OFFSETS = (-1, 0, 1)
DEFAULT_COUNT_SPREAD_WEIGHTS = (0.25, 0.50, 0.25)


def _log_binomial_pmf(size: int, probability: float) -> np.ndarray:
    # Match the historical channel's numerical boundary exactly.  This keeps
    # the independent baseline a genuine implementation-equivalent reference,
    # including the k=0 and k=n rows.
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    support = np.arange(size + 1, dtype=float)
    return (
        gammaln(size + 1.0)
        - gammaln(support + 1.0)
        - gammaln(size - support + 1.0)
        + support * np.log(p)
        + (size - support) * np.log1p(-p)
    )


def global_count_log_pmf(alpha: float, beta: float, n: int = N) -> np.ndarray:
    """Return log P(observed_count | true_count) for the IID baseline."""

    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    output = np.full((n + 1, n + 1), -np.inf, dtype=np.float64)
    for true_count in range(n + 1):
        first = np.exp(_log_binomial_pmf(true_count, 1.0 - float(beta)))
        second = np.exp(_log_binomial_pmf(n - true_count, float(alpha)))
        probability = np.convolve(first, second)[: n + 1]
        output[true_count] = np.log(np.maximum(probability, 1e-300))
    return output.astype(np.float32)


def _log_beta_binomial_pmf(size: int, probability: float, rho: float) -> np.ndarray:
    """Return a beta-binomial PMF; rho=0 is the ordinary binomial limit."""

    if rho <= 1e-12:
        return _log_binomial_pmf(size, probability)
    rho = float(np.clip(rho, 1e-8, 1.0 - 1e-8))
    concentration = 1.0 / rho - 1.0
    a = max(float(probability) * concentration, 1e-8)
    b = max((1.0 - float(probability)) * concentration, 1e-8)
    support = np.arange(size + 1, dtype=float)
    return (
        gammaln(size + 1.0)
        - gammaln(support + 1.0)
        - gammaln(size - support + 1.0)
        + betaln(support + a, size - support + b)
        - betaln(a, b)
    )


def overdispersed_count_log_pmf(
    alpha: float, beta: float, rho: float, n: int = N
) -> np.ndarray:
    """Count channel with beta-binomial over-dispersion in both classes."""

    output = np.full((int(n) + 1, int(n) + 1), -np.inf, dtype=np.float64)
    for true_count in range(int(n) + 1):
        detected = np.exp(_log_beta_binomial_pmf(true_count, 1.0 - beta, rho))
        false_positive = np.exp(_log_beta_binomial_pmf(int(n) - true_count, alpha, rho))
        probability = np.convolve(detected, false_positive)[: int(n) + 1]
        output[true_count] = np.log(np.maximum(probability, 1e-300))
    return output.astype(np.float32)


def contamination_count_log_pmf(
    alpha: float,
    beta: float,
    alpha_bad: float,
    beta_bad: float,
    contamination_probability: float,
    n: int = N,
) -> np.ndarray:
    """Trace-contamination mixture of a normal and a bad count channel."""

    gamma = float(np.clip(contamination_probability, 0.0, 1.0))
    normal = np.exp(global_count_log_pmf(alpha, beta, n).astype(np.float64))
    bad = np.exp(global_count_log_pmf(alpha_bad, beta_bad, n).astype(np.float64))
    return np.log(np.maximum((1.0 - gamma) * normal + gamma * bad, 1e-300)).astype(np.float32)


def poisson_binomial_log_pmf(probabilities: np.ndarray) -> np.ndarray:
    """Return ``log P(K=k)`` for a sum of independent Bernoulli variables.

    The recurrence is the coefficient update for
    ``prod_j ((1-p_j) + p_j z)``.  It costs ``O(N^2)`` arithmetic and never
    enumerates the ``2^N`` Bernoulli outcome configurations.  A small
    probability floor keeps the returned log-PMF finite for the overconfident
    float32 posteriors emitted by the GMM frontend.
    """

    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(values) < 1:
        raise ValueError("probabilities must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0,1]")
    p = np.clip(values, 1e-9, 1.0 - 1e-9)
    distribution = np.zeros(len(p) + 1, dtype=np.float64)
    distribution[0] = 1.0
    active = 0
    for value in p:
        active += 1
        distribution[1 : active + 1] = (
            distribution[1 : active + 1] * (1.0 - value)
            + distribution[:active] * value
        )
        distribution[0] *= 1.0 - value
    distribution /= max(float(np.sum(distribution)), 1e-300)
    return np.log(np.maximum(distribution, 1e-300)).astype(np.float32)


def poisson_binomial_log_pmf_batch(posterior: np.ndarray) -> np.ndarray:
    """Return one Poisson-binomial log-PMF for every posterior row."""

    values = np.asarray(posterior, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("posterior must have shape (M,N) with N >= 1")
    return np.stack([poisson_binomial_log_pmf(row) for row in values], axis=0)


def broaden_count_log_pmf(
    log_pmf: np.ndarray,
    offsets: tuple[int, ...] = DEFAULT_COUNT_SPREAD_OFFSETS,
    weights: tuple[float, ...] = DEFAULT_COUNT_SPREAD_WEIGHTS,
) -> np.ndarray:
    """Convolve count PMFs with a fixed discrete count-error kernel.

    The output is the distribution of ``K + E`` where ``K`` follows the input
    PMF and ``E`` takes values from ``offsets`` with probabilities ``weights``.
    Mass that would leave the valid count support is clamped to the nearest
    boundary, so every row remains normalized.  This is a count-level
    over-dispersion correction; it does not change the per-POI GMM posterior.
    """

    values = np.asarray(log_pmf, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("log_pmf must have shape (M,N+1) or (N+1,)")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_pmf contains NaN or positive infinity")

    offset_values = np.asarray(offsets, dtype=np.int64).reshape(-1)
    weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(offset_values) < 1 or len(offset_values) != len(weight_values):
        raise ValueError("offsets and weights must be non-empty and have equal length")
    if not np.all(np.isfinite(weight_values)) or np.any(weight_values < 0.0):
        raise ValueError("weights must be finite and non-negative")
    weight_sum = float(np.sum(weight_values))
    if weight_sum <= 0.0:
        raise ValueError("weights must have positive sum")
    weight_values /= weight_sum

    probabilities = np.exp(values)
    normalizers = np.sum(probabilities, axis=1, keepdims=True)
    if np.any(normalizers <= 0.0) or not np.all(np.isfinite(normalizers)):
        raise ValueError("log_pmf contains no finite probability mass")
    probabilities /= normalizers

    support_size = values.shape[1]
    source = np.arange(support_size, dtype=np.int64)
    broadened = np.zeros_like(probabilities)
    for offset, weight in zip(offset_values, weight_values):
        target = np.clip(source + int(offset), 0, support_size - 1)
        for row in range(len(probabilities)):
            np.add.at(broadened[row], target, float(weight) * probabilities[row])

    broadened /= np.maximum(np.sum(broadened, axis=1, keepdims=True), 1e-300)
    output = np.log(np.maximum(broadened, 1e-300)).astype(np.float32)
    return output[0] if was_vector else output


def soft_posterior_log_pmf(posterior: np.ndarray) -> np.ndarray:
    """Backward-compatible name for the per-trace Poisson-binomial PMFs."""

    return poisson_binomial_log_pmf_batch(posterior)


def _shuffle_marginal_probability_table(
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Compute P(Y=y | K=k) after uniform unknown subset marginalization.

    The dynamic program stores coefficients of a bivariate polynomial. The
    first index counts how many physical positions are selected as target
    positions, and the second index counts observed target events.
    """

    fp = np.clip(np.asarray(alpha, dtype=np.float64).reshape(-1), 1e-9, 1 - 1e-9)
    fn = np.clip(np.asarray(beta, dtype=np.float64).reshape(-1), 1e-9, 1 - 1e-9)
    if fp.shape != fn.shape or len(fp) < 1:
        raise ValueError("alpha and beta must be equal non-empty vectors")
    n = len(fp)
    dp = np.zeros((n + 1, n + 1), dtype=np.float64)
    dp[0, 0] = 1.0
    for j in range(n):
        new = np.zeros_like(dp)
        active = j + 1
        old = dp[:active, :active]
        # Position j is non-target: false-positive probability fp[j].
        new[:active, :active] += old * (1.0 - fp[j])
        new[:active, 1 : active + 1] += old * fp[j]
        # Position j is target: true-positive probability 1-fn[j].
        new[1 : active + 1, :active] += old * fn[j]
        new[1 : active + 1, 1 : active + 1] += old * (1.0 - fn[j])
        dp = new
    table = np.zeros_like(dp)
    for k in range(n + 1):
        table[k] = dp[k] / float(comb(n, k))
        total = float(np.sum(table[k]))
        if total <= 0:
            raise FloatingPointError(f"position-aware row {k} underflowed")
        table[k] /= total
    return np.log(np.maximum(table, 1e-300)).astype(np.float32)


def position_aware_shuffle_log_pmf(
    alpha_by_poi: np.ndarray,
    beta_by_poi: np.ndarray,
) -> np.ndarray:
    """Build a shuffle-marginalized position-aware count channel."""

    return _shuffle_marginal_probability_table(alpha_by_poi, beta_by_poi)


@dataclass
class CountChannel:
    """Solver-facing channel contract.

    ``kind='table'`` uses a (true_count, observed_count) table. ``kind='soft'``
    uses a per-trace posterior over true count; this is intentionally kept as
    a separate mode because it is a posterior observation channel rather than
    a hard-count generative model.
    """

    name: str
    kind: str
    table: np.ndarray | None = None
    fold_tables: np.ndarray | None = None
    log_count_posterior: np.ndarray | None = None
    fold_ids: np.ndarray | None = None
    metadata: dict[str, object] | None = None

    def validate(self, n_traces: int | None = None) -> None:
        if self.kind not in {"table", "soft"}:
            raise ValueError(f"unsupported channel kind: {self.kind}")
        if self.kind == "soft":
            if self.log_count_posterior is None:
                raise ValueError("soft channel requires log_count_posterior")
            if self.log_count_posterior.ndim != 2:
                raise ValueError("log_count_posterior must have shape (M,N+1)")
            if n_traces is not None and len(self.log_count_posterior) != n_traces:
                raise ValueError("soft channel trace count mismatch")
            return
        if self.table is None and self.fold_tables is None:
            raise ValueError("table channel requires table or fold_tables")
        if self.table is not None and self.table.shape != (N + 1, N + 1):
            raise ValueError("table must have shape (257,257)")
        if self.fold_tables is not None:
            if self.fold_tables.ndim != 3 or self.fold_tables.shape[1:] != (N + 1, N + 1):
                raise ValueError("fold_tables must have shape (F,257,257)")
            if self.fold_ids is None:
                raise ValueError("fold_tables requires fold_ids")
            if n_traces is not None and len(self.fold_ids) != n_traces:
                raise ValueError("fold_ids trace count mismatch")

    def log_likelihood_rows(
        self,
        observed_counts: np.ndarray,
        fold_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return rows indexed by predicted true count for each trace."""

        observed = np.asarray(observed_counts, dtype=np.int64).reshape(-1)
        self.validate(len(observed))
        if self.kind == "soft":
            return np.asarray(self.log_count_posterior, dtype=np.float32)
        if self.fold_tables is not None:
            ids = self.fold_ids if fold_ids is None else np.asarray(fold_ids, dtype=np.int64)
            if ids.shape != observed.shape:
                raise ValueError("fold_ids shape mismatch")
            return np.stack(
                [self.fold_tables[int(fid), :, int(y)] for fid, y in zip(ids, observed)], axis=0
            ).astype(np.float32)
        return np.asarray(self.table[:, observed].T, dtype=np.float32)

    def log_likelihood_for_counts(
        self,
        predicted_counts: np.ndarray,
        observed_counts: np.ndarray,
        fold_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate one direct likelihood value per predicted integer count.

        For a soft posterior channel this is exactly ``log P_i(K_i)``.  For
        a table channel it is the corresponding ``log P(Y_i | K_i)`` entry.
        This helper makes the integer-candidate semantics explicit while
        retaining the existing solver-facing channel interface.
        """

        predicted = np.asarray(predicted_counts, dtype=np.int64).reshape(-1)
        observed = np.asarray(observed_counts, dtype=np.int64).reshape(-1)
        if predicted.shape != observed.shape:
            raise ValueError("predicted_counts and observed_counts shape mismatch")
        if np.any(predicted < 0) or np.any(predicted > N):
            raise ValueError("predicted counts must be in [0,256]")
        rows = self.log_likelihood_rows(observed, fold_ids=fold_ids)
        return rows[np.arange(len(predicted)), predicted]


def make_global_channel(alpha: float, beta: float, n: int = N) -> CountChannel:
    return CountChannel(
        name="global_iid_alpha_beta",
        kind="table",
        table=global_count_log_pmf(alpha, beta, n),
        metadata={"alpha": float(alpha), "beta": float(beta)},
    )


def make_overdispersed_channel(alpha: float, beta: float, rho: float) -> CountChannel:
    return CountChannel(
        name="overdispersed_beta_binomial",
        kind="table",
        table=overdispersed_count_log_pmf(alpha, beta, rho),
        metadata={"alpha": float(alpha), "beta": float(beta), "rho": float(rho)},
    )


def make_contamination_channel(
    alpha: float,
    beta: float,
    alpha_bad: float,
    beta_bad: float,
    contamination_probability: float,
) -> CountChannel:
    return CountChannel(
        name="trace_contamination_mixture",
        kind="table",
        table=contamination_count_log_pmf(
            alpha, beta, alpha_bad, beta_bad, contamination_probability
        ),
        metadata={
            "alpha": float(alpha),
            "beta": float(beta),
            "alpha_bad": float(alpha_bad),
            "beta_bad": float(beta_bad),
            "contamination_probability": float(contamination_probability),
        },
    )


def estimate_unlabeled_trace_mixture(
    posterior: np.ndarray, bad_quantile: float = 0.90
) -> dict[str, float]:
    """Estimate a normal/bad trace mixture from posterior uncertainty only."""

    p = np.clip(np.asarray(posterior, dtype=float), 1e-7, 1.0 - 1e-7)
    hard = p >= 0.5
    fp_mass = np.sum((1.0 - p) * hard, axis=1)
    fn_mass = np.sum(p * (~hard), axis=1)
    non_mass = np.sum(1.0 - p, axis=1)
    target_mass = np.sum(p, axis=1)
    error_mass = (fp_mass + fn_mass) / max(p.shape[1], 1)
    threshold = float(np.quantile(error_mass, bad_quantile))
    bad = error_mass >= threshold
    if np.all(bad) or np.all(~bad):
        bad = error_mass > np.mean(error_mass)
    def rate(numerator: np.ndarray, denominator: np.ndarray, mask: np.ndarray) -> float:
        return float(np.sum(numerator[mask]) / max(float(np.sum(denominator[mask])), 1e-12))
    alpha_normal = rate(fp_mass, non_mass, ~bad)
    beta_normal = rate(fn_mass, target_mass, ~bad)
    alpha_bad = rate(fp_mass, non_mass, bad)
    beta_bad = rate(fn_mass, target_mass, bad)
    return {
        "alpha": alpha_normal,
        "beta": beta_normal,
        "alpha_bad": alpha_bad,
        "beta_bad": beta_bad,
        "contamination_probability": float(np.mean(bad)),
        "bad_quantile": float(bad_quantile),
        "trace_error_mass_mean": float(np.mean(error_mass)),
        "trace_error_mass_p95": float(np.quantile(error_mass, 0.95)),
        "labels_used": False,
    }


def make_unlabeled_contamination_channel(
    posterior: np.ndarray, bad_quantile: float = 0.90
) -> CountChannel:
    rates = estimate_unlabeled_trace_mixture(posterior, bad_quantile)
    channel = make_contamination_channel(
        rates["alpha"], rates["beta"], rates["alpha_bad"], rates["beta_bad"],
        rates["contamination_probability"],
    )
    channel.metadata = {**(channel.metadata or {}), "estimator": rates}
    return channel


def make_unlabeled_overdispersed_channel(
    posterior: np.ndarray, rho: float = 0.05
) -> CountChannel:
    p = np.clip(np.asarray(posterior, dtype=float), 1e-7, 1.0 - 1e-7)
    hard = p >= 0.5
    fp = np.sum((1.0 - p) * hard)
    fn = np.sum(p * (~hard))
    non = np.sum(1.0 - p)
    target = np.sum(p)
    alpha = float(fp / max(non, 1e-12))
    beta = float(fn / max(target, 1e-12))
    channel = make_overdispersed_channel(alpha, beta, rho)
    channel.metadata = {**(channel.metadata or {}), "labels_used": False}
    return channel


def make_posterior_count_channel(posterior: np.ndarray) -> CountChannel:
    """Build the GMM-posterior per-trace probability-count channel.

    ``posterior[i,j]`` is the parameter of an independent Bernoulli variable
    for trace ``i`` and POI ``j``.  The stored row is the resulting
    Poisson-binomial distribution over that trace's latent count.  No labels,
    secret values, or permutation metadata are consumed.
    """

    values = np.asarray(posterior, dtype=np.float32)
    log_pmf = poisson_binomial_log_pmf_batch(values)
    return CountChannel(
        name="gmm_posterior_per_trace_poisson_binomial",
        kind="soft",
        log_count_posterior=log_pmf,
        metadata={
            "posterior_shape": list(values.shape),
            "construction": "independent_bernoulli_sum",
            "distribution": "poisson_binomial",
            "dynamic_programming": True,
            "labels_used": False,
        },
    )


def make_posterior_count_spread_channel(
    posterior: np.ndarray,
    offsets: tuple[int, ...] = DEFAULT_COUNT_SPREAD_OFFSETS,
    weights: tuple[float, ...] = DEFAULT_COUNT_SPREAD_WEIGHTS,
) -> CountChannel:
    """Build the posterior count channel with fixed count-domain spreading.

    The base Poisson-binomial PMF is computed from the unchanged GMM
    posterior, then convolved with a small zero-mean count kernel.  The
    default ``(-1,0,+1)`` kernel is ``(0.25,0.50,0.25)``.
    """

    base = make_posterior_count_channel(posterior)
    spread = broaden_count_log_pmf(
        base.log_count_posterior,
        offsets=offsets,
        weights=weights,
    )
    offset_values = np.asarray(offsets, dtype=np.float64).reshape(-1)
    weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
    weight_values /= max(float(np.sum(weight_values)), 1e-300)
    kernel_mean = float(np.sum(offset_values * weight_values))
    kernel_variance = float(np.sum(weight_values * (offset_values - kernel_mean) ** 2))
    return CountChannel(
        name="gmm_posterior_per_trace_poisson_binomial_count_spread",
        kind="soft",
        log_count_posterior=spread,
        metadata={
            "posterior_shape": list(np.asarray(posterior).shape),
            "base_distribution": "poisson_binomial",
            "dynamic_programming": True,
            "count_spread": True,
            "count_spread_offsets": [int(value) for value in offset_values],
            "count_spread_weights": [float(value) for value in weight_values],
            "count_spread_mean": kernel_mean,
            "count_spread_variance": kernel_variance,
            "boundary_policy": "clamp_to_support",
            "labels_used": False,
        },
    )


def make_soft_channel(posterior: np.ndarray) -> CountChannel:
    """Backward-compatible alias for the original soft-PB channel name."""

    channel = make_posterior_count_channel(posterior)
    channel.name = "soft_posterior_poisson_binomial"
    return channel


def make_position_aware_channel(
    alpha_by_poi: np.ndarray,
    beta_by_poi: np.ndarray,
) -> CountChannel:
    return CountChannel(
        name="shuffle_marginal_position_aware_poisson_binomial",
        kind="table",
        table=position_aware_shuffle_log_pmf(alpha_by_poi, beta_by_poi),
        metadata={
            "poi_count": int(len(np.asarray(alpha_by_poi).reshape(-1))),
            "shuffle_model": "uniform_unknown_subset",
        },
    )


def make_fold_table_channel(
    name: str,
    tables: np.ndarray,
    fold_ids: np.ndarray,
    metadata: dict[str, object] | None = None,
) -> CountChannel:
    """Create a fixed cross-fitted table channel.

    Each held-out trace selects the table fitted on the other four blocks.
    The table is constructed once and can therefore be reused for every
    prefix without recalibration.
    """

    values = np.asarray(tables, dtype=np.float32)
    ids = np.asarray(fold_ids, dtype=np.int64).reshape(-1)
    if values.ndim != 3 or values.shape[1:] != (N + 1, N + 1):
        raise ValueError("tables must have shape (F,257,257)")
    if len(ids) < 1 or np.any(ids < 0) or np.any(ids >= len(values)):
        raise ValueError("fold_ids do not index tables")
    return CountChannel(
        name=name,
        kind="table",
        fold_tables=values,
        fold_ids=ids,
        metadata=metadata or {},
    )


def make_crossfit_global_channel(
    alpha_by_fold: np.ndarray,
    beta_by_fold: np.ndarray,
    fold_ids: np.ndarray,
) -> CountChannel:
    """Global IID baseline with one training-only table per fold."""

    alpha = np.asarray(alpha_by_fold, dtype=float).reshape(-1)
    beta = np.asarray(beta_by_fold, dtype=float).reshape(-1)
    tables = np.stack([global_count_log_pmf(a, b) for a, b in zip(alpha, beta)])
    return make_fold_table_channel(
        "global_iid_alpha_beta_crossfit",
        tables,
        fold_ids,
        {"alpha_by_fold": alpha.tolist(), "beta_by_fold": beta.tolist()},
    )


def make_crossfit_position_aware_channel(
    alpha_by_fold: np.ndarray,
    beta_by_fold: np.ndarray,
    fold_ids: np.ndarray,
) -> CountChannel:
    """Position-aware shuffle-marginalized tables with fixed fold fit."""

    alpha = np.asarray(alpha_by_fold, dtype=float)
    beta = np.asarray(beta_by_fold, dtype=float)
    if alpha.shape != beta.shape or alpha.ndim != 2 or alpha.shape[1] != N:
        raise ValueError("fold error profiles must have shape (F,256)")
    tables = np.stack([
        position_aware_shuffle_log_pmf(a, b) for a, b in zip(alpha, beta)
    ])
    return make_fold_table_channel(
        "shuffle_marginal_position_aware_poisson_binomial_crossfit",
        tables,
        fold_ids,
        {
            "shuffle_model": "uniform_unknown_subset",
            "alpha_by_fold_mean": alpha.mean(axis=1).tolist(),
            "beta_by_fold_mean": beta.mean(axis=1).tolist(),
        },
    )
