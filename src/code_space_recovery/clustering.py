"""Bernoulli Mixture Model (BMM) clustering for code-space recovery.

The main public function, ``assign_clusters_bmm``, takes globally unique
encoded-space bitstrings and global sample weights/probabilities, fits a
StepMix Bernoulli mixture model, and returns hard clusters directly compatible
with ``code_space_recovery.recovery.run_code_space_recovery``.

Contract
--------
Input bitstrings:
    shape = (N_unique, n_encoded)
    dtype = np.uint8
    values in {0, 1}
    n_encoded = 2 * n_logical
    n_encoded >= 2 and even
    rows globally unique

Input probabilities:
    shape = (N_unique,)
    dtype = np.float64
    finite
    nonnegative
    sum(probabilities) > 0

Output:
    clustered_samples = (
        (encoded_bitstrings_0, weights_0),
        (encoded_bitstrings_1, weights_1),
        ...,
        (encoded_bitstrings_{k-1}, weights_{k-1}),
    )

The returned weights are not normalized cluster-wise. They preserve the input
weight scale. ``run_code_space_recovery`` will normalize all clusters globally.
"""

from __future__ import annotations

import numbers
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import numpy as np

try:  # Optional at import time; required when assign_clusters_bmm is called.
    from stepmix.stepmix import StepMix  # type: ignore
except Exception:  # pragma: no cover - environment-dependent dependency
    StepMix = None  # type: ignore[assignment]

try:  # Optional; if unavailable, the code falls back to sequential n_init runs.
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover - environment-dependent dependency
    Parallel = None  # type: ignore[assignment]
    delayed = None  # type: ignore[assignment]


MODULE_VERSION = "v1.0"
__version__ = MODULE_VERSION
__clustering_name__ = "clustering_v1"

ClusteredSamples = tuple[tuple[np.ndarray, np.ndarray], ...]


# =============================================================================
# Row-key helper
# =============================================================================


def _row_keys_uint8(bitstrings: np.ndarray) -> np.ndarray:
    """View each row of a 2D uint8 matrix as a fixed-width key.

    This is used for fast row-level uniqueness checks without converting rows to
    Python strings.
    """
    bitstrings = np.ascontiguousarray(bitstrings)
    if bitstrings.ndim != 2:
        raise ValueError("bitstrings must be a 2D array.")
    row_dtype = np.dtype((np.void, bitstrings.dtype.itemsize * bitstrings.shape[1]))
    return bitstrings.view(row_dtype).reshape(-1)


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_bmm_inputs(
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    k: int,
    random_state: int | None,
    n_init: int,
) -> tuple[np.ndarray, np.ndarray, int, int | None, int]:
    """Validate the public assign_clusters_bmm inputs."""
    bits = np.asarray(bitstrings)
    probs = np.asarray(probabilities)

    if bits.ndim != 2:
        raise ValueError("bitstrings must have shape (N_unique, n_encoded).")
    if bits.shape[0] == 0:
        raise ValueError("bitstrings must contain at least one row.")
    if bits.dtype != np.uint8:
        raise TypeError(f"bitstrings must have dtype np.uint8, got {bits.dtype}.")
    if bits.shape[1] < 2 or bits.shape[1] % 2 != 0:
        raise ValueError(
            "n_encoded must be an even positive encoded length, i.e. >= 2."
        )
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bitstrings must contain only 0 or 1.")

    keys = _row_keys_uint8(bits)
    if len(np.unique(keys)) != len(keys):
        raise ValueError(
            "bitstrings must already be globally unique. "
            "Merge duplicate measurement rows before assign_clusters_bmm."
        )

    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1D array.")
    if probs.shape[0] != bits.shape[0]:
        raise ValueError(
            "probabilities must have the same length as bitstrings. "
            f"Got {probs.shape[0]} and {bits.shape[0]}."
        )
    if probs.dtype != np.float64:
        raise TypeError(
            f"probabilities must have dtype np.float64, got {probs.dtype}."
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite.")
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be nonnegative.")
    if float(np.sum(probs)) <= 0.0:
        raise ValueError("sum(probabilities) must be positive.")

    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise TypeError("k must be an integer.")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > bits.shape[0]:
        raise ValueError(
            f"k must be <= N_unique. Got k={k}, N_unique={bits.shape[0]}."
        )

    positive_weight_count = int(np.count_nonzero(probs > 0.0))
    if k > positive_weight_count:
        raise ValueError(
            "k must be <= the number of positive-weight unique bitstrings. "
            f"Got k={k}, positive_weight_count={positive_weight_count}."
        )

    if random_state is not None:
        if isinstance(random_state, bool) or not isinstance(
            random_state, numbers.Integral
        ):
            raise TypeError("random_state must be None or an integer.")
        random_state = int(random_state)
        if random_state < 0:
            raise ValueError("random_state must be nonnegative.")

    if isinstance(n_init, bool) or not isinstance(n_init, numbers.Integral):
        raise TypeError("n_init must be an integer.")
    n_init = int(n_init)
    if n_init < 1:
        raise ValueError("n_init must be >= 1.")

    return (
        np.ascontiguousarray(bits, dtype=np.uint8),
        np.ascontiguousarray(probs, dtype=np.float64),
        k,
        random_state,
        n_init,
    )


def _validate_n_jobs(n_jobs: int | None, *, n_init: int) -> int:
    """Validate and cap n_jobs by n_init."""
    if n_jobs is None:
        return min(n_init, os.cpu_count() or 1)

    if isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral):
        raise TypeError("n_jobs must be None or an integer.")
    n_jobs_eff = int(n_jobs)
    if n_jobs_eff < 1:
        raise ValueError("n_jobs must be >= 1.")
    return min(n_jobs_eff, n_init)


# =============================================================================
# StepMix helpers
# =============================================================================


def _require_stepmix() -> Any:
    if StepMix is None:
        raise ImportError(
            "assign_clusters_bmm requires the `stepmix` package. "
            "Install it with `pip install stepmix`."
        )
    return StepMix


def _make_stepmix_model(*, k: int, seed: int) -> Any:
    """Construct a StepMix Bernoulli mixture model.

    Some StepMix versions accept ``progress_bar`` and some older versions may
    not. The fallback keeps this module compatible with both.
    """
    stepmix_cls = _require_stepmix()
    kwargs: dict[str, Any] = {
        "n_components": k,
        "measurement": "binary",
        "verbose": 0,
        "random_state": seed,
        "n_init": 1,
    }

    try:
        return stepmix_cls(**kwargs, progress_bar=0)
    except TypeError as exc:
        if "progress_bar" not in str(exc):
            raise
        return stepmix_cls(**kwargs)


# =============================================================================
# Public API
# =============================================================================


def assign_clusters_bmm(
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    k: int,
    random_state: int | None = 42,
    n_init: int = 50,
    *,
    n_jobs: int | None = None,
) -> ClusteredSamples:
    """Partition globally unique encoded bitstrings into k hard clusters using BMM.

    Parameters
    ----------
    bitstrings:
        Encoded-space binary samples.

        Required contract:
            shape = (N_unique, n_encoded)
            dtype = np.uint8
            values in {0, 1}
            n_encoded >= 2 and even
            rows globally unique

        The rows may contain either valid encoded pairs, 01/10, or invalid
        encoded pairs, 00/11. Recovery is handled later by
        run_code_space_recovery.

    probabilities:
        Global sample weights or probabilities.

        Required contract:
            shape = (N_unique,)
            dtype = np.float64
            finite
            nonnegative
            sum(probabilities) > 0

        These weights are not normalized cluster-wise in the output.

    k:
        Number of Bernoulli mixture components / hard clusters.

    random_state:
        Base seed used to generate one deterministic seed per StepMix
        initialization. Use None for entropy-based randomness.

    n_init:
        Number of independent StepMix EM initializations. The model with the
        largest finite lower_bound_ is selected.

    n_jobs:
        Number of parallel jobs for the independent initializations. If None,
        uses min(n_init, os.cpu_count()). If joblib is unavailable, falls back
        to sequential execution.

    Returns
    -------
    clustered_samples:
        Tuple directly compatible with
        code_space_recovery.recovery.run_code_space_recovery:

            (
                (encoded_bitstrings_0, weights_0),
                (encoded_bitstrings_1, weights_1),
                ...,
                (encoded_bitstrings_{k-1}, weights_{k-1}),
            )

        For each cluster j:
            encoded_bitstrings_j.shape == (N_j, n_encoded)
            encoded_bitstrings_j.dtype == np.uint8
            weights_j.shape == (N_j,)
            weights_j.dtype == np.float64

        Empty clusters are represented as arrays with shapes
        (0, n_encoded) and (0,), respectively.

    Notes
    -----
    This function performs hard clustering. Each unique encoded row is assigned
    to exactly one cluster using ``best_model.predict(bitstrings)``. It does not
    split one row's probability mass across multiple clusters.
    """
    bits, probs, k, random_state, n_init = _validate_bmm_inputs(
        bitstrings,
        probabilities,
        k,
        random_state,
        n_init,
    )
    _require_stepmix()

    n_unique, n_encoded = bits.shape
    n_jobs_eff = _validate_n_jobs(n_jobs, n_init=n_init)

    # StepMix's likelihood is invariant to a global sample-weight scale.
    # Normalize for numerical stability, but preserve the original probs scale
    # in the returned clustered weights.
    fit_weights = probs / float(np.sum(probs))

    def fit_single_stepmix(seed: int) -> tuple[Any | None, float, str | None]:
        """Fit one StepMix model with exactly one EM initialization."""
        try:
            with open(os.devnull, "w") as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    model = _make_stepmix_model(k=k, seed=seed)
                    model.fit(bits, sample_weight=fit_weights)
            lower_bound = float(getattr(model, "lower_bound_", -np.inf))
            return model, lower_bound, None
        except Exception as exc:  # pragma: no cover - depends on StepMix internals
            return None, -np.inf, f"seed={seed}: {type(exc).__name__}: {exc}"

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int32).max,
        size=n_init,
        dtype=np.int64,
    )

    if Parallel is None or delayed is None or n_jobs_eff == 1:
        fitted = [fit_single_stepmix(int(seed)) for seed in seeds]
    else:
        fitted = Parallel(n_jobs=n_jobs_eff)(
            delayed(fit_single_stepmix)(int(seed)) for seed in seeds
        )

    successful = [
        (model, lb)
        for model, lb, _err in fitted
        if model is not None and np.isfinite(lb)
    ]
    if len(successful) == 0:
        errors = [err for _model, _lb, err in fitted if err is not None]
        error_preview = "; ".join(errors[:5]) if errors else "no finite lower_bound_"
        raise RuntimeError(
            "All StepMix initializations failed or returned non-finite lower_bound_. "
            f"First failures: {error_preview}"
        )

    best_model, _best_lower_bound = max(successful, key=lambda item: item[1])

    labels = np.asarray(best_model.predict(bits), dtype=np.int64)

    if labels.shape != (n_unique,):
        raise RuntimeError(
            f"StepMix returned labels with shape {labels.shape}, expected {(n_unique,)}."
        )
    if np.any((labels < 0) | (labels >= k)):
        raise RuntimeError("StepMix returned labels outside [0, k).")

    clustered: list[tuple[np.ndarray, np.ndarray]] = []

    for cluster_id in range(k):
        mask = labels == cluster_id

        bits_j = np.ascontiguousarray(bits[mask], dtype=np.uint8)
        probs_j = np.ascontiguousarray(probs[mask], dtype=np.float64)

        # Empty-cluster shape safety: bits_j must be (0, n_encoded), not (0,).
        bits_j = bits_j.reshape((-1, n_encoded))
        probs_j = probs_j.reshape((-1,))

        clustered.append((bits_j, probs_j))

    # Safety checks: hard partition, no duplicated rows, total weight preserved.
    all_bits = np.vstack([cluster[0] for cluster in clustered])
    all_probs = np.concatenate([cluster[1] for cluster in clustered])

    if all_bits.shape != bits.shape:
        raise RuntimeError("Internal error: clustered bitstring count changed.")
    if all_probs.shape != probs.shape:
        raise RuntimeError("Internal error: clustered probability count changed.")

    if len(np.unique(_row_keys_uint8(all_bits))) != len(all_bits):
        raise RuntimeError("Internal error: clustering output contains duplicate rows.")

    if not np.isclose(float(np.sum(all_probs)), float(np.sum(probs))):
        raise RuntimeError("Internal error: clustering output weight sum changed.")

    return tuple(clustered)


def cluster_weight_sums(clustered_samples: ClusteredSamples) -> np.ndarray:
    """Return total weight mass per cluster.

    This helper is optional and is not needed by run_code_space_recovery, but it is useful
    for logging/debugging BMM outputs.
    """
    return np.array([float(np.sum(weights)) for _bits, weights in clustered_samples], dtype=np.float64)


__all__ = [
    "ClusteredSamples",
    "assign_clusters_bmm",
    "cluster_weight_sums",
]
