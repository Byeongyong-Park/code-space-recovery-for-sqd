"""Self-consistent code-space recovery loop for pair-code SQD samples.

The pair code is written in displayed pair order:
logical bit 0 -> encoded pair 01, and logical bit 1 -> encoded pair 10.
All bitstring arrays are 2D ``np.uint8`` arrays, not strings.

This module validates clustered encoded samples, repairs invalid 00/11 pairs,
builds logical batches, and updates cluster reference vectors from the
global-best Ritz state. Projection and diagonalization are delegated to a
logical-space diagonalizer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Sequence
import hashlib
import math
import numbers
import os
import time

import numba as nb
import numpy as np

try:  # package import
    from ._version import ALGORITHM_VERSION, PACKAGE_VERSION
except ImportError:  # pragma: no cover - flat-module compatibility
    from _version import ALGORITHM_VERSION, PACKAGE_VERSION  # type: ignore


MODULE_VERSION = "v1.0"
# Conventional module version follows the distribution; MODULE_VERSION remains
# the legacy core/output-schema identifier.
__version__ = PACKAGE_VERSION
_CORE_NAME = "code_space_recovery_core_v1_0"


# =============================================================================
# Result containers
# =============================================================================


@dataclass
class DiagonalizationResult:
    """Lowest projected Ritz pair returned by a user-supplied diagonalizer.

    ``coefficients[i]`` corresponds to ``logical_basis[i]``. The returned basis
    must be a duplicate-free subset of the batch supplied to the diagonalizer.
    """

    energy: float
    logical_basis: np.ndarray
    coefficients: np.ndarray


@dataclass
class CodeSpaceRecoveryResult:
    """Final output and histories produced by ``run_code_space_recovery``.

    ``best_*`` fields describe the global best across all batches and
    iterations. Basis and coefficient arrays preserve row-wise correspondence.
    Histories are recorded once per iteration; batch histories are nested as
    ``[iteration][batch]``. ``batch_dim_history`` contains the basis size
    returned by each diagonalizer, which may be below ``max_dim``.

    ``reference_history`` stores each post-iteration reference update.
    ``final_stage_iteration_history`` is ``None`` before the final schedule
    stage. ``converged`` is true only when the final adaptive stage stops by its
    no-improvement patience criterion.
    """

    best_energy: float
    best_logical_basis: np.ndarray
    best_coefficients: np.ndarray
    final_reference_vectors: np.ndarray
    reference_history: list[np.ndarray]
    iteration_best_energy_history: list[float]
    global_best_energy_history: list[float]
    global_best_updated_history: list[bool]
    batch_energy_history: list[list[float]]
    batch_dim_history: list[list[int]]
    batch_diag_seconds_history: list[list[float]]
    batch_worker_metadata_history: list[list[dict[str, Any]]]
    recovery_metadata_history: list[dict[str, Any]]
    recovery_method_history: list[str]
    recovery_stage_history: list[str]
    recovery_stage_iteration_history: list[int]
    final_stage_iteration_history: list[int | None]
    converged: bool
    config: dict[str, Any]
    iteration_timing_history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Recovered encoded samples, aligned nonnegative weights, and metadata.

    Rows must satisfy the pair code. Duplicate rows are allowed; the driver
    merges them unless the recovery callable advertises unique output.
    """

    bitstrings: np.ndarray
    weights: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _RecoveryScheduleStage:
    """One normalized recovery stage used by run_code_space_recovery."""

    name: str
    recovery_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.random.Generator],
        RecoveryResult,
    ]
    iterations: int | None
    min_iterations: int | None
    max_iterations: int | None
    convergence_patience: int | None
    method: str
    config: dict[str, Any] | None


@dataclass
class SparseInvalidPairs:
    """Sparse representation of encoded pairs that need repair.

    Valid 01/10 pairs are omitted; only invalid 00/11 pairs are represented.
    """

    sample_indices: np.ndarray
    pair_indices: np.ndarray
    cluster_indices: np.ndarray
    invalid_pair_bits: np.ndarray
    invalid_pair_reference: np.ndarray


# =============================================================================
# Validation helpers
# =============================================================================


def _as_exact_binary_uint8(
    name: str,
    value: Any,
    *,
    ndim: int = 2,
) -> np.ndarray:
    """Validate exact binary values before converting them to ``uint8``.

    Checking the original array is essential: converting first would silently
    truncate fractional values and wrap large integers modulo 256. Numeric
    dtypes (including Boolean and complex values with zero imaginary part) are
    accepted when every element is finite and exactly equal to zero or one.
    """
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array. Got shape {array.shape}.")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(
            f"{name} must have a numeric binary dtype. Got {array.dtype}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain exactly 0 or 1.")

    # A complex value can pass the exact comparison only when its imaginary
    # part is zero. Taking the real component avoids ComplexWarning on cast.
    if np.issubdtype(array.dtype, np.complexfloating):
        array = array.real
    return np.asarray(array, dtype=np.uint8)


def _as_finite_real_array(
    name: str,
    value: Any,
    *,
    ndim: int | None = None,
) -> np.ndarray:
    """Return a finite real ``float64`` array without coercing bad dtypes."""
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array. Got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise TypeError(f"{name} must have a real numeric dtype. Got {array.dtype}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    with np.errstate(over="ignore", invalid="ignore"):
        out = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} cannot be represented as finite float64 values.")
    return out


def _as_finite_nonnegative_weights(
    name: str,
    value: Any,
) -> np.ndarray:
    """Validate a one-dimensional finite nonnegative weight array."""
    weights = _as_finite_real_array(name, value, ndim=1)
    if np.any(weights < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return weights


def _as_reference_vectors(
    name: str,
    value: Any,
    *,
    require_normalized: bool,
) -> np.ndarray:
    """Validate nonnegative pair-reference vectors.

    Consumers require every pair to be a probability pair whose entries sum
    to one. The normalizer accepts arbitrary finite nonnegative magnitudes and
    maps a zero pair to the neutral reference ``[0.5, 0.5]``.
    """
    reference_vectors = _as_finite_real_array(name, value, ndim=2)
    if reference_vectors.shape[1] % 2 != 0:
        raise ValueError(f"{name} must have an even number of columns.")
    if np.any(reference_vectors < 0.0):
        raise ValueError(f"{name} must be nonnegative.")

    pairs = reference_vectors.reshape(reference_vectors.shape[0], -1, 2)
    pair_sums = pairs.sum(axis=2)
    if not np.all(np.isfinite(pair_sums)):
        raise ValueError(f"{name} pair sums must be finite.")
    if require_normalized:
        if np.any(reference_vectors > 1.0):
            raise ValueError(f"{name} entries must be in [0, 1].")
        if not np.all(np.isclose(pair_sums, 1.0, rtol=1e-12, atol=1e-12)):
            raise ValueError(f"Each pair in {name} must sum to 1.")
    return reference_vectors


def _as_cluster_labels(
    labels: Any,
    *,
    num_samples: int,
    n_clusters: int,
) -> np.ndarray:
    """Validate cluster labels before using them for array indexing."""
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError(
            f"labels must be a 1D array. Got shape {labels_array.shape}."
        )
    if labels_array.shape[0] != num_samples:
        raise ValueError(
            "labels must have one entry per encoded sample. "
            f"Got {labels_array.shape[0]} labels for {num_samples} samples."
        )
    if np.issubdtype(labels_array.dtype, np.bool_) or not np.issubdtype(
        labels_array.dtype,
        np.integer,
    ):
        raise TypeError(
            f"labels must have an integer dtype. Got {labels_array.dtype}."
        )
    if np.any(labels_array < 0) or np.any(labels_array >= n_clusters):
        raise ValueError(
            f"labels must satisfy 0 <= label < {n_clusters}."
        )
    return labels_array.astype(np.int64, copy=False)


def _validate_positive_int(name: str, value: Any) -> int:
    """Validate a positive integer."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a positive integer, got {type(value).__name__}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _validate_positive_int_or_none(name: str, value: Any) -> int | None:
    """Validate an optional positive integer."""
    if value is None:
        return None
    return _validate_positive_int(name, value)


def _validate_nonnegative_int(name: str, value: Any) -> int:
    """Validate a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"{name} must be a non-negative integer, got {type(value).__name__}."
        )
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return value


def _validate_positive_real(name: str, value: Any) -> float:
    """Validate a positive finite real number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}.")
    return value


def _validate_unit_interval_real(name: str, value: Any) -> float:
    """Validate a finite real number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number in [0, 1], got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}.")
    return value



def _validate_open_unit_interval_real(name: str, value: Any) -> float:
    """Validate a finite real number in (0, 1)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number in (0, 1), got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1), got {value}.")
    return value


def _validate_bool(name: str, value: Any) -> bool:
    """Validate a Boolean flag."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")
    return bool(value)


def _callable_metadata(fn: Callable[..., Any]) -> dict[str, Any] | None:
    """Return optional metadata attached to a recovery callable."""
    for attr in (
        "_code_space_recovery_metadata",
        "_code_space_recovery_config",
    ):
        metadata = getattr(fn, attr, None)
        if isinstance(metadata, dict):
            return dict(metadata)
    return None


def _recovery_method_name(
    recovery_fn: Callable[..., Any],
    metadata: dict[str, Any] | None,
) -> str:
    if isinstance(metadata, dict):
        method = metadata.get("method")
        if method is not None:
            return str(method)
        family = metadata.get("family")
        if family is not None:
            return str(family)
    return getattr(recovery_fn, "__name__", recovery_fn.__class__.__name__)


def _default_recovery_stage_name(
    stage_index: int,
    recovery_fn: Callable[..., Any],
    metadata: dict[str, Any] | None,
) -> str:
    method = _recovery_method_name(recovery_fn, metadata)
    return f"stage_{stage_index + 1}_{method}"


_RECOVERY_SCHEDULE_STAGE_KEYS = frozenset(
    {
        "recovery_fn",
        "name",
        "iterations",
        "min_iterations",
        "max_iterations",
        "convergence_patience",
    }
)


def _normalize_recovery_schedule_entry(
    raw_stage: Mapping[str, Any],
    *,
    stage_index: int,
    default_min_iterations: int,
    default_max_iterations: int,
    default_convergence_patience: int,
) -> _RecoveryScheduleStage:
    unknown_keys = set(raw_stage).difference(_RECOVERY_SCHEDULE_STAGE_KEYS)
    if unknown_keys:
        rendered_unknown = ", ".join(
            sorted((repr(key) for key in unknown_keys))
        )
        rendered_allowed = ", ".join(
            repr(key) for key in sorted(_RECOVERY_SCHEDULE_STAGE_KEYS)
        )
        raise ValueError(
            f"recovery_schedule[{stage_index}] contains unsupported key(s): "
            f"{rendered_unknown}. Allowed keys are: {rendered_allowed}."
        )
    if "recovery_fn" not in raw_stage:
        raise ValueError(
            f"recovery_schedule[{stage_index}] must contain a 'recovery_fn'."
        )
    stage_recovery_fn = raw_stage["recovery_fn"]
    if not callable(stage_recovery_fn):
        raise TypeError(
            f"recovery_schedule[{stage_index}]['recovery_fn'] must be callable."
        )

    adaptive_keys = ("min_iterations", "max_iterations", "convergence_patience")
    raw_iterations = raw_stage.get("iterations", None)
    has_adaptive_keys = any(key in raw_stage for key in adaptive_keys)
    if raw_iterations is not None:
        if has_adaptive_keys:
            raise ValueError(
                f"recovery_schedule[{stage_index}] cannot mix fixed 'iterations' "
                "with adaptive min_iterations/max_iterations/convergence_patience."
            )
        iterations = _validate_positive_int(
            f"recovery_schedule[{stage_index}]['iterations']",
            raw_iterations,
        )
        stage_min_iterations = None
        stage_max_iterations = None
        stage_convergence_patience = None
    else:
        iterations = None
        stage_min_iterations = _validate_nonnegative_int(
            f"recovery_schedule[{stage_index}]['min_iterations']",
            raw_stage.get("min_iterations", default_min_iterations),
        )
        stage_max_iterations = _validate_positive_int(
            f"recovery_schedule[{stage_index}]['max_iterations']",
            raw_stage.get("max_iterations", default_max_iterations),
        )
        stage_convergence_patience = _validate_positive_int(
            f"recovery_schedule[{stage_index}]['convergence_patience']",
            raw_stage.get("convergence_patience", default_convergence_patience),
        )
        if stage_min_iterations > stage_max_iterations:
            raise ValueError(
                f"recovery_schedule[{stage_index}] has min_iterations > "
                f"max_iterations: {stage_min_iterations} > {stage_max_iterations}."
            )

    metadata = _callable_metadata(stage_recovery_fn)
    raw_name = raw_stage.get("name", None)
    if raw_name is None:
        name = _default_recovery_stage_name(stage_index, stage_recovery_fn, metadata)
    else:
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"recovery_schedule[{stage_index}]['name'] is empty.")

    return _RecoveryScheduleStage(
        name=name,
        recovery_fn=stage_recovery_fn,
        iterations=iterations,
        min_iterations=stage_min_iterations,
        max_iterations=stage_max_iterations,
        convergence_patience=stage_convergence_patience,
        method=_recovery_method_name(stage_recovery_fn, metadata),
        config=metadata,
    )


def _resolve_recovery_stages(
    *,
    recovery_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.random.Generator],
        RecoveryResult,
    ] | None,
    recovery_prob_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    recovery_schedule: Sequence[Mapping[str, Any]] | None,
    default_min_iterations: int,
    default_max_iterations: int,
    default_convergence_patience: int,
) -> tuple[list[_RecoveryScheduleStage], bool]:
    if recovery_schedule is not None:
        if recovery_fn is not None or recovery_prob_fn is not None:
            raise ValueError(
                "Pass either recovery_schedule or recovery_fn/recovery_prob_fn, not both."
            )
        if isinstance(recovery_schedule, (str, bytes)):
            raise TypeError("recovery_schedule must be a sequence of mapping entries.")
        stages_raw = list(recovery_schedule)
        if len(stages_raw) == 0:
            raise ValueError("recovery_schedule must contain at least one stage.")

        stages: list[_RecoveryScheduleStage] = []
        for stage_index, raw_stage in enumerate(stages_raw):
            if not isinstance(raw_stage, Mapping):
                raise TypeError(
                    f"recovery_schedule[{stage_index}] must be a mapping/dict."
                )
            stages.append(
                _normalize_recovery_schedule_entry(
                    raw_stage,
                    stage_index=stage_index,
                    default_min_iterations=default_min_iterations,
                    default_max_iterations=default_max_iterations,
                    default_convergence_patience=default_convergence_patience,
                )
            )
        return stages, True

    if recovery_fn is not None and recovery_prob_fn is not None:
        raise ValueError("Pass either recovery_fn or recovery_prob_fn, not both.")
    if recovery_fn is None:
        if recovery_prob_fn is None:
            raise TypeError(
                "Either recovery_fn, recovery_prob_fn, or recovery_schedule must be provided."
            )
        if not callable(recovery_prob_fn):
            raise TypeError("recovery_prob_fn must be callable.")
        recovery_fn = make_mrelu_recovery_fn(recovery_prob_fn=recovery_prob_fn)
    elif not callable(recovery_fn):
        raise TypeError("recovery_fn must be callable.")

    metadata = _callable_metadata(recovery_fn)
    stage = _RecoveryScheduleStage(
        name=_default_recovery_stage_name(0, recovery_fn, metadata),
        recovery_fn=recovery_fn,
        iterations=None,
        min_iterations=default_min_iterations,
        max_iterations=default_max_iterations,
        convergence_patience=default_convergence_patience,
        method=_recovery_method_name(recovery_fn, metadata),
        config=metadata,
    )
    return [stage], False


def _recovery_stage_is_adaptive(stage: _RecoveryScheduleStage) -> bool:
    return stage.iterations is None


def _recovery_stage_max_iterations(stage: _RecoveryScheduleStage) -> int:
    if stage.iterations is not None:
        return int(stage.iterations)
    assert stage.max_iterations is not None
    return int(stage.max_iterations)


def _recovery_stage_to_config(stage: _RecoveryScheduleStage) -> dict[str, Any]:
    return {
        "name": stage.name,
        "mode": "adaptive" if _recovery_stage_is_adaptive(stage) else "fixed",
        "iterations": stage.iterations,
        "min_iterations": stage.min_iterations,
        "max_iterations": stage.max_iterations,
        "convergence_patience": stage.convergence_patience,
        "method": stage.method,
        "recovery_fn": getattr(stage.recovery_fn, "__name__", repr(stage.recovery_fn)),
        "recovery_fn_config": None if stage.config is None else dict(stage.config),
    }


def _validate_seed(seed: Any) -> int | None:
    """Validate an optional non-negative RNG seed."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be None or a non-negative integer, got {type(seed).__name__}.")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")
    return seed


# =============================================================================
# Row-key helpers
# =============================================================================


@nb.njit(cache=True)
def _pack_binary_rows_uint64_serial(bitstrings: np.ndarray) -> np.ndarray:
    """Pack binary rows in display order so numeric and lexicographic order agree."""
    n_rows, n_bits = bitstrings.shape
    keys = np.empty(n_rows, dtype=np.uint64)
    for row_index in range(n_rows):
        key = np.uint64(0)
        for bit_index in range(n_bits):
            key = (key << np.uint64(1)) | np.uint64(
                bitstrings[row_index, bit_index]
            )
        keys[row_index] = key
    return keys


@nb.njit(cache=True, parallel=True)
def _pack_binary_rows_uint64_parallel(bitstrings: np.ndarray) -> np.ndarray:
    """Parallel counterpart used for large logical-basis row sets."""
    n_rows, n_bits = bitstrings.shape
    keys = np.empty(n_rows, dtype=np.uint64)
    for row_index in nb.prange(n_rows):
        key = np.uint64(0)
        for bit_index in range(n_bits):
            key = (key << np.uint64(1)) | np.uint64(
                bitstrings[row_index, bit_index]
            )
        keys[row_index] = key
    return keys


def _row_keys(bitstrings: np.ndarray) -> np.ndarray:
    """Return sortable exact row keys for binary bitstrings.

    Logical rows up to 64 bits are packed into one big-endian ``uint64``.  This
    is substantially cheaper for ``np.unique``/``np.isin`` than a fixed-width
    void scalar, while preserving lexicographic order. Wider
    encoded rows (72 columns for the 36-qubit pair code) retain the byte-exact
    fixed-width fallback.
    """
    bitstrings = np.ascontiguousarray(bitstrings)
    if bitstrings.ndim != 2:
        raise ValueError("bitstrings must be a 2D array.")
    if bitstrings.dtype == np.uint8 and bitstrings.shape[1] <= 64:
        if len(bitstrings) >= 4_096:
            return _pack_binary_rows_uint64_parallel(bitstrings)
        return _pack_binary_rows_uint64_serial(bitstrings)
    row_dtype = np.dtype((np.void, bitstrings.dtype.itemsize * bitstrings.shape[1]))
    return bitstrings.view(row_dtype).reshape(-1)


def _unique_rows_preserve_order(bitstrings: np.ndarray) -> np.ndarray:
    """Drop duplicate rows while preserving first-occurrence order."""
    bitstrings = np.asarray(bitstrings)
    if bitstrings.ndim != 2:
        raise ValueError("bitstrings must be a 2D array.")
    if len(bitstrings) == 0:
        return bitstrings.copy()

    keys = _row_keys(bitstrings)
    _, first_indices = np.unique(keys, return_index=True)
    first_indices = np.sort(first_indices)
    return bitstrings[first_indices].copy()


def _merge_duplicate_rows_sum_weights(
    bitstrings: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate bitstring rows and sum their weights."""
    bitstrings = _as_exact_binary_uint8("bitstrings", bitstrings)
    weights = _as_finite_nonnegative_weights("weights", weights)

    if weights.ndim != 1 or weights.shape[0] != bitstrings.shape[0]:
        raise ValueError("weights must be 1D and have the same length as bitstrings.")

    if len(bitstrings) == 0:
        return bitstrings.copy(), weights.copy()

    bitstrings_c = np.ascontiguousarray(bitstrings)
    keys = _row_keys(bitstrings_c)

    _, first_indices, inverse = np.unique(
        keys,
        return_index=True,
        return_inverse=True,
    )

    unique_weights = np.zeros(len(first_indices), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        np.add.at(unique_weights, inverse, weights)
    if not np.all(np.isfinite(unique_weights)):
        raise ValueError(
            "merged weights must remain finite; duplicate-row accumulation overflowed."
        )
    unique_bitstrings = bitstrings_c[first_indices]

    return unique_bitstrings.astype(np.uint8, copy=True), unique_weights


def _filter_rows_not_in_set(
    candidates: np.ndarray,
    forbidden: np.ndarray,
) -> np.ndarray:
    """Return a mask selecting candidate rows absent from ``forbidden``."""
    if len(candidates) == 0:
        return np.zeros(0, dtype=bool)
    if forbidden is None or len(forbidden) == 0:
        return np.ones(len(candidates), dtype=bool)

    cand_keys = _row_keys(candidates)
    forb_keys = _row_keys(forbidden)
    return ~np.isin(cand_keys, forb_keys, assume_unique=False)


# =============================================================================
# Logical/encoded bitstring conversion
# =============================================================================


def encode_logical_to_valid_encoded(logical_basis: np.ndarray) -> np.ndarray:
    """Encode logical bitstrings into valid pair-code bitstrings.

    Convention:
        logical 0 -> 01
        logical 1 -> 10

    Input
    -----
    logical_basis:
        shape = (num_basis, n_logical)
        dtype = np.uint8
        values in {0, 1}

    Output
    ------
    encoded_basis:
        shape = (num_basis, 2 * n_logical)
        dtype = np.uint8

    Example
    -------
    logical 001 -> encoded 01 01 10

    >>> logical = np.array([[0, 0, 1]], dtype=np.uint8)
    >>> encode_logical_to_valid_encoded(logical)
    array([[0, 1, 0, 1, 1, 0]], dtype=uint8)
    """
    logical_basis = _as_exact_binary_uint8("logical_basis", logical_basis)

    num_basis, n_logical = logical_basis.shape
    encoded = np.empty((num_basis, 2 * n_logical), dtype=np.uint8)
    encoded[:, 0::2] = logical_basis
    encoded[:, 1::2] = 1 - logical_basis
    return encoded


def is_valid_encoded_bitstrings(encoded_bitstrings: np.ndarray) -> np.ndarray:
    """Return whether each encoded bitstring satisfies the pair code.

    Valid pairs are 01 and 10. Invalid pairs are 00 and 11.

    Input
    -----
    encoded_bitstrings:
        shape = (num_rows, n_encoded)
        dtype = np.uint8

    Output
    ------
    valid_mask:
        shape = (num_rows,)
        dtype = bool
        ``valid_mask[i]`` is true when row ``i`` is valid in every pair.
    """
    encoded_bitstrings = _as_exact_binary_uint8(
        "encoded_bitstrings",
        encoded_bitstrings,
    )
    if encoded_bitstrings.shape[1] % 2 != 0:
        raise ValueError("encoded_bitstrings must have even number of columns.")

    pairs = encoded_bitstrings.reshape(encoded_bitstrings.shape[0], -1, 2)
    pair_sums = pairs.sum(axis=2)
    return np.all(pair_sums == 1, axis=1)


def decode_valid_encoded_to_logical(
    valid_encoded_bitstrings: np.ndarray,
    *,
    check_valid: bool = True,
) -> np.ndarray:
    """Decode valid pair-code bitstrings to logical bitstrings.

    Convention:
        01 -> 0
        10 -> 1

    The first rail of each valid pair is the logical bit.

    Input
    -----
    valid_encoded_bitstrings:
        shape = (num_rows, n_encoded)
        dtype = np.uint8
        all pairs must be valid when ``check_valid=True``.

    check_valid:
        If false, skip pair-code validation and return each pair's first rail.
        The caller is then responsible for providing valid encoded rows.

    Output
    ------
    logical_bitstrings:
        shape = (num_rows, n_logical)
        dtype = np.uint8

    Example
    -------
    encoded 01 01 10 -> logical 001
    """
    valid_encoded_bitstrings = _as_exact_binary_uint8(
        "valid_encoded_bitstrings",
        valid_encoded_bitstrings,
    )
    if valid_encoded_bitstrings.shape[1] % 2 != 0:
        raise ValueError("encoded bitstring length must be even.")
    check_valid = _validate_bool("check_valid", check_valid)
    if check_valid and not np.all(is_valid_encoded_bitstrings(valid_encoded_bitstrings)):
        raise ValueError("All encoded bitstrings must be valid before decoding.")

    return valid_encoded_bitstrings[:, 0::2].astype(np.uint8, copy=True)


# =============================================================================
# Input validation and flattening
# =============================================================================


def _validate_and_flatten_clustered_samples(
    clustered_samples: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_logical: int,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], np.ndarray, np.ndarray, np.ndarray]:
    """Validate clustered samples and build flattened arrays.

    Input weights may be raw counts or global weights. They are normalized once
    over the full sample pool, not separately within each cluster. Encoded rows
    must form a globally unique partition across all clusters.
    """
    n_encoded = 2 * n_logical

    if not isinstance(clustered_samples, (tuple, list)) or len(clustered_samples) == 0:
        raise ValueError("clustered_samples must be a non-empty tuple/list of clusters.")

    normalized_clusters: list[tuple[np.ndarray, np.ndarray]] = []
    all_bits_list: list[np.ndarray] = []
    all_weights_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    total_weight = 0.0

    for k, cluster in enumerate(clustered_samples):
        if not isinstance(cluster, (tuple, list)) or len(cluster) != 2:
            raise TypeError(
                "Each cluster must be a tuple (encoded_bitstrings_k, weights_k)."
            )

        encoded_bitstrings_k, weights_k = cluster
        encoded_bitstrings_k = np.asarray(encoded_bitstrings_k)
        weights_k = _as_finite_nonnegative_weights(
            f"weights for cluster {k}",
            weights_k,
        )

        if encoded_bitstrings_k.ndim != 2:
            raise ValueError(f"encoded_bitstrings for cluster {k} must be a 2D array.")
        if encoded_bitstrings_k.dtype != np.uint8:
            raise TypeError(
                f"encoded_bitstrings for cluster {k} must have dtype np.uint8. "
                f"Got {encoded_bitstrings_k.dtype}."
            )
        if encoded_bitstrings_k.shape[1] != n_encoded:
            raise ValueError(
                f"encoded_bitstrings for cluster {k} must have shape "
                f"(n_unique_k, {n_encoded}). Got {encoded_bitstrings_k.shape}."
            )
        if not np.all((encoded_bitstrings_k == 0) | (encoded_bitstrings_k == 1)):
            raise ValueError(f"encoded_bitstrings for cluster {k} must contain only 0 or 1.")

        if weights_k.shape[0] != encoded_bitstrings_k.shape[0]:
            raise ValueError(
                f"weights for cluster {k} must have length matching bitstrings. "
                f"Got {weights_k.shape[0]} and {encoded_bitstrings_k.shape[0]}."
            )
        cluster_weight = float(np.sum(weights_k))
        total_weight += cluster_weight

        normalized_clusters.append((encoded_bitstrings_k, weights_k))
        all_bits_list.append(encoded_bitstrings_k)
        all_weights_list.append(weights_k)
        labels_list.append(np.full(encoded_bitstrings_k.shape[0], k, dtype=np.int64))

    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(
            "The total weight over all clusters must be finite and positive. "
            f"Got total_weight={total_weight}."
        )

    # Use one global probability scale across all clusters.
    inv_total_weight = 1.0 / total_weight
    normalized_clusters = tuple(
        (bits, weights * inv_total_weight)
        for bits, weights in normalized_clusters
    )

    all_bits = np.vstack(all_bits_list).astype(np.uint8, copy=False)
    all_weights = (
        np.concatenate(all_weights_list).astype(np.float64, copy=False)
        * inv_total_weight
    )
    initial_labels = np.concatenate(labels_list).astype(np.int64, copy=False)

    # Raw measurement duplicates must be merged before clustering.
    all_keys = _row_keys(all_bits)
    n_unique_rows = len(np.unique(all_keys))
    if n_unique_rows != len(all_keys):
        n_duplicates = len(all_keys) - n_unique_rows
        raise ValueError(
            "clustered_samples must be a partition of globally unique encoded bitstrings. "
            "Duplicate encoded bitstrings were found across or within clusters. "
            f"Found {n_duplicates} duplicate row occurrence(s). "
            "Merge raw measurement duplicates globally before clustering, and assign each "
            "unique encoded bitstring to exactly one cluster."
        )

    return normalized_clusters, all_bits, all_weights, initial_labels


def _validate_deterministic_logical_basis(
    deterministic_logical_basis: np.ndarray | None,
    *,
    n_logical: int,
    max_dim: int,
) -> np.ndarray:
    """Validate logical basis states that must appear in every batch.

    The input is a logical basis, not an encoded basis. ``None`` is converted to
    an empty array with shape ``(0, n_logical)``.
    """
    if deterministic_logical_basis is None:
        return np.empty((0, n_logical), dtype=np.uint8)

    det = np.asarray(deterministic_logical_basis)
    if det.ndim != 2:
        raise ValueError("deterministic_logical_basis must be a 2D array.")
    if det.dtype != np.uint8:
        raise TypeError(
            "deterministic_logical_basis must have dtype np.uint8. "
            f"Got {det.dtype}."
        )
    if det.shape[1] != n_logical:
        raise ValueError(
            "deterministic_logical_basis must have shape "
            f"(n_deterministic, {n_logical}). Got {det.shape}."
        )
    if not np.all((det == 0) | (det == 1)):
        raise ValueError("deterministic_logical_basis must contain only 0 or 1.")

    det = _unique_rows_preserve_order(det)
    if len(det) > max_dim:
        raise ValueError(
            "The number of deterministic logical basis states cannot exceed max_dim. "
            f"Got {len(det)} deterministic states and max_dim={max_dim}."
        )
    return det.astype(np.uint8, copy=False)


# =============================================================================
# Reference-vector initialization and reassignment
# =============================================================================


def pairwise_normalize_reference_vectors(reference_vectors: np.ndarray) -> np.ndarray:
    """Normalize each encoded pair in every cluster reference vector."""
    reference_vectors = _as_reference_vectors(
        "reference_vectors",
        reference_vectors,
        require_normalized=False,
    )

    n_clusters, n_encoded = reference_vectors.shape
    n_logical = n_encoded // 2

    out = reference_vectors.copy()
    pairs = out.reshape(n_clusters, n_logical, 2)
    denom = pairs.sum(axis=2, keepdims=True)

    # Normalize nonzero pairs in place.
    np.divide(pairs, denom, out=pairs, where=denom > 0)

    # Use a neutral reference for empty pairs.
    zero_mask = (denom[..., 0] <= 0)
    if np.any(zero_mask):
        pairs[zero_mask, 0] = 0.5
        pairs[zero_mask, 1] = 0.5

    return out


def initialize_reference_vectors(
    clustered_samples: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_logical: int,
) -> np.ndarray:
    """Build initial cluster references from weighted encoded samples."""
    n_logical = _validate_positive_int("n_logical", n_logical)
    if isinstance(clustered_samples, (str, bytes)) or not isinstance(
        clustered_samples,
        Sequence,
    ):
        raise TypeError("clustered_samples must be a sequence of (bits, weights) pairs.")

    n_encoded = 2 * n_logical
    n_clusters = len(clustered_samples)
    refs = np.zeros((n_clusters, n_encoded), dtype=np.float64)

    for k, cluster in enumerate(clustered_samples):
        if not isinstance(cluster, (tuple, list)) or len(cluster) != 2:
            raise TypeError(
                f"clustered_samples[{k}] must be a (bits, weights) pair."
            )
        bits, weights = cluster
        bits = _as_exact_binary_uint8(
            f"clustered_samples[{k}].bitstrings",
            bits,
        )
        weights = _as_finite_nonnegative_weights(
            f"clustered_samples[{k}].weights",
            weights,
        )
        if bits.shape[1] != n_encoded:
            raise ValueError(
                f"clustered_samples[{k}].bitstrings must have width {n_encoded}. "
                f"Got {bits.shape[1]}."
            )
        if weights.shape[0] != bits.shape[0]:
            raise ValueError(
                f"clustered_samples[{k}].weights must match the number of rows."
            )
        if len(bits) == 0 or float(np.sum(weights)) <= 0.0:
            # Empty clusters start from a neutral pair reference.
            refs[k, 0::2] = 0.5
            refs[k, 1::2] = 0.5
            continue

        # Pair normalization removes the global weight scale.
        refs[k] = weights @ bits.astype(np.float64)

    refs = pairwise_normalize_reference_vectors(refs)
    return refs


@nb.njit(cache=True, parallel=True)
def _assign_to_nearest_reference_kernel(
    encoded_bitstrings: np.ndarray,
    reference_vectors: np.ndarray,
) -> np.ndarray:
    """Assign rows with deterministic lowest-index tie breaking."""
    num_samples, n_encoded = encoded_bitstrings.shape
    n_clusters = reference_vectors.shape[0]
    labels = np.empty(num_samples, dtype=np.int64)

    for sample_index in nb.prange(num_samples):
        best_cluster = 0
        best_distance = np.inf
        for cluster_index in range(n_clusters):
            distance = 0.0
            for encoded_index in range(n_encoded):
                difference = (
                    float(encoded_bitstrings[sample_index, encoded_index])
                    - reference_vectors[cluster_index, encoded_index]
                )
                distance += abs(difference)
            # Strict comparison keeps the lowest cluster index on exact ties.
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster_index
        labels[sample_index] = best_cluster

    return labels


def assign_to_nearest_reference(
    encoded_bitstrings: np.ndarray,
    reference_vectors: np.ndarray,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Assign encoded bitstrings to the nearest reference by L1 distance.

    ``chunk_size`` is the positive number of rows per Numba-kernel dispatch.
    Distances are evaluated without materializing a
    ``(chunk_size, n_clusters)`` matrix. Exact ties select the lowest cluster
    index.
    """
    chunk_size = _validate_positive_int("chunk_size", chunk_size)
    encoded_bitstrings = _as_exact_binary_uint8(
        "encoded_bitstrings",
        encoded_bitstrings,
    )
    reference_vectors = _as_reference_vectors(
        "reference_vectors",
        reference_vectors,
        require_normalized=True,
    )

    num_samples = encoded_bitstrings.shape[0]
    n_clusters = reference_vectors.shape[0]
    if n_clusters <= 0:
        raise ValueError("reference_vectors must contain at least one cluster.")
    if encoded_bitstrings.shape[1] != reference_vectors.shape[1]:
        raise ValueError(
            "encoded_bitstrings and reference_vectors must have the same width."
        )
    labels = np.empty(num_samples, dtype=np.int64)

    for start in range(0, num_samples, chunk_size):
        stop = min(start + chunk_size, num_samples)
        labels[start:stop] = _assign_to_nearest_reference_kernel(
            encoded_bitstrings[start:stop],
            reference_vectors,
        )

    return labels


# =============================================================================
# Sparse invalid-pair recovery
# =============================================================================


def build_sparse_invalid_pairs(
    all_encoded_bitstrings: np.ndarray,
    labels: np.ndarray,
    reference_vectors: np.ndarray,
) -> SparseInvalidPairs:
    """Collect only invalid encoded pairs and their cluster references."""
    all_encoded_bitstrings = _as_exact_binary_uint8(
        "all_encoded_bitstrings",
        all_encoded_bitstrings,
    )
    reference_vectors = _as_reference_vectors(
        "reference_vectors",
        reference_vectors,
        require_normalized=True,
    )

    num_samples, n_encoded = all_encoded_bitstrings.shape
    if n_encoded % 2 != 0:
        raise ValueError("all_encoded_bitstrings must have an even number of columns.")
    n_clusters = reference_vectors.shape[0]
    if n_clusters <= 0:
        raise ValueError("reference_vectors must contain at least one cluster.")
    if reference_vectors.shape[1] != n_encoded:
        raise ValueError(
            "all_encoded_bitstrings and reference_vectors must have the same width."
        )
    labels = _as_cluster_labels(
        labels,
        num_samples=num_samples,
        n_clusters=n_clusters,
    )
    n_logical = n_encoded // 2

    pairs = all_encoded_bitstrings.reshape(num_samples, n_logical, 2)
    pair_sums = pairs.sum(axis=2)
    invalid_mask = pair_sums != 1

    sample_indices, pair_indices = np.nonzero(invalid_mask)
    sample_indices = sample_indices.astype(np.int64, copy=False)
    pair_indices = pair_indices.astype(np.int64, copy=False)

    if len(sample_indices) == 0:
        return SparseInvalidPairs(
            sample_indices=np.empty(0, dtype=np.int64),
            pair_indices=np.empty(0, dtype=np.int64),
            cluster_indices=np.empty(0, dtype=np.int64),
            invalid_pair_bits=np.empty((0, 2), dtype=np.uint8),
            invalid_pair_reference=np.empty((0, 2), dtype=np.float64),
        )

    cluster_indices = labels[sample_indices].astype(np.int64, copy=False)
    invalid_pair_bits = pairs[sample_indices, pair_indices].astype(np.uint8, copy=True)

    ref_pairs = reference_vectors.reshape(reference_vectors.shape[0], n_logical, 2)
    invalid_pair_reference = ref_pairs[cluster_indices, pair_indices].astype(np.float64, copy=True)

    return SparseInvalidPairs(
        sample_indices=sample_indices,
        pair_indices=pair_indices,
        cluster_indices=cluster_indices,
        invalid_pair_bits=invalid_pair_bits,
        invalid_pair_reference=invalid_pair_reference,
    )


def mrelu_recover_encoded_samples(
    all_encoded_bitstrings: np.ndarray,
    all_weights: np.ndarray,
    labels: np.ndarray,
    reference_vectors: np.ndarray,
    recovery_prob_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> RecoveryResult:
    """Run modified-ReLU invalid-pair recovery and return metadata."""
    recovered = _as_exact_binary_uint8(
        "all_encoded_bitstrings",
        all_encoded_bitstrings,
    ).copy()
    all_weights = _as_finite_nonnegative_weights("all_weights", all_weights)
    if all_weights.shape[0] != recovered.shape[0]:
        raise ValueError(
            "all_weights must have one entry per encoded sample. "
            f"Got {all_weights.shape[0]} weights for {recovered.shape[0]} samples."
        )
    if not callable(recovery_prob_fn):
        raise TypeError("recovery_prob_fn must be callable.")

    invalid_data = build_sparse_invalid_pairs(recovered, labels, reference_vectors)
    num_invalid_pairs = len(invalid_data.sample_indices)

    if num_invalid_pairs > 0:
        try:
            p_first = recovery_prob_fn(
                invalid_data.invalid_pair_bits,
                invalid_data.invalid_pair_reference,
            )
        except Exception as exc:
            raise RuntimeError(
                "recovery_prob_fn failed during mReLU recovery: "
                f"sample_index={int(invalid_data.sample_indices[0])}, "
                f"pair_index={int(invalid_data.pair_indices[0])}, "
                f"cluster_index={int(invalid_data.cluster_indices[0])}: {exc}"
            ) from exc
        p_first_raw = np.asarray(p_first)
        if p_first_raw.shape != (num_invalid_pairs,):
            raise ValueError(
                "recovery_prob_fn must return p_first with shape "
                f"({num_invalid_pairs},). Got {p_first_raw.shape}."
            )
        p_first = _as_finite_real_array(
            "recovery_prob_fn output",
            p_first_raw,
            ndim=1,
        )
        if np.any((p_first < 0.0) | (p_first > 1.0)):
            raise ValueError("recovery_prob_fn must return probabilities in [0, 1].")

        flip_first = rng.random(num_invalid_pairs) < p_first
        flip_columns = 2 * invalid_data.pair_indices + (~flip_first).astype(np.int64)
        rows = invalid_data.sample_indices
        recovered[rows, flip_columns] = 1 - recovered[rows, flip_columns]

    if not np.all(is_valid_encoded_bitstrings(recovered)):
        raise RuntimeError("Internal error: mReLU recovery produced invalid encoded bitstrings.")

    # Valid pair-code decoding is bijective and preserves lexicographic order
    # (01 < 10).  Merge on the <=64-bit logical representation so the hot path
    # uses uint64 keys instead of sorting 72-byte void scalars, then restore the
    # public encoded-output contract.
    recovered_logical = decode_valid_encoded_to_logical(recovered)
    recovered_logical_unique, recovered_weights = _merge_duplicate_rows_sum_weights(
        recovered_logical,
        all_weights,
    )
    recovered_unique = encode_logical_to_valid_encoded(recovered_logical_unique)
    prob_metadata = _callable_metadata(recovery_prob_fn) or {}
    metadata = {
        "method": "mrelu",
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "num_invalid_pairs": int(num_invalid_pairs),
        "num_zero_denominator_errors": 0,
        "num_recovered_unique_samples": int(len(recovered_unique)),
        "output_rows_are_unique": True,
        "recovery_prob_fn": getattr(recovery_prob_fn, "__name__", repr(recovery_prob_fn)),
        "recovery_prob_fn_config": dict(prob_metadata),
    }
    for key in ("delta", "corner", "type", "family"):
        if key in prob_metadata:
            metadata[key] = prob_metadata[key]
    return RecoveryResult(recovered_unique, recovered_weights, metadata)


def _validate_recovery_result(result: RecoveryResult, *, n_encoded: int) -> RecoveryResult:
    if not isinstance(result, RecoveryResult):
        raise TypeError("recovery_fn must return a RecoveryResult.")
    bitstrings = _as_exact_binary_uint8(
        "RecoveryResult.bitstrings",
        result.bitstrings,
    )
    weights = _as_finite_nonnegative_weights(
        "RecoveryResult.weights",
        result.weights,
    )
    if bitstrings.shape[1] != n_encoded:
        raise ValueError(
            "RecoveryResult.bitstrings must have shape "
            f"(n_recovered, {n_encoded}). Got {bitstrings.shape}."
        )
    if weights.shape[0] != bitstrings.shape[0]:
        raise ValueError("RecoveryResult.weights must be 1D and match bitstrings.")
    if not np.all(is_valid_encoded_bitstrings(bitstrings)):
        raise ValueError("RecoveryResult.bitstrings must all satisfy the pair code.")
    if not isinstance(result.metadata, Mapping):
        raise TypeError("RecoveryResult.metadata must be a mapping/dict.")
    metadata = dict(result.metadata)
    return RecoveryResult(bitstrings.astype(np.uint8, copy=True), weights, metadata)


# =============================================================================
# Batch construction
# =============================================================================


def _weighted_sample_without_replacement(
    weights: np.ndarray,
    n_draw: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw indices without replacement using non-negative weights.

    Positive-weight rows are sampled first. If too few positive weights are
    available, the remaining slots are filled uniformly from zero-weight rows.
    """
    weights = np.asarray(weights, dtype=np.float64)
    n_available = len(weights)
    n_draw = min(int(n_draw), n_available)

    if n_draw <= 0:
        return np.empty(0, dtype=np.int64)

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite.")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative.")

    positive_indices = np.flatnonzero(weights > 0.0)

    # Fall back to uniform sampling when all weights are zero.
    if len(positive_indices) == 0:
        return rng.choice(n_available, size=n_draw, replace=False).astype(np.int64)

    # Use weighted sampling when enough positive-weight rows are available.
    if len(positive_indices) >= n_draw:
        positive_weights = weights[positive_indices]
        with np.errstate(over="ignore", invalid="ignore"):
            positive_total = float(np.sum(positive_weights))
        if math.isfinite(positive_total) and positive_total > 0.0:
            # Preserve the historical arithmetic for ordinary finite totals.
            p = positive_weights / positive_total
        else:
            # Scale only on overflow so finite large weights remain usable.
            scale = float(np.max(positive_weights))
            scaled_weights = positive_weights / scale
            scaled_total = float(np.sum(scaled_weights))
            if not math.isfinite(scaled_total) or scaled_total <= 0.0:
                raise ValueError("positive weights could not be normalized safely.")
            p = scaled_weights / scaled_total
        # `Generator.choice(..., replace=False, p=...)` requires at least
        # `n_draw` strictly positive probabilities. Extremely different finite
        # magnitudes can underflow during normalization, so fall back to a
        # uniform draw over the positive-weight rows in that pathological case.
        representable = np.flatnonzero(p > 0.0)
        if len(representable) < n_draw:
            # Preserve every state whose normalized probability remains
            # representable, then fill only from weights that underflowed to
            # zero. This keeps a dominant state from being lost to a uniform
            # fallback over the entire pool.
            underflowed = np.flatnonzero(p == 0.0)
            fill = rng.choice(
                underflowed,
                size=n_draw - len(representable),
                replace=False,
            )
            chosen_local = np.concatenate([representable, fill])
            rng.shuffle(chosen_local)
        else:
            chosen_local = rng.choice(
                len(positive_indices),
                size=n_draw,
                replace=False,
                p=p,
            )
        return positive_indices[chosen_local].astype(np.int64)

    # Keep all positive-weight rows and fill the rest uniformly.
    chosen = [positive_indices]
    remaining = n_draw - len(positive_indices)

    zero_indices = np.flatnonzero(weights <= 0.0)
    if remaining > 0:
        fill = rng.choice(zero_indices, size=remaining, replace=False)
        chosen.append(fill.astype(np.int64, copy=False))

    out = np.concatenate(chosen).astype(np.int64, copy=False)
    rng.shuffle(out)
    return out


def _maximum_available_batch_dimension(
    recovered_logical_pool: np.ndarray,
    *,
    deterministic_logical_basis: np.ndarray,
    carryover_logical_basis: np.ndarray,
    forced_valid_initial_logical_basis: np.ndarray | None,
    max_dim: int,
) -> int:
    """Return the largest batch dimension reachable from the current pool."""
    recovered_logical_pool = np.asarray(recovered_logical_pool, dtype=np.uint8)
    deterministic_logical_basis = np.asarray(
        deterministic_logical_basis,
        dtype=np.uint8,
    )
    carryover_logical_basis = np.asarray(
        carryover_logical_basis,
        dtype=np.uint8,
    )

    n_logical = recovered_logical_pool.shape[1]
    if forced_valid_initial_logical_basis is None:
        forced_valid_initial_logical_basis = np.empty(
            (0, n_logical),
            dtype=np.uint8,
        )
    else:
        forced_valid_initial_logical_basis = np.asarray(
            forced_valid_initial_logical_basis,
            dtype=np.uint8,
        )

    forced_parts = (
        deterministic_logical_basis,
        carryover_logical_basis,
        forced_valid_initial_logical_basis,
    )
    forced_nonempty = [part for part in forced_parts if len(part) > 0]
    if forced_nonempty:
        forced = _unique_rows_preserve_order(
            np.vstack(forced_nonempty).astype(np.uint8, copy=False)
        )
    else:
        forced = np.empty((0, n_logical), dtype=np.uint8)

    if len(forced) >= max_dim:
        return int(max_dim)

    available_mask = _filter_rows_not_in_set(recovered_logical_pool, forced)
    return int(min(max_dim, len(forced) + int(np.count_nonzero(available_mask))))


def build_one_batch(
    recovered_logical_pool: np.ndarray,
    recovered_weights: np.ndarray,
    *,
    deterministic_logical_basis: np.ndarray,
    carryover_logical_basis: np.ndarray,
    max_dim: int,
    rng: np.random.Generator,
    forced_valid_initial_logical_basis: np.ndarray | None = None,
) -> np.ndarray:
    """Build one unique logical-basis batch.

    ``max_dim`` is the target maximum number of unique rows in the final batch,
    not the number of random draws. Deterministic rows, carry-over rows, and
    optional forced valid initial rows are inserted before weighted sampling
    from the recovered logical pool. The batch can be smaller than ``max_dim``
    when the unique eligible pool is insufficient.
    """
    max_dim = _validate_positive_int("max_dim", max_dim)
    recovered_logical_pool = _as_exact_binary_uint8(
        "recovered_logical_pool",
        recovered_logical_pool,
    )
    recovered_weights = _as_finite_nonnegative_weights(
        "recovered_weights",
        recovered_weights,
    )
    if recovered_weights.shape[0] != recovered_logical_pool.shape[0]:
        raise ValueError(
            "recovered_weights must have one entry per recovered logical row."
        )
    deterministic_logical_basis = _as_exact_binary_uint8(
        "deterministic_logical_basis",
        deterministic_logical_basis,
    )
    carryover_logical_basis = _as_exact_binary_uint8(
        "carryover_logical_basis",
        carryover_logical_basis,
    )
    n_logical = recovered_logical_pool.shape[1]
    for name, basis in (
        ("deterministic_logical_basis", deterministic_logical_basis),
        ("carryover_logical_basis", carryover_logical_basis),
    ):
        if basis.shape[1] != n_logical:
            raise ValueError(
                f"{name} must have width {n_logical}. Got {basis.shape[1]}."
            )
    if forced_valid_initial_logical_basis is not None:
        forced_valid_initial_logical_basis = _as_exact_binary_uint8(
            "forced_valid_initial_logical_basis",
            forced_valid_initial_logical_basis,
        )
        if forced_valid_initial_logical_basis.shape[1] != n_logical:
            raise ValueError(
                "forced_valid_initial_logical_basis must have width "
                f"{n_logical}. Got {forced_valid_initial_logical_basis.shape[1]}."
            )

    forced, available_pool, available_weights = _prepare_batch_sampling_inputs(
        recovered_logical_pool,
        recovered_weights,
        deterministic_logical_basis=deterministic_logical_basis,
        carryover_logical_basis=carryover_logical_basis,
        forced_valid_initial_logical_basis=forced_valid_initial_logical_basis,
    )
    return _build_one_batch_from_prepared_inputs(
        forced,
        available_pool,
        available_weights,
        max_dim=max_dim,
        rng=rng,
        enforce_output_uniqueness=True,
    )


def _prepare_batch_sampling_inputs(
    recovered_logical_pool: np.ndarray,
    recovered_weights: np.ndarray,
    *,
    deterministic_logical_basis: np.ndarray,
    carryover_logical_basis: np.ndarray,
    forced_valid_initial_logical_basis: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute forced rows and the eligible weighted pool once per iteration."""
    recovered_logical_pool = np.asarray(recovered_logical_pool, dtype=np.uint8)
    recovered_weights = np.asarray(recovered_weights, dtype=np.float64)
    deterministic_logical_basis = np.asarray(
        deterministic_logical_basis,
        dtype=np.uint8,
    )
    carryover_logical_basis = np.asarray(
        carryover_logical_basis,
        dtype=np.uint8,
    )
    n_logical = recovered_logical_pool.shape[1]
    if forced_valid_initial_logical_basis is None:
        forced_valid_initial_logical_basis = np.empty(
            (0, n_logical),
            dtype=np.uint8,
        )
    else:
        forced_valid_initial_logical_basis = np.asarray(
            forced_valid_initial_logical_basis,
            dtype=np.uint8,
        )

    forced_nonempty = [
        part
        for part in (
            deterministic_logical_basis,
            carryover_logical_basis,
            forced_valid_initial_logical_basis,
        )
        if len(part) > 0
    ]
    if forced_nonempty:
        forced = _unique_rows_preserve_order(
            np.vstack(forced_nonempty).astype(np.uint8, copy=False)
        )
    else:
        forced = np.empty((0, n_logical), dtype=np.uint8)

    available_mask = _filter_rows_not_in_set(recovered_logical_pool, forced)
    return (
        forced,
        recovered_logical_pool[available_mask],
        recovered_weights[available_mask],
    )


def _build_one_batch_from_prepared_inputs(
    forced: np.ndarray,
    available_pool: np.ndarray,
    available_weights: np.ndarray,
    *,
    max_dim: int,
    rng: np.random.Generator,
    enforce_output_uniqueness: bool,
) -> np.ndarray:
    """Sample one batch from iteration-level precomputed arrays."""
    if len(forced) >= max_dim:
        return forced[:max_dim].astype(np.uint8, copy=True)
    remaining = max_dim - len(forced)
    if len(available_pool) == 0 or remaining <= 0:
        return forced.astype(np.uint8, copy=True)

    n_draw = min(remaining, len(available_pool))
    chosen_indices = _weighted_sample_without_replacement(
        available_weights,
        n_draw,
        rng,
    )
    chosen = available_pool[chosen_indices]
    batch = np.vstack([forced, chosen]) if len(forced) > 0 else chosen
    if enforce_output_uniqueness:
        # Keep the public helper robust to duplicate rows in the recovered pool.
        batch = _unique_rows_preserve_order(batch)
        return batch[:max_dim].astype(np.uint8, copy=True)
    # Prepared inputs are unique and disjoint, and sampling is without
    # replacement, so no additional full-batch unique pass is needed.
    return np.ascontiguousarray(batch[:max_dim], dtype=np.uint8)


# =============================================================================
# Carry-over and reference update
# =============================================================================


def select_carryover_basis(
    best_logical_basis: np.ndarray,
    best_coefficients: np.ndarray,
    *,
    carryover_threshold: float,
    max_keep: int,
) -> np.ndarray:
    """Select high-amplitude global-best basis states for the next iteration.

    Rows with ``abs(coefficient) >= carryover_threshold`` are candidates. When
    too many candidates are present, ``np.argpartition`` avoids a full sort.
    """
    best_logical_basis = _as_exact_binary_uint8(
        "best_logical_basis",
        best_logical_basis,
    )
    best_coefficients = np.asarray(best_coefficients)
    if best_coefficients.ndim != 1:
        raise ValueError("best_coefficients must be a 1D array.")
    if not np.issubdtype(best_coefficients.dtype, np.number):
        raise TypeError(
            "best_coefficients must have a numeric dtype. "
            f"Got {best_coefficients.dtype}."
        )
    if not np.all(np.isfinite(best_coefficients)):
        raise ValueError("best_coefficients must contain only finite values.")
    if best_coefficients.shape[0] != best_logical_basis.shape[0]:
        raise ValueError(
            "best_coefficients must have one entry per logical basis row."
        )
    carryover_threshold = _validate_unit_interval_real(
        "carryover_threshold",
        carryover_threshold,
    )
    max_keep = _validate_nonnegative_int("max_keep", max_keep)

    if len(best_logical_basis) == 0 or max_keep <= 0:
        return np.empty((0, best_logical_basis.shape[1]), dtype=np.uint8)

    with np.errstate(over="ignore", invalid="ignore"):
        amplitudes = np.abs(best_coefficients)
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("Absolute best_coefficients must be finite.")
    candidate_indices = np.flatnonzero(amplitudes >= carryover_threshold)
    if len(candidate_indices) == 0:
        return np.empty((0, best_logical_basis.shape[1]), dtype=np.uint8)

    candidate_amp = amplitudes[candidate_indices]

    if len(candidate_indices) > max_keep:
        top_local = np.argpartition(candidate_amp, -max_keep)[-max_keep:]
        # Return selected rows in descending amplitude order.
        top_local = top_local[np.argsort(candidate_amp[top_local])[::-1]]
    else:
        top_local = np.argsort(candidate_amp)[::-1]

    selected_indices = candidate_indices[top_local]
    return best_logical_basis[selected_indices].astype(np.uint8, copy=True)


def update_reference_vectors(
    best_logical_basis: np.ndarray,
    best_coefficients: np.ndarray,
    old_reference_vectors: np.ndarray,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Update cluster reference vectors from the global-best wavefunction.

    The logical basis is re-encoded, weighted by ``|coefficient|^2``, assigned
    to the nearest old reference, and averaged within each cluster. Empty
    clusters keep their previous reference. ``chunk_size`` is the positive
    number of encoded rows per assignment dispatch.
    """
    chunk_size = _validate_positive_int("chunk_size", chunk_size)
    best_logical_basis = _as_exact_binary_uint8(
        "best_logical_basis",
        best_logical_basis,
    )
    best_coefficients = np.asarray(best_coefficients)
    if best_coefficients.ndim != 1:
        raise ValueError(
            "best_coefficients must be a 1D array. "
            f"Got shape {best_coefficients.shape}."
        )
    if not np.issubdtype(best_coefficients.dtype, np.number):
        raise TypeError(
            "best_coefficients must have a numeric dtype. "
            f"Got {best_coefficients.dtype}."
        )
    if not np.all(np.isfinite(best_coefficients)):
        raise ValueError("best_coefficients must contain only finite values.")
    if best_coefficients.shape[0] != best_logical_basis.shape[0]:
        raise ValueError(
            "best_coefficients must have one entry per logical basis row."
        )
    old_reference_vectors = _as_reference_vectors(
        "old_reference_vectors",
        old_reference_vectors,
        require_normalized=True,
    )

    n_clusters, n_encoded = old_reference_vectors.shape
    if n_clusters <= 0:
        raise ValueError("old_reference_vectors must contain at least one cluster.")
    if best_logical_basis.shape[1] * 2 != n_encoded:
        raise ValueError(
            "best_logical_basis width must be half the old reference width. "
            f"Got {best_logical_basis.shape[1]} and {n_encoded}."
        )

    if len(best_logical_basis) == 0:
        return old_reference_vectors.copy()

    encoded_basis = encode_logical_to_valid_encoded(best_logical_basis)
    with np.errstate(over="ignore", invalid="ignore"):
        coeff_weights = np.abs(best_coefficients).astype(np.float64) ** 2
    if not np.all(np.isfinite(coeff_weights)):
        raise ValueError("Squared best_coefficients must be finite.")

    total_coeff_weight = float(np.sum(coeff_weights))
    if total_coeff_weight <= 0.0 or not math.isfinite(total_coeff_weight):
        raise ValueError(
            "best_coefficients must have a finite positive squared norm "
            "for a nonempty logical basis."
        )
    coeff_weights = coeff_weights / total_coeff_weight

    labels = assign_to_nearest_reference(
        encoded_basis,
        old_reference_vectors,
        chunk_size=chunk_size,
    )

    new_refs = old_reference_vectors.copy()

    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            # Keep the previous reference for empty reassigned clusters.
            continue

        w = coeff_weights[mask]
        bits = encoded_basis[mask].astype(np.float64, copy=False)
        denom = float(np.sum(w))
        if denom <= 0.0:
            continue

        new_refs[k] = (w @ bits) / denom

    new_refs = pairwise_normalize_reference_vectors(new_refs)
    return new_refs


# =============================================================================
# diagonalize_fn call and validation
# =============================================================================


def _call_and_validate_diagonalize_fn(
    diagonalize_fn: Callable[..., DiagonalizationResult],
    hamiltonian: Any,
    logical_basis: np.ndarray,
    *,
    seed: int | None,
    warm_start_basis: np.ndarray | None = None,
    warm_start_keys: np.ndarray | None = None,
    warm_start_coefficients: np.ndarray | None = None,
) -> DiagonalizationResult:
    """Call ``diagonalize_fn`` and validate the returned Ritz pair.

    diagonalize_fn expected signature:
        diagonalize_fn(
            hamiltonian,
            logical_basis,
            *,
            seed=None,
            warm_start_basis=None,
            warm_start_keys=None,
            warm_start_coefficients=None,
        ) -> DiagonalizationResult
    """
    if warm_start_keys is not None and warm_start_basis is not None:
        raise ValueError("Pass either warm_start_basis or warm_start_keys, not both.")
    if warm_start_keys is not None and warm_start_coefficients is None:
        raise ValueError(
            "warm_start_keys and warm_start_coefficients must be provided together."
        )
    if warm_start_keys is None and (
        (warm_start_basis is None) != (warm_start_coefficients is None)
    ):
        raise ValueError(
            "warm_start_basis and warm_start_coefficients must be provided together."
        )

    kwargs: dict[str, Any] = {"seed": seed}
    if warm_start_keys is not None and warm_start_coefficients is not None:
        kwargs["warm_start_keys"] = warm_start_keys
        kwargs["warm_start_coefficients"] = warm_start_coefficients
    elif warm_start_basis is not None and warm_start_coefficients is not None:
        kwargs["warm_start_basis"] = warm_start_basis
        kwargs["warm_start_coefficients"] = warm_start_coefficients

    result = diagonalize_fn(hamiltonian, logical_basis, **kwargs)

    # Accept compatible result objects from external modules by duck typing.
    required_attrs = ("energy", "logical_basis", "coefficients")
    if not all(hasattr(result, attr) for attr in required_attrs):
        raise TypeError(
            "diagonalize_fn must return an object with attributes "
            "energy, logical_basis, and coefficients."
        )

    if isinstance(result.energy, (bool, np.bool_)) or not isinstance(
        result.energy,
        numbers.Real,
    ):
        raise TypeError("diagonalize_fn energy must be a real numeric value, not bool.")
    energy = float(result.energy)
    if not math.isfinite(energy):
        raise ValueError("diagonalize_fn returned non-finite energy.")

    basis = np.asarray(result.logical_basis)
    coeffs = np.asarray(result.coefficients)

    if basis.ndim != 2:
        raise ValueError("DiagonalizationResult.logical_basis must be a 2D array.")
    if basis.dtype != np.uint8:
        raise TypeError("DiagonalizationResult.logical_basis must have dtype np.uint8.")
    if coeffs.ndim != 1:
        raise ValueError("DiagonalizationResult.coefficients must be a 1D array.")
    if coeffs.shape[0] != basis.shape[0]:
        raise ValueError(
            "DiagonalizationResult.coefficients length must match logical_basis rows."
        )
    if not np.all((basis == 0) | (basis == 1)):
        raise ValueError("DiagonalizationResult.logical_basis must contain only 0 or 1.")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("DiagonalizationResult.coefficients must be finite.")

    if basis.shape[1] != logical_basis.shape[1]:
        raise ValueError(
            "DiagonalizationResult.logical_basis has the wrong number of logical bits. "
            f"Expected {logical_basis.shape[1]}, got {basis.shape[1]}."
        )

    # The diagonalizer must only return rows from the provided batch.
    outside_mask = _filter_rows_not_in_set(basis, logical_basis)
    if np.any(outside_mask):
        raise ValueError(
            "DiagonalizationResult.logical_basis must be a subset of the input "
            "batch logical_basis. The diagonalizer must not add basis states that "
            "were not provided by run_code_space_recovery."
        )

    # Duplicate output rows would make the coefficient-to-basis map ambiguous.
    if len(_unique_rows_preserve_order(basis)) != len(basis):
        raise ValueError(
            "DiagonalizationResult.logical_basis must not contain duplicate rows. "
            "The diagonalizer should merge/deduplicate its basis before returning."
        )

    # Normalize minor numerical drift, but reject invalid coefficient norms.
    norm = float(np.sum(np.abs(coeffs) ** 2))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("DiagonalizationResult.coefficients have invalid norm.")

    coeffs = coeffs / math.sqrt(norm)

    return DiagonalizationResult(
        energy=energy,
        logical_basis=basis.astype(np.uint8, copy=True),
        coefficients=coeffs.astype(np.complex128, copy=False),
    )


# =============================================================================
# Batch diagonalization helper
# =============================================================================


def _resolve_batch_parallel_config(
    *,
    n_batches: int,
    parallelize_batches: bool,
    max_parallel_batches: int | None,
    diag_num_threads_per_batch: int | None,
) -> tuple[int, int | None, int]:
    available_cpu_threads = int(os.cpu_count() or 1)
    if parallelize_batches:
        effective_parallel_batches = min(
            n_batches,
            max_parallel_batches if max_parallel_batches is not None else n_batches,
        )
        if diag_num_threads_per_batch is None:
            resolved_threads = max(1, available_cpu_threads // effective_parallel_batches)
        else:
            resolved_threads = diag_num_threads_per_batch
    else:
        effective_parallel_batches = 1
        resolved_threads = diag_num_threads_per_batch

    return effective_parallel_batches, resolved_threads, available_cpu_threads


def _validate_batch_parallel_backend(value: Any) -> str:
    """Normalize the implementation used for concurrent batch solves."""
    if not isinstance(value, str):
        raise TypeError(
            "batch_parallel_backend must be 'threading' or 'loky', "
            f"got {type(value).__name__}."
        )
    backend = value.strip().lower()
    if backend not in {"threading", "loky"}:
        raise ValueError(
            "batch_parallel_backend must be 'threading' or 'loky', "
            f"got {value!r}."
        )
    return backend


def _validate_batch_parallel_max_nbytes(value: Any) -> int | str | None:
    """Validate joblib's input-array memmap threshold without importing joblib."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            "batch_parallel_max_nbytes must be None, a non-negative integer, "
            "or a joblib size string such as '1M'."
        )
    if isinstance(value, numbers.Integral):
        value_int = int(value)
        if value_int < 0:
            raise ValueError("batch_parallel_max_nbytes must be >= 0.")
        return value_int
    if isinstance(value, str):
        value_str = value.strip()
        if not value_str:
            raise ValueError("batch_parallel_max_nbytes must not be an empty string.")
        return value_str
    raise TypeError(
        "batch_parallel_max_nbytes must be None, a non-negative integer, "
        "or a joblib size string such as '1M'."
    )


def _validate_batch_parallel_temp_folder(value: Any) -> str | None:
    """Normalize the optional joblib temporary directory."""
    if value is None:
        return None
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise TypeError(
            "batch_parallel_temp_folder must be None or path-like."
        ) from exc
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    path = str(path).strip()
    if not path:
        raise ValueError("batch_parallel_temp_folder must not be empty.")
    return path


def _spawn_diagonalize_fn(
    diagonalize_fn: Callable[..., DiagonalizationResult],
    *,
    num_threads: int | None,
) -> Callable[..., DiagonalizationResult]:
    if num_threads is None:
        return diagonalize_fn
    spawn = getattr(diagonalize_fn, "spawn", None)
    if callable(spawn):
        return spawn(num_threads=num_threads)
    return diagonalize_fn


def _diagonalize_batch_worker(
    batch_index: int,
    diagonalize_fn: Callable[..., DiagonalizationResult],
    hamiltonian: Any,
    batch_logical_basis: np.ndarray,
    diag_seed: int,
    num_threads: int | None,
    warm_start_basis: np.ndarray | None,
    warm_start_keys: np.ndarray | None,
    warm_start_coefficients: np.ndarray | None,
    batch_build_seconds: float = 0.0,
    batch_basis_sha256: str | None = None,
    batch_seed: int | None = None,
) -> tuple[int, DiagonalizationResult, float, dict[str, Any]]:
    """Top-level, picklable implementation shared by thread and process workers."""
    worker_diagonalize_fn = _spawn_diagonalize_fn(
        diagonalize_fn,
        num_threads=num_threads,
    )
    diag_start = time.perf_counter()
    diag_result = _call_and_validate_diagonalize_fn(
        worker_diagonalize_fn,
        hamiltonian,
        batch_logical_basis,
        seed=diag_seed,
        warm_start_basis=warm_start_basis,
        warm_start_keys=warm_start_keys,
        warm_start_coefficients=warm_start_coefficients,
    )
    diag_seconds = time.perf_counter() - diag_start
    actual_num_threads: int | None = None
    get_runtime_num_threads = getattr(
        worker_diagonalize_fn,
        "get_runtime_num_threads",
        None,
    )
    if callable(get_runtime_num_threads):
        actual_num_threads = int(get_runtime_num_threads())
        if num_threads is not None and actual_num_threads != int(num_threads):
            raise RuntimeError(
                "Numerical worker thread count mismatch: "
                f"requested={num_threads}, actual={actual_num_threads}. "
                "Start each Loky worker in a fresh process so "
                "NUMBA_NUM_THREADS is set before Numba is imported."
            )
    stats = getattr(worker_diagonalize_fn, "last_stats", None)
    stats_metadata: dict[str, Any] | None = None
    if stats is not None:
        stats_metadata = {
            "package_version": str(
                getattr(stats, "package_version", PACKAGE_VERSION)
            ),
            "algorithm_version": str(
                getattr(stats, "algorithm_version", ALGORITHM_VERSION)
            ),
        }
        for key in (
            "basis_dim",
            "nnz",
            "csr_memory_gib",
            "pack_seconds",
            "hash_seconds",
            "count_seconds",
            "fill_seconds",
            "solve_seconds",
            "residual_norm",
            "relative_residual",
            "used_warm_start",
        ):
            value = getattr(stats, key, None)
            if isinstance(value, (bool, np.bool_)):
                stats_metadata[key] = bool(value)
            elif isinstance(value, (np.integer, numbers.Integral)):
                stats_metadata[key] = int(value)
            elif isinstance(value, (np.floating, numbers.Real)):
                stats_metadata[key] = float(value)
            elif value is None:
                stats_metadata[key] = None

    worker_metadata = {
        "worker_pid": int(os.getpid()),
        "requested_num_threads": (
            None if num_threads is None else int(num_threads)
        ),
        "actual_numba_threads": actual_num_threads,
        "batch_seed": None if batch_seed is None else int(batch_seed),
        "diag_seed": int(diag_seed),
        "batch_build_seconds": float(batch_build_seconds),
        "batch_basis_sha256": batch_basis_sha256,
        "diagonalization_seconds": float(diag_seconds),
        "diagonalizer_stats": stats_metadata,
    }
    return batch_index, diag_result, diag_seconds, worker_metadata


def _sha256_logical_basis(logical_basis: np.ndarray) -> str:
    basis = np.ascontiguousarray(logical_basis, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(x) for x in basis.shape)).encode("ascii"))
    digest.update(basis.dtype.str.encode("ascii"))
    digest.update(memoryview(basis).cast("B"))
    return digest.hexdigest()


def _derive_loky_batch_seeds(
    root_seed: int,
    max_dim: int,
    iteration: int,
    batch_index: int,
) -> tuple[int, int]:
    """Derive worker-scheduling-independent batch and diagonalizer seeds."""
    batch_sequence = np.random.SeedSequence(
        [int(root_seed), int(max_dim), int(iteration), int(batch_index), 0]
    )
    diagonalization_sequence = np.random.SeedSequence(
        [int(root_seed), int(max_dim), int(iteration), int(batch_index), 1]
    )
    batch_seed = int(batch_sequence.generate_state(1, dtype=np.uint64)[0])
    diag_word = diagonalization_sequence.generate_state(1, dtype=np.uint64)[0]
    diag_seed = int(diag_word % np.uint64(np.iinfo(np.int32).max))
    return batch_seed, diag_seed


def _loky_build_and_diagonalize_batch_worker(
    batch_index: int,
    diagonalize_fn: Callable[..., DiagonalizationResult],
    hamiltonian: Any,
    forced_logical_basis: np.ndarray,
    available_logical_pool: np.ndarray,
    available_weights: np.ndarray,
    max_dim: int,
    batch_seed: int,
    diag_seed: int,
    num_threads: int | None,
    record_basis_sha256: bool,
    warm_start_basis: np.ndarray | None,
    warm_start_keys: np.ndarray | None,
    warm_start_coefficients: np.ndarray | None,
) -> tuple[int, DiagonalizationResult, float, dict[str, Any]]:
    """Build and diagonalize one batch entirely inside an independent process."""
    try:
        build_start = time.perf_counter()
        batch_logical_basis = _build_one_batch_from_prepared_inputs(
            forced_logical_basis,
            available_logical_pool,
            available_weights,
            max_dim=max_dim,
            rng=np.random.default_rng(batch_seed),
            enforce_output_uniqueness=False,
        )
        batch_build_seconds = time.perf_counter() - build_start
        if len(batch_logical_basis) == 0:
            raise RuntimeError(
                "Constructed an empty batch. Check samples, deterministic basis, and max_dim."
            )
        batch_basis_sha256 = (
            _sha256_logical_basis(batch_logical_basis)
            if record_basis_sha256
            else None
        )
        return _diagonalize_batch_worker(
            batch_index,
            diagonalize_fn,
            hamiltonian,
            batch_logical_basis,
            diag_seed,
            num_threads,
            warm_start_basis,
            warm_start_keys,
            warm_start_coefficients,
            batch_build_seconds,
            batch_basis_sha256,
            batch_seed,
        )
    except Exception as exc:
        raise RuntimeError(f"batch={batch_index + 1}: {exc}") from exc


class _PersistentLokyBatchPool:
    """Small lifecycle wrapper keeping one joblib pool alive across iterations."""

    def __init__(
        self,
        *,
        n_jobs: int,
        diagonalize_fn: Callable[..., DiagonalizationResult],
        hamiltonian: Any,
        num_threads: int | None,
        max_nbytes: int | str | None,
        temp_folder: str | None,
    ) -> None:
        try:
            from joblib import Parallel, delayed, parallel_config  # type: ignore
        except Exception as exc:
            raise ImportError(
                "batch_parallel_backend='loky' requires joblib. "
                "Install it with `pip install joblib`."
            ) from exc

        self._delayed = delayed
        self._diagonalize_fn = diagonalize_fn
        self._hamiltonian = hamiltonian
        self._num_threads = num_threads
        self._entered = False
        self._closed = False
        # This public joblib context sets the child environment before Numba or
        # BLAS is imported. The configured backend object is retained by the
        # explicitly-entered Parallel instance after this context exits.
        config_context = parallel_config(
            backend="loky",
            n_jobs=n_jobs,
            inner_max_num_threads=num_threads,
            temp_folder=temp_folder,
            max_nbytes=max_nbytes,
            mmap_mode="r",
        )
        config_context.__enter__()
        try:
            self._parallel = Parallel(
                return_as="generator_unordered",
                batch_size=1,
                pre_dispatch="all",
            )
            self._parallel.__enter__()
            self._entered = True
        finally:
            config_context.__exit__(None, None, None)
        # Explicitly entering Parallel prevents it from releasing the backend
        # after a wave, so the same processes serve all recovery iterations.

    def run(
        self,
        batch_specs: Sequence[tuple[int, int, int]],
        *,
        forced_logical_basis: np.ndarray,
        available_logical_pool: np.ndarray,
        available_weights: np.ndarray,
        max_dim: int,
        record_basis_sha256: bool,
        warm_start_basis: np.ndarray | None,
        warm_start_keys: np.ndarray | None,
        warm_start_coefficients: np.ndarray | None,
    ) -> Iterable[tuple[int, DiagonalizationResult, float, dict[str, Any]]]:
        tasks = (
            self._delayed(_loky_build_and_diagonalize_batch_worker)(
                batch_index,
                self._diagonalize_fn,
                self._hamiltonian,
                forced_logical_basis,
                available_logical_pool,
                available_weights,
                max_dim,
                batch_seed,
                diag_seed,
                self._num_threads,
                record_basis_sha256,
                warm_start_basis,
                warm_start_keys,
                warm_start_coefficients,
            )
            for batch_index, batch_seed, diag_seed in batch_specs
        )
        return self._parallel(tasks)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._entered:
            self._parallel.__exit__(None, None, None)

    def __del__(self) -> None:  # pragma: no cover - exception-path fallback
        try:
            self.close()
        except Exception:
            pass


_ACTIVE_LOKY_BATCH_POOLS: ContextVar[list[_PersistentLokyBatchPool] | None] = (
    ContextVar("active_loky_batch_pools", default=None)
)


def _close_loky_batch_pools_on_exit(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Guarantee process-pool cleanup on every exit path from the public driver."""

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        pools: list[_PersistentLokyBatchPool] = []
        token = _ACTIVE_LOKY_BATCH_POOLS.set(pools)
        try:
            return fn(*args, **kwargs)
        finally:
            for pool in reversed(pools):
                try:
                    pool.close()
                except Exception:
                    # Cleanup must not mask the scientific failure that caused
                    # the driver to unwind. Normal completion closes explicitly.
                    pass
            _ACTIVE_LOKY_BATCH_POOLS.reset(token)

    return wrapped


# =============================================================================
# Main driver
# =============================================================================


@_close_loky_batch_pools_on_exit
def run_code_space_recovery(
    clustered_samples: Sequence[tuple[np.ndarray, np.ndarray]],
    hamiltonian: Any,
    *,
    n_batches: int,
    max_dim: int,
    min_iterations: int,
    max_iterations: int,
    convergence_patience: int,
    carryover_threshold: float,
    diagonalize_fn: Callable[..., DiagonalizationResult],
    recovery_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.random.Generator],
        RecoveryResult,
    ] | None = None,
    recovery_prob_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    recovery_schedule: Sequence[Mapping[str, Any]] | None = None,
    max_recovery_draws_per_iteration: int = 5,
    reassign_clusters_each_iteration: bool = True,
    use_global_best_warm_start: bool = True,
    parallelize_batches: bool = False,
    max_parallel_batches: int | None = None,
    diag_num_threads_per_batch: int | None = None,
    batch_parallel_backend: str = "threading",
    batch_parallel_max_nbytes: int | str | None = "1M",
    batch_parallel_temp_folder: str | os.PathLike[str] | None = None,
    record_batch_basis_sha256: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    seed: int | None = None,
    deterministic_logical_basis: np.ndarray | None = None,
) -> CodeSpaceRecoveryResult:
    """Run the self-consistent code-space recovery loop for pair-code samples.

    Parameters
    ----------
    clustered_samples:
        Clustered encoded samples. Each entry is
        ``(encoded_bitstrings_k, weights_k)`` with shape
        ``(n_unique_k, 2 * n_logical)``. Encoded rows must form a globally
        unique partition across all clusters. Weights may be raw counts or
        global weights and are normalized once over the complete sample pool.

    hamiltonian:
        Original logical Hamiltonian. Projection and diagonalization are
        performed in the logical ``n``-qubit space, not in the encoded
        ``2n``-qubit space. The object must expose ``num_qubits``.

    n_batches:
        Number of batches generated per iteration.

    max_dim:
        Target upper bound on unique logical basis states per batch. Batches
        can underfill when the eligible unique pool is too small. The reported
        batch dimension is the basis size returned by ``diagonalize_fn``.

    min_iterations:
        Default minimum iterations for an adaptive stage before patience-based
        stopping can trigger. Without a schedule, this applies to the sole
        adaptive stage; with a schedule, it fills omitted stage values.

    max_iterations:
        Default maximum iterations for an adaptive stage, used directly when
        no schedule is supplied and for omitted adaptive-stage values.

    convergence_patience:
        Default consecutive global-best no-improvement limit for an adaptive
        stage. The counter resets after an improvement and when entering a new
        schedule stage.

    carryover_threshold:
        Logical basis states with ``abs(coefficient) >= carryover_threshold``
        in the global-best eigenvector are carried into the next iteration.

    recovery_fn:
        Recovery module. One of ``recovery_fn``, ``recovery_prob_fn``, or
        ``recovery_schedule`` must be provided. Use
        ``make_mrelu_recovery_fn(...)`` for the built-in modified-ReLU rule.

        expected signature:
            result = recovery_fn(
                all_encoded_bitstrings,
                all_weights,
                labels,
                reference_vectors,
                rng,
            )

        The return value must be ``RecoveryResult`` with valid encoded rows,
        aligned finite nonnegative weights, and mapping-compatible metadata.

    recovery_prob_fn:
        Pair-repair function returning the probability of flipping the first
        rail for each invalid pair. If provided without ``recovery_fn``, it is
        wrapped in the built-in recovery module.

        expected signature:
            p_first = recovery_prob_fn(invalid_pair_bits, invalid_pair_reference)

        invalid_pair_bits.shape = (num_invalid_pairs, 2)
        invalid_pair_bits.dtype = np.uint8
        each row is either [0, 0] or [1, 1]

        invalid_pair_reference.shape = (num_invalid_pairs, 2)
        invalid_pair_reference.dtype = np.float64

        p_first.shape = (num_invalid_pairs,)
        p_first[i] is the probability of flipping the first rail of pair i

    recovery_schedule:
        Optional recovery schedule. If provided, pass neither recovery_fn nor
        recovery_prob_fn. The schedule is a nonempty sequence of dict-like
        stages. A stage can be fixed-count:

            {
                "name": "mrelu_warmup",
                "recovery_fn": make_mrelu_recovery_fn(...),
                "iterations": 3,
            }

        Or adaptive:

            {
                "name": "mrelu_adaptive",
                "recovery_fn": make_mrelu_recovery_fn(...),
                "min_iterations": 3,
                "max_iterations": 20,
                "convergence_patience": 4,
            }

        Adaptive stages use stage-local min/max/patience. If an adaptive stage
        omits any of those fields, the run-level min_iterations,
        max_iterations, or convergence_patience value is used. Explicit
        ``iterations=None`` also selects adaptive behavior. The global best is
        retained across stages, while the no-improvement counter resets.

    max_recovery_draws_per_iteration:
        Maximum number of recovery draws within one self-consistent iteration,
        including the initial draw. If the current recovered candidate pool
        cannot fill ``max_dim``, recovery is repeated on the same original
        weighted encoded samples with fixed labels and references. Draws are
        merged, their duplicate logical-state weights are accumulated, and the
        combined weights are averaged over the number of draws. The loop stops
        early once a full batch can be formed. Set this value to 1 to retain
        the single-draw behavior.

    diagonalize_fn:
        External callable that projects the Hamiltonian and diagonalizes the
        supplied logical basis.

        expected signature:
            diagonalize_fn(
                hamiltonian,
                logical_basis,
                *,
                seed=None,
                warm_start_basis=None,
                warm_start_keys=None,
                warm_start_coefficients=None,
            )
            -> DiagonalizationResult

        logical_basis.shape = (basis_dim, n_logical)
        logical_basis.dtype = np.uint8
        The returned ``logical_basis`` must be a duplicate-free subset of the
        supplied batch. Warm starts pass coefficients with either basis rows or
        prepared keys, never both. A callable may provide
        ``prepare_warm_start(basis, coefficients)`` to generate keys and
        ``spawn(num_threads=...)`` to create a thread-limited worker instance.

    use_global_best_warm_start:
        If true, each batch diagonalization receives the global-best
        wavefunction from the start of the current iteration as warm-start
        data.

    parallelize_batches:
        If True, diagonalize batches within one iteration concurrently.
        If False, batches are evaluated sequentially.

    max_parallel_batches:
        Maximum number of concurrent batch diagonalizations. If None, all
        n_batches may run concurrently when parallelize_batches=True.

    diag_num_threads_per_batch:
        Requested numerical threads per batch. If None while batches are
        parallel, it is ``os.cpu_count() // effective_parallel_batches``.
        Loky applies this as an inner-thread limit; otherwise exact enforcement
        requires ``diagonalize_fn.spawn(num_threads=...)`` support.

    batch_parallel_backend:
        Concurrent execution backend. ``"threading"`` uses an in-process
        ``ThreadPoolExecutor``. ``"loky"`` uses independent
        joblib worker processes and keeps that pool alive across all recovery
        iterations in this call. Loky workers build their batches from shared
        prepared arrays and use worker-scheduling-independent SeedSequence streams keyed
        by (seed, max_dim, iteration, batch index, purpose); this intentionally
        differs from the shared parent-generator trajectory used by threading.
        The Hamiltonian and diagonalizer must be cloudpickle-compatible; worker-
        local mutable state such as ``last_stats`` is reported in batch metadata
        rather than propagated back onto the parent diagonalizer object.

    batch_parallel_max_nbytes:
        Joblib/loky threshold for memory-mapping large input arrays instead of
        copying them into every worker. Accepts None, bytes as an integer, or a
        joblib size string such as ``"1M"``. Ignored by the threading backend.

    batch_parallel_temp_folder:
        Optional directory for joblib/loky memory maps. None uses joblib's
        normal temporary-directory selection (including /dev/shm when usable).
        Ignored by the threading backend.

    record_batch_basis_sha256:
        If True, compute and report a SHA-256 digest for every constructed
        logical batch. This is useful for reproducibility checks but
        adds one full read of each batch, so it defaults to False.

    progress_callback:
        Optional callback called at the end of each iteration with a dictionary
        containing batch energies, batch dimensions, batch diagonalization
        seconds, iteration time, and current global-best information.

    seed:
        Root random seed. Recovery and sequential/threaded batch construction
        use ``np.random.default_rng(seed)``. Loky derives deterministic per-batch
        construction and diagonalization seeds with ``SeedSequence``.

    deterministic_logical_basis:
        Logical basis states included in every batch. These are not encoded
        bitstrings.

    reassign_clusters_each_iteration:
        If True, iterations after the first reassign initial samples to the
        nearest current reference vector. If False, the initial cluster labels
        are reused for every iteration.

    Returns
    -------
    CodeSpaceRecoveryResult
        Best energy, best logical basis, best coefficients, final reference
        vectors, histories, and configuration metadata.

    Algorithm sketch
    ----------------
    1. Validate and flatten clustered encoded samples.
    2. Build initial cluster reference vectors.
    3. In the first iteration, prioritize the initially valid samples: use only
       that pool if it can fill a batch, otherwise force all of its eligible
       rows into every batch. Repair invalid encoded pairs and, when the
       candidate pool is too small,
       repeat recovery up to ``max_recovery_draws_per_iteration`` times. Every
       draw is applied to the original initial weighted samples with the current
       labels and references, not to a previously recovered pool. Merge and
       average the draw results, decode the candidate pool, build batches,
       diagonalize each batch, update the global best, update reference vectors,
       and refresh carry-over states.
    4. Stop according to the recovery schedule or patience-based global-best
       energy criterion. Reference-vector changes are not used as a stopping
       criterion.
    """
    # -------------------------------------------------------------------------
    # 0. Hyperparameter validation
    # -------------------------------------------------------------------------
    n_batches = _validate_positive_int("n_batches", n_batches)
    max_dim = _validate_positive_int("max_dim", max_dim)
    max_recovery_draws_per_iteration = _validate_positive_int(
        "max_recovery_draws_per_iteration",
        max_recovery_draws_per_iteration,
    )
    min_iterations = _validate_nonnegative_int("min_iterations", min_iterations)
    max_iterations = _validate_positive_int("max_iterations", max_iterations)
    convergence_patience = _validate_positive_int(
        "convergence_patience",
        convergence_patience,
    )
    if min_iterations > max_iterations:
        raise ValueError(
            "min_iterations must be <= max_iterations. "
            f"Got min_iterations={min_iterations}, max_iterations={max_iterations}."
        )
    carryover_threshold = _validate_unit_interval_real(
        "carryover_threshold",
        carryover_threshold,
    )
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)

    reassign_clusters_each_iteration = _validate_bool(
        "reassign_clusters_each_iteration",
        reassign_clusters_each_iteration,
    )
    use_global_best_warm_start = _validate_bool(
        "use_global_best_warm_start",
        use_global_best_warm_start,
    )
    parallelize_batches = _validate_bool("parallelize_batches", parallelize_batches)
    max_parallel_batches = _validate_positive_int_or_none(
        "max_parallel_batches",
        max_parallel_batches,
    )
    diag_num_threads_per_batch = _validate_positive_int_or_none(
        "diag_num_threads_per_batch",
        diag_num_threads_per_batch,
    )
    batch_parallel_backend = _validate_batch_parallel_backend(
        batch_parallel_backend
    )
    batch_parallel_max_nbytes = _validate_batch_parallel_max_nbytes(
        batch_parallel_max_nbytes
    )
    batch_parallel_temp_folder = _validate_batch_parallel_temp_folder(
        batch_parallel_temp_folder
    )
    record_batch_basis_sha256 = _validate_bool(
        "record_batch_basis_sha256",
        record_batch_basis_sha256,
    )
    (
        effective_parallel_batches,
        resolved_diag_num_threads_per_batch,
        available_cpu_threads,
    ) = _resolve_batch_parallel_config(
        n_batches=n_batches,
        parallelize_batches=parallelize_batches,
        max_parallel_batches=max_parallel_batches,
        diag_num_threads_per_batch=diag_num_threads_per_batch,
    )
    loky_batch_seed_root: int | None = None
    if parallelize_batches and batch_parallel_backend == "loky":
        loky_batch_seed_root = (
            int(seed)
            if seed is not None
            else int(
                np.random.SeedSequence().generate_state(
                    1,
                    dtype=np.uint64,
                )[0]
            )
        )
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    if not callable(diagonalize_fn):
        raise TypeError("diagonalize_fn must be callable.")
    recovery_stages, recovery_schedule_enabled = _resolve_recovery_stages(
        recovery_fn=recovery_fn,
        recovery_prob_fn=recovery_prob_fn,
        recovery_schedule=recovery_schedule,
        default_min_iterations=min_iterations,
        default_max_iterations=max_iterations,
        default_convergence_patience=convergence_patience,
    )
    fixed_recovery_iterations_total = int(
        sum(stage.iterations or 0 for stage in recovery_stages)
    )
    adaptive_recovery_iterations_max_total = int(
        sum(
            0 if stage.iterations is not None else _recovery_stage_max_iterations(stage)
            for stage in recovery_stages
        )
    )
    total_max_iterations = int(
        sum(_recovery_stage_max_iterations(stage) for stage in recovery_stages)
    )
    final_recovery_stage = recovery_stages[-1]

    if not hasattr(hamiltonian, "num_qubits"):
        raise TypeError("hamiltonian must have a num_qubits attribute.")

    raw_n_logical = hamiltonian.num_qubits
    if isinstance(raw_n_logical, (bool, np.bool_)) or not isinstance(
        raw_n_logical, numbers.Integral
    ):
        raise TypeError(
            "hamiltonian.num_qubits must be an integer, not a value that requires "
            f"coercion; got {raw_n_logical!r}."
        )
    n_logical = int(raw_n_logical)
    if n_logical < 1:
        raise ValueError(f"hamiltonian.num_qubits must be >= 1, got {n_logical}.")
    n_encoded = 2 * n_logical

    # -------------------------------------------------------------------------
    # 1. Sample validation and flattening
    # -------------------------------------------------------------------------
    clustered_samples, all_bits, all_weights, initial_labels = _validate_and_flatten_clustered_samples(
        clustered_samples,
        n_logical=n_logical,
    )
    n_clusters = len(clustered_samples)

    deterministic_logical_basis = _validate_deterministic_logical_basis(
        deterministic_logical_basis,
        n_logical=n_logical,
        max_dim=max_dim,
    )

    # -------------------------------------------------------------------------
    # 2. Initial valid-sample pool
    # -------------------------------------------------------------------------
    # Precompute valid initial configurations for first-iteration batch filling.
    valid_initial_mask = is_valid_encoded_bitstrings(all_bits)
    valid_initial_encoded = all_bits[valid_initial_mask]
    valid_initial_weights = all_weights[valid_initial_mask]

    if len(valid_initial_encoded) > 0:
        valid_initial_logical = decode_valid_encoded_to_logical(valid_initial_encoded)
        valid_initial_logical, valid_initial_logical_weights = _merge_duplicate_rows_sum_weights(
            valid_initial_logical,
            valid_initial_weights,
        )
    else:
        valid_initial_logical = np.empty((0, n_logical), dtype=np.uint8)
        valid_initial_logical_weights = np.empty(0, dtype=np.float64)

    # -------------------------------------------------------------------------
    # 3. Initial reference vectors
    # -------------------------------------------------------------------------
    reference_vectors = initialize_reference_vectors(
        clustered_samples,
        n_logical=n_logical,
    )

    # -------------------------------------------------------------------------
    # 4. State initialization
    # -------------------------------------------------------------------------
    best_energy = float("inf")
    best_logical_basis = np.empty((0, n_logical), dtype=np.uint8)
    best_coefficients = np.empty(0, dtype=np.complex128)

    carryover_logical_basis = np.empty((0, n_logical), dtype=np.uint8)

    reference_history: list[np.ndarray] = []
    iteration_best_energy_history: list[float] = []
    global_best_energy_history: list[float] = []
    global_best_updated_history: list[bool] = []
    batch_energy_history: list[list[float]] = []
    batch_dim_history: list[list[int]] = []
    batch_diag_seconds_history: list[list[float]] = []
    batch_worker_metadata_history: list[list[dict[str, Any]]] = []
    iteration_timing_history: list[dict[str, float]] = []
    recovery_metadata_history: list[dict[str, Any]] = []
    recovery_method_history: list[str] = []
    recovery_stage_history: list[str] = []
    recovery_stage_iteration_history: list[int] = []
    final_stage_iteration_history: list[int | None] = []

    converged = False
    no_global_best_update_count = 0
    stage_completed_iterations = [0 for _stage in recovery_stages]
    stage_stop_reasons: list[str | None] = [None for _stage in recovery_stages]
    recovery_stage_index = 0
    iteration = 0
    stop_reason = "max_iterations"

    loky_batch_pool: _PersistentLokyBatchPool | None = None
    if parallelize_batches and batch_parallel_backend == "loky":
        loky_batch_pool = _PersistentLokyBatchPool(
            n_jobs=effective_parallel_batches,
            diagonalize_fn=diagonalize_fn,
            hamiltonian=hamiltonian,
            num_threads=resolved_diag_num_threads_per_batch,
            max_nbytes=batch_parallel_max_nbytes,
            temp_folder=batch_parallel_temp_folder,
        )
        active_loky_pools = _ACTIVE_LOKY_BATCH_POOLS.get()
        if active_loky_pools is not None:
            active_loky_pools.append(loky_batch_pool)

    # -------------------------------------------------------------------------
    # 5. Main self-consistent iteration loop
    # -------------------------------------------------------------------------
    while iteration < total_max_iterations and recovery_stage_index < len(recovery_stages):
        iteration_start = time.perf_counter()
        current_recovery_stage = recovery_stages[recovery_stage_index]
        recovery_stage_iteration = stage_completed_iterations[recovery_stage_index] + 1
        is_final_recovery_stage = recovery_stage_index == len(recovery_stages) - 1
        final_stage_iteration = (
            recovery_stage_iteration if is_final_recovery_stage else None
        )
        current_recovery_fn = current_recovery_stage.recovery_fn
        recovery_method_history.append(current_recovery_stage.method)
        recovery_stage_history.append(current_recovery_stage.name)
        recovery_stage_iteration_history.append(int(recovery_stage_iteration))
        final_stage_iteration_history.append(
            None if final_stage_iteration is None else int(final_stage_iteration)
        )
        reassignment_start = time.perf_counter()
        stage_setup_seconds = reassignment_start - iteration_start
        # ---------------------------------------------------------------------
        # 5a. Cluster labels for this iteration
        # ---------------------------------------------------------------------
        labels_reassigned = bool(iteration > 0 and reassign_clusters_each_iteration)
        if iteration == 0 or not reassign_clusters_each_iteration:
            # The first iteration always uses the initial cluster labels.
            labels_t = initial_labels
        else:
            # Later iterations may reassign samples to the nearest references.
            labels_t = assign_to_nearest_reference(all_bits, reference_vectors)
        reassignment_seconds = time.perf_counter() - reassignment_start
        recovery_start = time.perf_counter()

        # ---------------------------------------------------------------------
        # 5b. First-iteration valid-sample priority
        # ---------------------------------------------------------------------
        # Additional draws expand only the recovered pool used when the valid
        # initial rows cannot fill a batch.
        if iteration == 0:
            priority_basis = _unique_rows_preserve_order(
                np.vstack([deterministic_logical_basis, carryover_logical_basis])
            )
            remaining_capacity = max(0, max_dim - len(priority_basis))
            eligible_valid_mask = _filter_rows_not_in_set(
                valid_initial_logical,
                priority_basis,
            )
            eligible_valid_logical = valid_initial_logical[eligible_valid_mask]
            eligible_valid_weights = valid_initial_logical_weights[eligible_valid_mask]
            use_valid_initial_pool = len(eligible_valid_logical) >= remaining_capacity
            forced_valid_initial = (
                None if use_valid_initial_pool else eligible_valid_logical
            )
        else:
            eligible_valid_logical = np.empty((0, n_logical), dtype=np.uint8)
            eligible_valid_weights = np.empty(0, dtype=np.float64)
            use_valid_initial_pool = False
            forced_valid_initial = None

        # ---------------------------------------------------------------------
        # 5c. Recover and, if needed, expand the shared logical candidate pool
        # ---------------------------------------------------------------------
        recovered_logical = np.empty((0, n_logical), dtype=np.uint8)
        recovered_weight_sums = np.empty(0, dtype=np.float64)
        recovery_draw_metadata: list[dict[str, Any]] = []
        recovery_draw_unique_counts: list[int] = []
        recovery_new_unique_counts: list[int] = []
        recovery_cumulative_unique_counts: list[int] = []
        recovery_available_dimension_by_draw: list[int] = []
        recovery_augmentation_stop_reason = "max_draws_reached"

        for recovery_draw_index in range(max_recovery_draws_per_iteration):
            recovery_draw_number = recovery_draw_index + 1
            try:
                recovery_result = current_recovery_fn(
                    all_bits,
                    all_weights,
                    labels_t,
                    reference_vectors,
                    rng,
                )
            except Exception as exc:
                raise RuntimeError(
                    "recovery_fn failed at "
                    f"iteration={iteration + 1}, "
                    f"recovery_stage={current_recovery_stage.name!r}, "
                    f"recovery_draw={recovery_draw_number}: {exc}"
                ) from exc

            recovery_result = _validate_recovery_result(
                recovery_result,
                n_encoded=n_encoded,
            )
            draw_metadata = dict(recovery_result.metadata)

            draw_logical = decode_valid_encoded_to_logical(
                recovery_result.bitstrings
            )
            recovery_output_is_unique = bool(
                getattr(
                    current_recovery_fn,
                    "_code_space_recovery_output_rows_are_unique",
                    False,
                )
            )
            if recovery_output_is_unique:
                # The built-in mReLU module already merged encoded duplicates;
                # valid pair-code decoding is bijective, so a second full
                # logical-basis unique/sum pass would be redundant.  Custom
                # recovery callables remain on the defensive merge path unless
                # they explicitly opt in to the same contract.
                draw_logical = draw_logical.astype(np.uint8, copy=True)
                draw_weights = recovery_result.weights.astype(np.float64, copy=True)
                with np.errstate(over="ignore", invalid="ignore"):
                    draw_weight_total = float(np.sum(draw_weights))
                if not math.isfinite(draw_weight_total):
                    raise ValueError(
                        "RecoveryResult weight total must remain finite. A recovery "
                        "callable that advertises unique rows returned an overflowing sum."
                    )
            else:
                draw_logical, draw_weights = _merge_duplicate_rows_sum_weights(
                    draw_logical,
                    recovery_result.weights,
                )
            draw_metadata["driver_duplicate_merge_skipped"] = bool(
                recovery_output_is_unique
            )
            draw_unique_count = int(len(draw_logical))
            previous_cumulative_unique_count = int(len(recovered_logical))

            if recovery_draw_index == 0:
                recovered_logical = draw_logical.astype(np.uint8, copy=True)
                recovered_weight_sums = draw_weights.astype(np.float64, copy=True)
            else:
                recovered_logical, recovered_weight_sums = (
                    _merge_duplicate_rows_sum_weights(
                        np.vstack([recovered_logical, draw_logical]),
                        np.concatenate([recovered_weight_sums, draw_weights]),
                    )
                )

            cumulative_unique_count = int(len(recovered_logical))
            new_unique_count = (
                cumulative_unique_count - previous_cumulative_unique_count
            )

            if use_valid_initial_pool:
                candidate_pool_for_dimension = eligible_valid_logical
            else:
                candidate_pool_for_dimension = recovered_logical

            available_dimension = _maximum_available_batch_dimension(
                candidate_pool_for_dimension,
                deterministic_logical_basis=deterministic_logical_basis,
                carryover_logical_basis=carryover_logical_basis,
                forced_valid_initial_logical_basis=forced_valid_initial,
                max_dim=max_dim,
            )

            draw_metadata["recovery_draw_number"] = int(recovery_draw_number)
            draw_metadata["logical_unique_count"] = draw_unique_count
            recovery_draw_metadata.append(draw_metadata)
            recovery_draw_unique_counts.append(draw_unique_count)
            recovery_new_unique_counts.append(int(new_unique_count))
            recovery_cumulative_unique_counts.append(cumulative_unique_count)
            recovery_available_dimension_by_draw.append(available_dimension)

            if available_dimension >= max_dim:
                recovery_augmentation_stop_reason = (
                    "augmentation_not_needed"
                    if recovery_draw_number == 1
                    else "target_reached"
                )
                break

            if (
                bool(
                    getattr(
                        current_recovery_fn,
                        "_code_space_recovery_no_invalid_pairs_are_terminal",
                        False,
                    )
                )
                and draw_metadata.get("num_invalid_pairs") == 0
            ):
                recovery_augmentation_stop_reason = "no_invalid_pairs"
                break

            if recovery_draw_number >= max_recovery_draws_per_iteration:
                recovery_augmentation_stop_reason = "max_draws_reached"
                break

        recovery_draw_count = int(len(recovery_draw_metadata))
        if recovery_draw_count <= 0:
            raise RuntimeError("Internal error: recovery produced no draws.")
        if not np.all(np.isfinite(recovered_weight_sums)):
            raise RuntimeError(
                "Accumulated recovery weights became non-finite across draws."
            )

        recovered_logical_weights = recovered_weight_sums / float(
            recovery_draw_count
        )

        if use_valid_initial_pool:
            batch_pool_logical = eligible_valid_logical
            batch_pool_weights = eligible_valid_weights
        else:
            batch_pool_logical = recovered_logical
            batch_pool_weights = recovered_logical_weights

        recovery_metadata = dict(recovery_draw_metadata[0])
        recovery_metadata.pop("recovery_draw_number", None)
        recovery_metadata.pop("logical_unique_count", None)
        recovery_metadata["iteration"] = int(iteration + 1)
        recovery_metadata["recovery_stage_index"] = int(recovery_stage_index)
        recovery_metadata["recovery_stage_name"] = current_recovery_stage.name
        recovery_metadata["recovery_stage_mode"] = (
            "adaptive"
            if _recovery_stage_is_adaptive(current_recovery_stage)
            else "fixed"
        )
        recovery_metadata["recovery_method"] = current_recovery_stage.method
        recovery_metadata["recovery_stage_iteration"] = int(recovery_stage_iteration)
        recovery_metadata["recovery_stage_min_iterations"] = (
            None
            if current_recovery_stage.min_iterations is None
            else int(current_recovery_stage.min_iterations)
        )
        recovery_metadata["recovery_stage_max_iterations"] = (
            None
            if current_recovery_stage.max_iterations is None
            else int(current_recovery_stage.max_iterations)
        )
        recovery_metadata["recovery_stage_convergence_patience"] = (
            None
            if current_recovery_stage.convergence_patience is None
            else int(current_recovery_stage.convergence_patience)
        )
        recovery_metadata["final_stage_iteration"] = (
            None if final_stage_iteration is None else int(final_stage_iteration)
        )
        recovery_metadata["is_final_recovery_stage"] = bool(is_final_recovery_stage)
        recovery_metadata["labels_reassigned"] = bool(labels_reassigned)
        recovery_metadata["recovery_draw_count"] = recovery_draw_count
        recovery_metadata["max_recovery_draws_per_iteration"] = int(
            max_recovery_draws_per_iteration
        )
        recovery_metadata["recovery_augmentation_triggered"] = bool(
            recovery_draw_count > 1
        )
        recovery_metadata["recovery_augmentation_stop_reason"] = (
            recovery_augmentation_stop_reason
        )
        recovery_metadata["recovery_draw_unique_counts"] = list(
            recovery_draw_unique_counts
        )
        recovery_metadata["recovery_new_unique_counts"] = list(
            recovery_new_unique_counts
        )
        recovery_metadata["recovery_cumulative_unique_counts"] = list(
            recovery_cumulative_unique_counts
        )
        recovery_metadata["recovery_available_dimension_by_draw"] = list(
            recovery_available_dimension_by_draw
        )
        recovery_metadata["recovery_final_available_dimension"] = int(
            recovery_available_dimension_by_draw[-1]
        )
        recovery_metadata["num_recovered_unique_samples"] = int(
            len(recovered_logical)
        )
        recovery_metadata["combined_recovered_unique_count"] = int(
            len(recovered_logical)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            combined_recovered_weight_sum = float(
                np.sum(recovered_logical_weights)
            )
        if not math.isfinite(combined_recovered_weight_sum):
            raise RuntimeError("Combined recovered weight sum became non-finite.")
        recovery_metadata["combined_recovered_weight_sum"] = (
            combined_recovered_weight_sum
        )
        recovery_metadata["recovery_draw_metadata"] = recovery_draw_metadata
        recovery_metadata_history.append(recovery_metadata)
        recovery_seconds = time.perf_counter() - recovery_start

        # ---------------------------------------------------------------------
        # 5d. Batch construction and diagonalization
        # ---------------------------------------------------------------------
        batch_preparation_start = time.perf_counter()
        batch_preparation_seconds = 0.0
        parallel_wave_seconds = 0.0
        current_batch_energies: list[float] = [float("nan")] * n_batches
        current_batch_dims: list[int] = [0] * n_batches
        current_batch_diag_seconds: list[float] = [float("nan")] * n_batches
        current_batch_worker_metadata: list[dict[str, Any]] = [
            {} for _ in range(n_batches)
        ]

        # Store only the iteration-best basis/coefficient data to keep memory bounded.
        iteration_best_result: DiagonalizationResult | None = None
        iteration_best_energy = float("inf")
        iteration_best_batch_index = n_batches

        warm_start_basis_t: np.ndarray | None = None
        warm_start_keys_t: np.ndarray | None = None
        warm_start_coefficients_t: np.ndarray | None = None
        if use_global_best_warm_start and len(best_logical_basis) > 0:
            prepare_warm_start = getattr(diagonalize_fn, "prepare_warm_start", None)
            if callable(prepare_warm_start):
                warm_start_keys_t, warm_start_coefficients_t = prepare_warm_start(
                    best_logical_basis,
                    best_coefficients,
                )
            else:
                warm_start_basis_t = best_logical_basis.copy()
                warm_start_coefficients_t = best_coefficients.copy()
        batch_preparation_seconds += time.perf_counter() - batch_preparation_start

        def make_batch_spec(
            batch_index: int,
        ) -> tuple[int, np.ndarray, int, float, str | None]:
            batch_build_start = time.perf_counter()
            batch_logical_basis = build_one_batch(
                batch_pool_logical,
                batch_pool_weights,
                deterministic_logical_basis=deterministic_logical_basis,
                carryover_logical_basis=carryover_logical_basis,
                max_dim=max_dim,
                rng=rng,
                forced_valid_initial_logical_basis=forced_valid_initial,
            )
            batch_build_seconds = time.perf_counter() - batch_build_start
            if len(batch_logical_basis) == 0:
                raise RuntimeError(
                    "Constructed an empty batch. Check samples, deterministic basis, and max_dim."
                )
            diag_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            batch_basis_sha256 = (
                _sha256_logical_basis(batch_logical_basis)
                if record_batch_basis_sha256
                else None
            )
            return (
                batch_index,
                batch_logical_basis,
                diag_seed,
                batch_build_seconds,
                batch_basis_sha256,
            )

        def record_batch_result(
            batch_index: int,
            diag_result: DiagonalizationResult,
            diag_seconds: float,
            worker_metadata: dict[str, Any],
        ) -> None:
            nonlocal iteration_best_energy, iteration_best_result
            nonlocal iteration_best_batch_index
            energy_b = float(diag_result.energy)
            current_batch_energies[batch_index] = energy_b
            current_batch_dims[batch_index] = int(diag_result.logical_basis.shape[0])
            current_batch_diag_seconds[batch_index] = float(diag_seconds)
            current_batch_worker_metadata[batch_index] = dict(worker_metadata)
            if energy_b < iteration_best_energy or (
                energy_b == iteration_best_energy
                and batch_index < iteration_best_batch_index
            ):
                iteration_best_energy = energy_b
                iteration_best_result = diag_result
                iteration_best_batch_index = batch_index

        if parallelize_batches and batch_parallel_backend == "loky":
            assert loky_batch_pool is not None
            assert loky_batch_seed_root is not None
            batch_preparation_start = time.perf_counter()
            forced_for_batches, available_pool_for_batches, available_weights_for_batches = (
                _prepare_batch_sampling_inputs(
                    batch_pool_logical,
                    batch_pool_weights,
                    deterministic_logical_basis=deterministic_logical_basis,
                    carryover_logical_basis=carryover_logical_basis,
                    forced_valid_initial_logical_basis=forced_valid_initial,
                )
            )
            batch_specs = [
                (
                    batch_index,
                    *_derive_loky_batch_seeds(
                        loky_batch_seed_root,
                        max_dim,
                        iteration,
                        batch_index,
                    ),
                )
                for batch_index in range(n_batches)
            ]
            batch_preparation_seconds += time.perf_counter() - batch_preparation_start
            parallel_wave_start = time.perf_counter()
            try:
                batch_results = loky_batch_pool.run(
                    batch_specs,
                    forced_logical_basis=forced_for_batches,
                    available_logical_pool=available_pool_for_batches,
                    available_weights=available_weights_for_batches,
                    max_dim=max_dim,
                    record_basis_sha256=record_batch_basis_sha256,
                    warm_start_basis=warm_start_basis_t,
                    warm_start_keys=warm_start_keys_t,
                    warm_start_coefficients=warm_start_coefficients_t,
                )
                for (
                    done_batch_index,
                    diag_result,
                    diag_seconds,
                    worker_metadata,
                ) in batch_results:
                    record_batch_result(
                        done_batch_index,
                        diag_result,
                        diag_seconds,
                        worker_metadata,
                    )
            except Exception as exc:
                raise RuntimeError(
                    "diagonalize_fn failed in loky process backend at "
                    f"iteration={iteration + 1}: {exc}"
                ) from exc
            parallel_wave_seconds += time.perf_counter() - parallel_wave_start
            del (
                batch_results,
                batch_specs,
                diag_result,
                forced_for_batches,
                available_pool_for_batches,
                available_weights_for_batches,
            )
        elif parallelize_batches:
            for wave_start in range(0, n_batches, effective_parallel_batches):
                wave_stop = min(n_batches, wave_start + effective_parallel_batches)
                batch_preparation_start = time.perf_counter()
                batch_specs = [
                    make_batch_spec(batch_index)
                    for batch_index in range(wave_start, wave_stop)
                ]
                batch_preparation_seconds += (
                    time.perf_counter() - batch_preparation_start
                )
                parallel_wave_start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=len(batch_specs)) as executor:
                    futures = [
                        executor.submit(
                            _diagonalize_batch_worker,
                            batch_index,
                            diagonalize_fn,
                            hamiltonian,
                            batch_basis,
                            diag_seed,
                            resolved_diag_num_threads_per_batch,
                            warm_start_basis_t,
                            warm_start_keys_t,
                            warm_start_coefficients_t,
                            batch_build_seconds,
                            batch_basis_sha256,
                            None,
                        )
                        for (
                            batch_index,
                            batch_basis,
                            diag_seed,
                            batch_build_seconds,
                            batch_basis_sha256,
                        ) in batch_specs
                    ]
                    for future, batch_spec in zip(futures, batch_specs):
                        batch_index = batch_spec[0]
                        try:
                            (
                                done_batch_index,
                                diag_result,
                                diag_seconds,
                                worker_metadata,
                            ) = future.result()
                        except Exception as exc:
                            raise RuntimeError(
                                "diagonalize_fn failed at "
                                f"iteration={iteration + 1}, batch={batch_index + 1}: {exc}"
                            ) from exc
                        record_batch_result(
                            done_batch_index,
                            diag_result,
                            diag_seconds,
                            worker_metadata,
                        )
                parallel_wave_seconds += time.perf_counter() - parallel_wave_start
        else:
            for batch_index in range(n_batches):
                batch_preparation_start = time.perf_counter()
                (
                    batch_index,
                    batch_logical_basis,
                    diag_seed,
                    batch_build_seconds,
                    batch_basis_sha256,
                ) = make_batch_spec(batch_index)
                batch_preparation_seconds += (
                    time.perf_counter() - batch_preparation_start
                )
                parallel_wave_start = time.perf_counter()
                try:
                    (
                        done_batch_index,
                        diag_result,
                        diag_seconds,
                        worker_metadata,
                    ) = _diagonalize_batch_worker(
                        batch_index,
                        diagonalize_fn,
                        hamiltonian,
                        batch_logical_basis,
                        diag_seed,
                        resolved_diag_num_threads_per_batch,
                        warm_start_basis_t,
                        warm_start_keys_t,
                        warm_start_coefficients_t,
                        batch_build_seconds,
                        batch_basis_sha256,
                        None,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "diagonalize_fn failed at "
                        f"iteration={iteration + 1}, batch={batch_index + 1}: {exc}"
                    ) from exc
                record_batch_result(
                    done_batch_index,
                    diag_result,
                    diag_seconds,
                    worker_metadata,
                )
                parallel_wave_seconds += time.perf_counter() - parallel_wave_start

        batch_bookkeeping_start = time.perf_counter()
        if iteration_best_result is None:
            raise RuntimeError("No batch diagonalization result was produced.")

        batch_energy_history.append(current_batch_energies)
        batch_dim_history.append(current_batch_dims)
        batch_diag_seconds_history.append(current_batch_diag_seconds)
        batch_worker_metadata_history.append(current_batch_worker_metadata)

        # ---------------------------------------------------------------------
        # 5e. Iteration-best bookkeeping
        # ---------------------------------------------------------------------
        iteration_best_energy_history.append(iteration_best_energy)

        # ---------------------------------------------------------------------
        # 5f. Global-best update
        # ---------------------------------------------------------------------
        improved_global_best = iteration_best_energy < best_energy

        if improved_global_best:
            best_energy = iteration_best_energy
            best_logical_basis = iteration_best_result.logical_basis.copy()
            best_coefficients = iteration_best_result.coefficients.copy()
            no_global_best_update_count = 0
        else:
            no_global_best_update_count += 1

        global_best_energy_history.append(float(best_energy))
        global_best_updated_history.append(bool(improved_global_best))
        batch_bookkeeping_seconds = time.perf_counter() - batch_bookkeeping_start

        progress_assembly_start = time.perf_counter()
        progress_info: dict[str, Any] | None = None
        if progress_callback is not None:
            progress_info = {
                    "event": "iteration_end",
                    "iteration": int(iteration + 1),
                    "batch_energies": [float(x) for x in current_batch_energies],
                    "batch_dims": [int(x) for x in current_batch_dims],
                    "batch_diag_seconds": [float(x) for x in current_batch_diag_seconds],
                    "batch_build_seconds": [
                        float(metadata.get("batch_build_seconds", 0.0))
                        for metadata in current_batch_worker_metadata
                    ],
                    "batch_basis_sha256": [
                        metadata.get("batch_basis_sha256")
                        for metadata in current_batch_worker_metadata
                    ],
                    "batch_worker_metadata": [
                        dict(metadata)
                        for metadata in current_batch_worker_metadata
                    ],
                    "iteration_seconds": 0.0,
                    "iteration_best_energy": float(iteration_best_energy),
                    "iteration_best_batch_index": int(iteration_best_batch_index),
                    "global_best_energy": float(best_energy),
                    "global_best_updated": bool(improved_global_best),
                    "recovery_schedule_enabled": bool(recovery_schedule_enabled),
                    "recovery_stage_index": int(recovery_stage_index),
                    "recovery_stage_name": current_recovery_stage.name,
                    "recovery_stage_mode": (
                        "adaptive"
                        if _recovery_stage_is_adaptive(current_recovery_stage)
                        else "fixed"
                    ),
                    "recovery_method": current_recovery_stage.method,
                    "recovery_draw_count": int(recovery_draw_count),
                    "max_recovery_draws_per_iteration": int(
                        max_recovery_draws_per_iteration
                    ),
                    "recovery_augmentation_triggered": bool(
                        recovery_draw_count > 1
                    ),
                    "recovery_augmentation_stop_reason": (
                        recovery_augmentation_stop_reason
                    ),
                    "recovery_final_available_dimension": int(
                        recovery_available_dimension_by_draw[-1]
                    ),
                    "recovery_stage_iteration": int(recovery_stage_iteration),
                    "recovery_stage_min_iterations": (
                        None
                        if current_recovery_stage.min_iterations is None
                        else int(current_recovery_stage.min_iterations)
                    ),
                    "recovery_stage_max_iterations": (
                        None
                        if current_recovery_stage.max_iterations is None
                        else int(current_recovery_stage.max_iterations)
                    ),
                    "recovery_stage_convergence_patience": (
                        None
                        if current_recovery_stage.convergence_patience is None
                        else int(current_recovery_stage.convergence_patience)
                    ),
                    "recovery_stage_no_global_best_update_count": int(
                        no_global_best_update_count
                    ),
                    "final_stage_iteration": (
                        None
                        if final_stage_iteration is None
                        else int(final_stage_iteration)
                    ),
                    "is_final_recovery_stage": bool(is_final_recovery_stage),
                    "parallelize_batches": bool(parallelize_batches),
                    "batch_parallel_backend": (
                        batch_parallel_backend
                        if parallelize_batches
                        else "sequential"
                    ),
                    "batch_parallel_max_nbytes": batch_parallel_max_nbytes,
                    "batch_parallel_temp_folder": batch_parallel_temp_folder,
                    "batch_parallel_mmap_mode": (
                        "r"
                        if parallelize_batches and batch_parallel_backend == "loky"
                        else None
                    ),
                    "batch_process_pool_persistent": bool(
                        parallelize_batches and batch_parallel_backend == "loky"
                    ),
                    "effective_parallel_batches": int(effective_parallel_batches),
                    "diag_num_threads_per_batch": (
                        None
                        if resolved_diag_num_threads_per_batch is None
                        else int(resolved_diag_num_threads_per_batch)
                    ),
                }
        progress_assembly_seconds = time.perf_counter() - progress_assembly_start

        # ---------------------------------------------------------------------
        # 5g. Reference update from the global-best wavefunction
        # ---------------------------------------------------------------------
        reference_update_start = time.perf_counter()
        new_reference_vectors = update_reference_vectors(
            best_logical_basis,
            best_coefficients,
            reference_vectors,
        )

        reference_history.append(new_reference_vectors.copy())
        reference_update_seconds = time.perf_counter() - reference_update_start

        # ---------------------------------------------------------------------
        # 5h. Carry-over basis update
        # ---------------------------------------------------------------------
        carryover_start = time.perf_counter()
        # Leave room for deterministic rows in every future batch.
        max_carry = max(0, max_dim - len(deterministic_logical_basis))
        carryover_logical_basis = select_carryover_basis(
            best_logical_basis,
            best_coefficients,
            carryover_threshold=carryover_threshold,
            max_keep=max_carry,
        )
        carryover_seconds = time.perf_counter() - carryover_start

        # ---------------------------------------------------------------------
        # 5i. Stage stopping and transitions
        # ---------------------------------------------------------------------
        stopping_start = time.perf_counter()
        stage_completed_iterations[recovery_stage_index] = int(
            recovery_stage_iteration
        )

        advance_to_next_stage = False
        stop_run = False
        current_stage_stop_reason: str | None = None

        if current_recovery_stage.iterations is not None:
            if recovery_stage_iteration >= current_recovery_stage.iterations:
                current_stage_stop_reason = "fixed_iterations_completed"
                if is_final_recovery_stage:
                    stop_reason = current_stage_stop_reason
                    stop_run = True
                else:
                    advance_to_next_stage = True
        else:
            assert current_recovery_stage.min_iterations is not None
            assert current_recovery_stage.max_iterations is not None
            assert current_recovery_stage.convergence_patience is not None
            if recovery_stage_iteration >= current_recovery_stage.max_iterations:
                current_stage_stop_reason = "max_iterations"
                if is_final_recovery_stage:
                    stop_reason = current_stage_stop_reason
                    stop_run = True
                else:
                    advance_to_next_stage = True
            elif (
                recovery_stage_iteration >= current_recovery_stage.min_iterations
                and no_global_best_update_count
                >= current_recovery_stage.convergence_patience
            ):
                current_stage_stop_reason = "no_global_best_improvement"
                if is_final_recovery_stage:
                    converged = True
                    stop_reason = current_stage_stop_reason
                    stop_run = True
                else:
                    advance_to_next_stage = True

        reference_vectors = new_reference_vectors

        if current_stage_stop_reason is not None:
            stage_stop_reasons[recovery_stage_index] = current_stage_stop_reason

        stopping_seconds = time.perf_counter() - stopping_start
        iteration_seconds = time.perf_counter() - iteration_start
        measured_phase_seconds = (
            stage_setup_seconds
            + reassignment_seconds
            + recovery_seconds
            + batch_preparation_seconds
            + parallel_wave_seconds
            + batch_bookkeeping_seconds
            + progress_assembly_seconds
            + reference_update_seconds
            + carryover_seconds
            + stopping_seconds
        )
        iteration_timing = {
            "stage_setup_seconds": float(stage_setup_seconds),
            "reassignment_seconds": float(reassignment_seconds),
            "recovery_seconds": float(recovery_seconds),
            "batch_preparation_seconds": float(batch_preparation_seconds),
            "parallel_wave_seconds": float(parallel_wave_seconds),
            "batch_bookkeeping_seconds": float(batch_bookkeeping_seconds),
            "progress_assembly_seconds": float(progress_assembly_seconds),
            "reference_update_seconds": float(reference_update_seconds),
            "carryover_seconds": float(carryover_seconds),
            "stopping_seconds": float(stopping_seconds),
            "unattributed_seconds": float(
                max(0.0, iteration_seconds - measured_phase_seconds)
            ),
            "iteration_seconds": float(iteration_seconds),
        }
        iteration_timing_history.append(iteration_timing)
        recovery_metadata["iteration_timing_seconds"] = dict(iteration_timing)
        if progress_info is not None:
            progress_info["iteration_seconds"] = float(iteration_seconds)
            progress_info["iteration_timing_seconds"] = dict(iteration_timing)
            progress_callback(progress_info)

        if stop_run:
            break

        if advance_to_next_stage:
            recovery_stage_index += 1
            no_global_best_update_count = 0
            if recovery_stage_index >= len(recovery_stages):
                break

        iteration += 1

    if loky_batch_pool is not None:
        loky_batch_pool.close()

    # -------------------------------------------------------------------------
    # 6. Result assembly
    # -------------------------------------------------------------------------
    recovery_fn_config = (
        None if final_recovery_stage.config is None else dict(final_recovery_stage.config)
    )
    recovery_prob_fn_config = (
        _callable_metadata(recovery_prob_fn) if recovery_prob_fn is not None else None
    )
    if recovery_prob_fn_config is not None:
        recovery_prob_fn_config = dict(recovery_prob_fn_config)

    completed_iterations_total = int(len(iteration_best_energy_history))
    completed_fixed_iterations = int(
        sum(
            count
            for stage, count in zip(recovery_stages, stage_completed_iterations)
            if not _recovery_stage_is_adaptive(stage)
        )
    )
    completed_adaptive_iterations = int(
        sum(
            count
            for stage, count in zip(recovery_stages, stage_completed_iterations)
            if _recovery_stage_is_adaptive(stage)
        )
    )
    completed_final_iterations = int(
        stage_completed_iterations[-1] if stage_completed_iterations else 0
    )
    recovery_schedule_config = [
        _recovery_stage_to_config(stage) for stage in recovery_stages
    ]

    config: dict[str, Any] = {
        "function_name": "run_code_space_recovery",
        "core_name": _CORE_NAME,
        "package_version": PACKAGE_VERSION,
        "module_version": MODULE_VERSION,
        "core_version": MODULE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "n_logical": n_logical,
        "n_encoded": n_encoded,
        "n_clusters": n_clusters,
        "n_batches": n_batches,
        "max_dim": max_dim,
        "max_recovery_draws_per_iteration": int(
            max_recovery_draws_per_iteration
        ),
        "min_iterations": min_iterations,
        "max_iterations": max_iterations,
        "total_max_iterations": int(total_max_iterations),
        "convergence_patience": convergence_patience,
        "no_global_best_update_count": int(no_global_best_update_count),
        "stop_reason": stop_reason,
        "stopping_criterion": (
            "stage_local_global_best_no_improvement_patience"
            if recovery_schedule_enabled
            else "global_best_no_improvement_patience"
        ),
        "uses_reference_vector_convergence": False,
        "carryover_threshold": carryover_threshold,
        "seed": seed,
        "recovery_fn": (
            "recovery_schedule"
            if recovery_schedule_enabled
            else getattr(final_recovery_stage.recovery_fn, "__name__", repr(final_recovery_stage.recovery_fn))
        ),
        "recovery_fn_config": recovery_fn_config,
        "recovery_method": (
            recovery_fn_config.get("method")
            if isinstance(recovery_fn_config, dict)
            else final_recovery_stage.method
        ),
        "recovery_schedule_enabled": bool(recovery_schedule_enabled),
        "recovery_schedule": recovery_schedule_config,
        "fixed_recovery_iterations_total": int(fixed_recovery_iterations_total),
        "adaptive_recovery_iterations_max_total": int(
            adaptive_recovery_iterations_max_total
        ),
        "final_recovery_stage_name": final_recovery_stage.name,
        "final_recovery_stage_method": final_recovery_stage.method,
        "final_stage_min_iterations": (
            None
            if final_recovery_stage.min_iterations is None
            else int(final_recovery_stage.min_iterations)
        ),
        "final_stage_max_iterations": (
            None
            if final_recovery_stage.max_iterations is None
            else int(final_recovery_stage.max_iterations)
        ),
        "final_stage_convergence_patience": (
            None
            if final_recovery_stage.convergence_patience is None
            else int(final_recovery_stage.convergence_patience)
        ),
        "recovery_stage_completed_iterations": [
            int(x) for x in stage_completed_iterations
        ],
        "recovery_stage_stop_reasons": list(stage_stop_reasons),
        "completed_fixed_iterations": completed_fixed_iterations,
        "completed_adaptive_iterations": completed_adaptive_iterations,
        "completed_final_iterations": completed_final_iterations,
        "reassign_clusters_each_iteration": bool(reassign_clusters_each_iteration),
        "use_global_best_warm_start": bool(use_global_best_warm_start),
        "parallelize_batches": bool(parallelize_batches),
        "batch_parallel_backend": (
            batch_parallel_backend if parallelize_batches else "sequential"
        ),
        "batch_parallel_backend_requested": batch_parallel_backend,
        "batch_parallel_max_nbytes": batch_parallel_max_nbytes,
        "batch_parallel_temp_folder": batch_parallel_temp_folder,
        "record_batch_basis_sha256": bool(record_batch_basis_sha256),
        "batch_rng_scheme": (
            "seedsequence_v1(root_seed,max_dim,iteration,batch_index,purpose)"
            if parallelize_batches and batch_parallel_backend == "loky"
            else "legacy_shared_parent_generator"
        ),
        "loky_batch_seed_root": (
            None if loky_batch_seed_root is None else int(loky_batch_seed_root)
        ),
        "batch_parallel_mmap_mode": (
            "r"
            if parallelize_batches and batch_parallel_backend == "loky"
            else None
        ),
        "batch_process_pool_persistent": bool(
            parallelize_batches and batch_parallel_backend == "loky"
        ),
        "max_parallel_batches": (
            None if max_parallel_batches is None else int(max_parallel_batches)
        ),
        "effective_parallel_batches": int(effective_parallel_batches),
        "diag_num_threads_per_batch": (
            None
            if resolved_diag_num_threads_per_batch is None
            else int(resolved_diag_num_threads_per_batch)
        ),
        "available_cpu_threads": int(available_cpu_threads),
        "recovery_prob_fn": (
            getattr(recovery_prob_fn, "__name__", repr(recovery_prob_fn))
            if recovery_prob_fn is not None
            else None
        ),
        "recovery_prob_fn_config": recovery_prob_fn_config,
        "diagonalize_fn": getattr(diagonalize_fn, "__name__", repr(diagonalize_fn)),
        "has_progress_callback": progress_callback is not None,
        "deterministic_basis_count": int(len(deterministic_logical_basis)),
        "valid_initial_unique_count": int(len(valid_initial_logical)),
        "completed_iterations": completed_iterations_total,
        "iteration_timing_fields": (
            list(iteration_timing_history[0]) if iteration_timing_history else []
        ),
        "iteration_timing_history": [
            dict(timing) for timing in iteration_timing_history
        ],
    }

    if recovery_fn_config is not None:
        for key in (
            "type",
            "family",
            "method",
            "delta",
            "corner",
        ):
            if key in recovery_fn_config:
                config[f"recovery_fn_{key}"] = recovery_fn_config[key]

    if recovery_prob_fn_config is not None:
        for key in ("type", "family", "delta", "corner"):
            if key in recovery_prob_fn_config:
                config[f"recovery_prob_fn_{key}"] = recovery_prob_fn_config[key]

    return CodeSpaceRecoveryResult(
        best_energy=float(best_energy),
        best_logical_basis=best_logical_basis.astype(np.uint8, copy=True),
        best_coefficients=best_coefficients.astype(np.complex128, copy=True),
        final_reference_vectors=reference_vectors.astype(np.float64, copy=True),
        reference_history=reference_history,
        iteration_best_energy_history=iteration_best_energy_history,
        global_best_energy_history=global_best_energy_history,
        global_best_updated_history=global_best_updated_history,
        batch_energy_history=batch_energy_history,
        batch_dim_history=batch_dim_history,
        batch_diag_seconds_history=batch_diag_seconds_history,
        batch_worker_metadata_history=batch_worker_metadata_history,
        recovery_metadata_history=recovery_metadata_history,
        recovery_method_history=recovery_method_history,
        recovery_stage_history=recovery_stage_history,
        recovery_stage_iteration_history=recovery_stage_iteration_history,
        final_stage_iteration_history=final_stage_iteration_history,
        converged=bool(converged),
        config=config,
        iteration_timing_history=iteration_timing_history,
    )


# =============================================================================
# Modified-ReLU recovery-probability factory
# =============================================================================


def _validate_relu_corner(corner: Any) -> float:
    """Validate the modified-ReLU corner parameter."""
    corner = _validate_open_unit_interval_real("corner", corner)
    return corner


def modified_relu_score(
    distance: np.ndarray,
    *,
    delta: float = 0.01,
    corner: float = 0.5,
) -> np.ndarray:
    """Compute the modified-ReLU score used for invalid-pair repair.

    The score is nondecreasing and piecewise linear in the distance.

    Parameters
    ----------
    distance:
        Distance between the bit value and the reference value.

    delta:
        Score at ``distance=corner``. The low-distance slope is
        ``delta / corner``. Must satisfy ``0 < delta <= 1``.

    corner:
        Kink location of the piecewise-linear score. Must lie in ``(0, 1)``.
    """
    distance = _as_finite_real_array("distance", distance)
    delta = _validate_positive_real("delta", delta)
    if delta > 1.0:
        raise ValueError(f"delta must be <= 1, got {delta}.")
    corner = _validate_relu_corner(corner)
    return np.where(
        distance <= corner,
        delta * distance / corner,
        delta + (1.0 - delta) * (distance - corner) / (1.0 - corner),
    )


def make_relu_pair_recovery_prob_fn(
    *,
    delta: float = 0.01,
    corner: float = 0.5,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create a modified-ReLU pair repair probability function.

    The returned callable can be passed as ``recovery_prob_fn`` and carries
    metadata used in the final run configuration.

    Parameters
    ----------
    delta:
        Score at ``distance=corner``; the low-distance slope is
        ``delta / corner``. Must satisfy ``0 < delta <= 1``.

    corner:
        Kink location ``h`` of the modified ReLU.

    Returns
    -------
    recovery_prob_fn:
        Callable compatible with
        ``run_code_space_recovery(recovery_prob_fn=...)``.
    """
    delta = _validate_positive_real("delta", delta)
    if delta > 1.0:
        raise ValueError(f"delta must be <= 1, got {delta}.")
    corner = _validate_relu_corner(corner)

    def recovery_prob_fn(
        invalid_pair_bits: np.ndarray,
        invalid_pair_reference: np.ndarray,
    ) -> np.ndarray:
        """Return first-rail flip probabilities for invalid encoded pairs.

        ``p_first[i]`` is the probability of flipping the first rail of pair i.
        """
        invalid_pair_bits_arr = _as_exact_binary_uint8(
            "invalid_pair_bits",
            invalid_pair_bits,
        )
        if invalid_pair_bits_arr.shape[1] != 2:
            raise ValueError("invalid_pair_bits must have shape (num_invalid_pairs, 2).")
        invalid_pair_reference_arr = _as_reference_vectors(
            "invalid_pair_reference",
            invalid_pair_reference,
            require_normalized=True,
        )
        if invalid_pair_reference_arr.shape != invalid_pair_bits_arr.shape:
            raise ValueError(
                "invalid_pair_reference must have the same shape as invalid_pair_bits."
            )
        if np.any(invalid_pair_bits_arr[:, 0] != invalid_pair_bits_arr[:, 1]):
            raise ValueError("invalid_pair_bits rows must be invalid pairs 00 or 11.")

        distances = np.abs(
            invalid_pair_bits_arr.astype(np.float64) - invalid_pair_reference_arr
        )
        scores = modified_relu_score(distances, delta=delta, corner=corner)
        denom = scores[:, 0] + scores[:, 1]

        if np.any(denom <= 0.0):
            bad_count = int(np.count_nonzero(denom <= 0.0))
            first_bad = int(np.flatnonzero(denom <= 0.0)[0])
            raise RuntimeError(
                "modified-ReLU recovery denominator was zero. "
                "This indicates an invalid recovery score configuration or reference state. "
                f"bad_pairs={bad_count}, first_bad_index={first_bad}, "
                f"invalid_pair={invalid_pair_bits_arr[first_bad].tolist()}, "
                f"reference_pair={invalid_pair_reference_arr[first_bad].tolist()}."
            )
        p_first = scores[:, 0] / denom
        return np.clip(p_first, 0.0, 1.0)

    recovery_prob_fn.__name__ = (
        f"relu_pair_recovery_prob_fn_delta_{delta:g}_corner_{corner:g}"
    )
    recovery_metadata = {
        "type": "modified_relu_pair_recovery_prob_fn",
        "family": "modified_relu_pair_recovery",
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "delta": float(delta),
        "corner": float(corner),
    }
    recovery_prob_fn.delta = float(delta)  # type: ignore[attr-defined]
    recovery_prob_fn.corner = float(corner)  # type: ignore[attr-defined]
    recovery_prob_fn.config = recovery_metadata  # type: ignore[attr-defined]
    recovery_prob_fn._code_space_recovery_metadata = recovery_metadata  # type: ignore[attr-defined]
    recovery_prob_fn._code_space_recovery_config = recovery_metadata  # type: ignore[attr-defined]
    return recovery_prob_fn


def make_mrelu_recovery_fn(
    *,
    recovery_prob_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    delta: float = 0.01,
    corner: float = 0.5,
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.random.Generator],
    RecoveryResult,
]:
    """Create a modified-ReLU recovery callable.

    The returned callable repairs only invalid pairs, merges duplicate recovered
    rows, and returns ``RecoveryResult`` metadata. If ``recovery_prob_fn`` is
    omitted, ``delta`` and ``corner`` configure the built-in pair probability
    rule; otherwise that callable determines the repair probabilities.
    """
    if recovery_prob_fn is None:
        recovery_prob_fn = make_relu_pair_recovery_prob_fn(delta=delta, corner=corner)
    elif not callable(recovery_prob_fn):
        raise TypeError("recovery_prob_fn must be callable.")

    prob_metadata = _callable_metadata(recovery_prob_fn) or {}
    metadata: dict[str, Any] = {
        "type": "mrelu_recovery_fn",
        "family": "modified_relu_pair_recovery",
        "method": "mrelu",
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "recovery_prob_fn": getattr(recovery_prob_fn, "__name__", repr(recovery_prob_fn)),
        "recovery_prob_fn_config": dict(prob_metadata),
    }
    for key in ("delta", "corner"):
        if key in prob_metadata:
            metadata[key] = prob_metadata[key]

    def recovery_fn(
        all_encoded_bitstrings: np.ndarray,
        all_weights: np.ndarray,
        labels: np.ndarray,
        reference_vectors: np.ndarray,
        rng: np.random.Generator,
    ) -> RecoveryResult:
        return mrelu_recover_encoded_samples(
            all_encoded_bitstrings,
            all_weights,
            labels,
            reference_vectors,
            recovery_prob_fn,
            rng,
        )

    recovery_fn.__name__ = "mrelu_recovery_fn"
    recovery_fn.config = metadata  # type: ignore[attr-defined]
    recovery_fn._code_space_recovery_metadata = metadata  # type: ignore[attr-defined]
    recovery_fn._code_space_recovery_config = metadata  # type: ignore[attr-defined]
    recovery_fn._code_space_recovery_no_invalid_pairs_are_terminal = True  # type: ignore[attr-defined]
    recovery_fn._code_space_recovery_output_rows_are_unique = True  # type: ignore[attr-defined]
    return recovery_fn


__all__ = [
    "DiagonalizationResult",
    "CodeSpaceRecoveryResult",
    "RecoveryResult",
    "SparseInvalidPairs",
    "encode_logical_to_valid_encoded",
    "is_valid_encoded_bitstrings",
    "decode_valid_encoded_to_logical",
    "pairwise_normalize_reference_vectors",
    "initialize_reference_vectors",
    "assign_to_nearest_reference",
    "build_sparse_invalid_pairs",
    "mrelu_recover_encoded_samples",
    "build_one_batch",
    "select_carryover_basis",
    "update_reference_vectors",
    "run_code_space_recovery",
    "modified_relu_score",
    "make_relu_pair_recovery_prob_fn",
    "make_mrelu_recovery_fn",
]
