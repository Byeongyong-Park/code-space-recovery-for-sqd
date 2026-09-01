"""PRIMME-only projected Pauli-sum diagonalizer for code-space recovery.

Use case
--------
- logical qubits: up to 36 by default; uint64 kernels can support up to 63
- projected dimension: limited by the configured CSR-memory cap and available memory
- Hamiltonian: linear combination of Pauli operators on logical qubits
- eigensolver: PRIMME eigsh only; SciPy supplies the LinearOperator interface

Internal convention
-------------------
By default, logical_basis rows follow the code-space recovery displayed-bit
convention:
logical_basis[:, 0] is the leftmost displayed logical bit and maps to uint64 bit
num_qubits-1. Thus [0, 0, 1] is packed as integer 1. This matches Qiskit Pauli
labels where the rightmost label character is qubit 0.

Set logical_basis_bit_order="little_endian" only if logical_basis[:, q] should map
directly to uint64 bit q. For Qiskit SparsePauliOp labels, the default
pauli_label_convention="qiskit" should be kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math
import numbers
import time
import sys

import numpy as np

try:  # package import
    from ._version import ALGORITHM_VERSION, PACKAGE_VERSION
except ImportError:  # pragma: no cover - flat-module compatibility
    from _version import ALGORITHM_VERSION, PACKAGE_VERSION  # type: ignore


def _patch_coverage_for_numba_import() -> None:
    """Work around a coverage/numba import incompatibility in some environments."""
    try:
        import coverage  # type: ignore
    except Exception:
        return
    coverage_types = getattr(coverage, "types", None)
    if coverage_types is None:
        return
    if not hasattr(coverage_types, "Tracer"):
        setattr(coverage_types, "Tracer", type("Tracer", (), {}))
    if not hasattr(coverage_types, "TShouldTraceFn"):
        setattr(coverage_types, "TShouldTraceFn", object)
    if not hasattr(coverage_types, "TShouldStartContextFn"):
        setattr(coverage_types, "TShouldStartContextFn", object)


try:
    try:
        import numba as nb
    except Exception:
        # A failed numba import can leave partially initialized numba modules in
        # sys.modules. Remove them before retrying with the coverage compatibility
        # patch above.
        for _name in list(sys.modules):
            if _name == "numba" or _name.startswith("numba."):
                del sys.modules[_name]
        _patch_coverage_for_numba_import()
        import numba as nb
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "projected_pauli_primme_diagonalizer requires numba. Install it with `pip install numba`."
    ) from exc

try:
    from scipy.sparse.linalg import LinearOperator
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "projected_pauli_primme_diagonalizer requires scipy.sparse.linalg.LinearOperator. "
        "SciPy is used only as the operator interface; PRIMME is the only eigensolver."
    ) from exc

try:
    try:  # package import
        from .recovery import DiagonalizationResult
    except Exception:  # pragma: no cover - flat-module import fallback
        from recovery import DiagonalizationResult  # type: ignore
except Exception:  # pragma: no cover
    @dataclass
    class DiagonalizationResult:  # type: ignore[no-redef]
        energy: float
        logical_basis: np.ndarray
        coefficients: np.ndarray


EMPTY_U64 = np.uint64(0xFFFFFFFFFFFFFFFF)
MISSING_I32 = np.int32(-1)
DEFAULT_MAX_LOGICAL_QUBITS = 36


@dataclass(frozen=True)
class PauliTerm:
    """Explicit Pauli term accepted by the parser."""

    pauli: Any
    coeff: complex


@dataclass(frozen=True)
class CompiledPauliHamiltonian:
    """Grouped bit-mask representation of a Pauli-sum Hamiltonian."""

    num_qubits: int
    group_x_masks: np.ndarray          # uint64, shape (n_groups,)
    group_offsets: np.ndarray          # int64, shape (n_groups + 1,)
    term_z_masks: np.ndarray           # uint64, shape (n_terms,)
    term_coeffs_complex: np.ndarray    # complex128, includes i**n_y prefactor
    term_coeffs_real: np.ndarray       # float64, used if projected matrix is real
    is_real: bool
    n_input_terms: int
    n_compiled_terms: int
    coefficient_cutoff: float
    pauli_label_convention: str

    @property
    def n_groups(self) -> int:
        return int(self.group_x_masks.shape[0])


@dataclass(frozen=True)
class ProjectedCSR:
    """Projected Hamiltonian in CSR-array form in input-basis row order."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    is_real: bool
    nnz: int
    memory_bytes: int


@dataclass
class ProjectedPauliBuildStats:
    """Timing, size, and residual diagnostics for projected CSR builds and PRIMME solves."""

    basis_dim: int
    num_qubits: int
    n_groups: int
    n_terms: int
    nnz: int
    csr_memory_gib: float
    pack_seconds: float = 0.0
    hash_seconds: float = 0.0
    count_seconds: float = 0.0
    fill_seconds: float = 0.0
    solve_seconds: float = 0.0
    residual_norm: float | None = None
    relative_residual: float | None = None
    used_warm_start: bool = False
    primme_stats: dict[str, Any] | None = None
    package_version: str = PACKAGE_VERSION
    algorithm_version: str = ALGORITHM_VERSION


# =============================================================================
# Numba kernels
# =============================================================================


@nb.njit(cache=True, inline="always")
def _splitmix64(x: np.uint64) -> np.uint64:
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@nb.njit(cache=True, inline="always")
def _parity_u64(x: np.uint64) -> np.int64:
    p = np.int64(0)
    while x != np.uint64(0):
        p ^= np.int64(1)
        x &= x - np.uint64(1)
    return p


@nb.njit(cache=True, parallel=True)
def _pack_logical_basis_uint64_kernel(basis: np.ndarray) -> np.ndarray:
    n_rows, n_qubits = basis.shape
    keys = np.empty(n_rows, dtype=np.uint64)
    for i in nb.prange(n_rows):
        key = np.uint64(0)
        for q in range(n_qubits):
            if basis[i, q] != 0:
                key |= np.uint64(1) << np.uint64(q)
        keys[i] = key
    return keys


@nb.njit(cache=True, parallel=True)
def _pack_logical_basis_uint64_big_endian_kernel(basis: np.ndarray) -> np.ndarray:
    """Pack rows where basis[:, 0] is the most-significant logical bit.

    This matches the code-space recovery convention used in examples such as
    logical row [0, 0, 1] representing the displayed bitstring "001",
    whose integer key should be 1.
    """
    n_rows, n_qubits = basis.shape
    keys = np.empty(n_rows, dtype=np.uint64)
    for i in nb.prange(n_rows):
        key = np.uint64(0)
        for col in range(n_qubits):
            if basis[i, col] != 0:
                bit_index = n_qubits - 1 - col
                key |= np.uint64(1) << np.uint64(bit_index)
        keys[i] = key
    return keys


@nb.njit(cache=True)
def _build_hash_table_uint64_i32_kernel(
    keys: np.ndarray,
    hash_keys: np.ndarray,
    hash_vals: np.ndarray,
) -> np.int64:
    mask = np.uint64(hash_keys.shape[0] - 1)
    for i in range(hash_keys.shape[0]):
        hash_keys[i] = EMPTY_U64
        hash_vals[i] = MISSING_I32

    for i in range(keys.shape[0]):
        key = keys[i]
        slot = _splitmix64(key) & mask
        while True:
            existing = hash_keys[slot]
            if existing == EMPTY_U64:
                hash_keys[slot] = key
                hash_vals[slot] = np.int32(i)
                break
            if existing == key:
                return np.int64(i)
            slot = (slot + np.uint64(1)) & mask
    return np.int64(-1)


@nb.njit(cache=True, inline="always")
def _lookup_uint64_i32(key: np.uint64, hash_keys: np.ndarray, hash_vals: np.ndarray) -> np.int32:
    mask = np.uint64(hash_keys.shape[0] - 1)
    slot = _splitmix64(key) & mask
    while True:
        existing = hash_keys[slot]
        if existing == EMPTY_U64:
            return MISSING_I32
        if existing == key:
            return hash_vals[slot]
        slot = (slot + np.uint64(1)) & mask


@nb.njit(cache=True, inline="always")
def _eval_group_complex(
    col_key: np.uint64,
    start: np.int64,
    stop: np.int64,
    term_z_masks: np.ndarray,
    term_coeffs: np.ndarray,
) -> np.complex128:
    acc = np.complex128(0.0 + 0.0j)
    for t in range(start, stop):
        coeff = term_coeffs[t]
        if _parity_u64(term_z_masks[t] & col_key) == 1:
            acc -= coeff
        else:
            acc += coeff
    return acc


@nb.njit(cache=True, inline="always")
def _eval_group_real(
    col_key: np.uint64,
    start: np.int64,
    stop: np.int64,
    term_z_masks: np.ndarray,
    term_coeffs: np.ndarray,
) -> float:
    acc = 0.0
    for t in range(start, stop):
        coeff = term_coeffs[t]
        if _parity_u64(term_z_masks[t] & col_key) == 1:
            acc -= coeff
        else:
            acc += coeff
    return acc


@nb.njit(cache=True, parallel=True)
def _count_projected_rows_complex(
    keys: np.ndarray,
    hash_keys: np.ndarray,
    hash_vals: np.ndarray,
    group_x_masks: np.ndarray,
    group_offsets: np.ndarray,
    term_z_masks: np.ndarray,
    term_coeffs: np.ndarray,
    matrix_element_cutoff: float,
) -> np.ndarray:
    n_rows = keys.shape[0]
    n_groups = group_x_masks.shape[0]
    row_counts = np.zeros(n_rows, dtype=np.int64)
    cutoff2 = matrix_element_cutoff * matrix_element_cutoff
    for row in nb.prange(n_rows):
        row_key = keys[row]
        count = 0
        for g in range(n_groups):
            col_key = row_key ^ group_x_masks[g]
            col = _lookup_uint64_i32(col_key, hash_keys, hash_vals)
            if col >= 0:
                value = _eval_group_complex(
                    col_key, group_offsets[g], group_offsets[g + 1], term_z_masks, term_coeffs
                )
                mag2 = value.real * value.real + value.imag * value.imag
                if mag2 > cutoff2:
                    count += 1
        row_counts[row] = count
    return row_counts


@nb.njit(cache=True, parallel=True)
def _count_projected_rows_real(
    keys: np.ndarray,
    hash_keys: np.ndarray,
    hash_vals: np.ndarray,
    group_x_masks: np.ndarray,
    group_offsets: np.ndarray,
    term_z_masks: np.ndarray,
    term_coeffs: np.ndarray,
    matrix_element_cutoff: float,
) -> np.ndarray:
    n_rows = keys.shape[0]
    n_groups = group_x_masks.shape[0]
    row_counts = np.zeros(n_rows, dtype=np.int64)
    for row in nb.prange(n_rows):
        row_key = keys[row]
        count = 0
        for g in range(n_groups):
            col_key = row_key ^ group_x_masks[g]
            col = _lookup_uint64_i32(col_key, hash_keys, hash_vals)
            if col >= 0:
                value = _eval_group_real(
                    col_key, group_offsets[g], group_offsets[g + 1], term_z_masks, term_coeffs
                )
                if abs(value) > matrix_element_cutoff:
                    count += 1
        row_counts[row] = count
    return row_counts


@nb.njit(cache=True, parallel=True)
def _fill_projected_csr_complex(
    keys: np.ndarray,
    hash_keys: np.ndarray,
    hash_vals: np.ndarray,
    group_x_masks: np.ndarray,
    group_offsets: np.ndarray,
    term_z_masks: np.ndarray,
    term_coeffs: np.ndarray,
    matrix_element_cutoff: float,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
) -> None:
    n_rows = keys.shape[0]
    n_groups = group_x_masks.shape[0]
    cutoff2 = matrix_element_cutoff * matrix_element_cutoff
    for row in nb.prange(n_rows):
        row_key = keys[row]
        cursor = indptr[row]
        for g in range(n_groups):
            col_key = row_key ^ group_x_masks[g]
            col = _lookup_uint64_i32(col_key, hash_keys, hash_vals)
            if col >= 0:
                value = _eval_group_complex(
                    col_key, group_offsets[g], group_offsets[g + 1], term_z_masks, term_coeffs
                )
                mag2 = value.real * value.real + value.imag * value.imag
                if mag2 > cutoff2:
                    indices[cursor] = col
                    data[cursor] = value
                    cursor += 1


@nb.njit(cache=True, parallel=True)
def _fill_projected_csr_real(
    keys: np.ndarray,
    hash_keys: np.ndarray,
    hash_vals: np.ndarray,
    group_x_masks: np.ndarray,
    group_offsets: np.ndarray,
    term_z_masks: np.ndarray,
    term_coeffs: np.ndarray,
    matrix_element_cutoff: float,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
) -> None:
    n_rows = keys.shape[0]
    n_groups = group_x_masks.shape[0]
    for row in nb.prange(n_rows):
        row_key = keys[row]
        cursor = indptr[row]
        for g in range(n_groups):
            col_key = row_key ^ group_x_masks[g]
            col = _lookup_uint64_i32(col_key, hash_keys, hash_vals)
            if col >= 0:
                value = _eval_group_real(
                    col_key, group_offsets[g], group_offsets[g + 1], term_z_masks, term_coeffs
                )
                if abs(value) > matrix_element_cutoff:
                    indices[cursor] = col
                    data[cursor] = value
                    cursor += 1


@nb.njit(cache=True, parallel=True)
def _csr_matvec_real(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, x: np.ndarray) -> np.ndarray:
    n_rows = indptr.shape[0] - 1
    y = np.zeros(n_rows, dtype=np.float64)
    for row in nb.prange(n_rows):
        acc = 0.0
        for p in range(indptr[row], indptr[row + 1]):
            acc += data[p] * x[indices[p]]
        y[row] = acc
    return y


@nb.njit(cache=True, parallel=True)
def _csr_matvec_complex(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, x: np.ndarray) -> np.ndarray:
    n_rows = indptr.shape[0] - 1
    y = np.zeros(n_rows, dtype=np.complex128)
    for row in nb.prange(n_rows):
        acc = np.complex128(0.0 + 0.0j)
        for p in range(indptr[row], indptr[row + 1]):
            acc += data[p] * x[indices[p]]
        y[row] = acc
    return y


@nb.njit(cache=True, parallel=True)
def _csr_matmat_real(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, X: np.ndarray) -> np.ndarray:
    n_rows = indptr.shape[0] - 1
    n_cols = X.shape[1]
    Y = np.zeros((n_rows, n_cols), dtype=np.float64)
    for row in nb.prange(n_rows):
        for p in range(indptr[row], indptr[row + 1]):
            col = indices[p]
            value = data[p]
            for j in range(n_cols):
                Y[row, j] += value * X[col, j]
    return Y


@nb.njit(cache=True, parallel=True)
def _csr_matmat_complex(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, X: np.ndarray) -> np.ndarray:
    n_rows = indptr.shape[0] - 1
    n_cols = X.shape[1]
    Y = np.zeros((n_rows, n_cols), dtype=np.complex128)
    for row in nb.prange(n_rows):
        for p in range(indptr[row], indptr[row + 1]):
            col = indices[p]
            value = data[p]
            for j in range(n_cols):
                Y[row, j] += value * X[col, j]
    return Y


@nb.njit(cache=True)
def _fill_warm_start_real(
    old_keys: np.ndarray,
    old_coeffs: np.ndarray,
    hash_keys: np.ndarray,
    hash_vals: np.ndarray,
    out: np.ndarray,
) -> None:
    for i in range(old_keys.shape[0]):
        idx = _lookup_uint64_i32(old_keys[i], hash_keys, hash_vals)
        if idx >= 0:
            out[idx] = old_coeffs[i].real


@nb.njit(cache=True)
def _fill_warm_start_complex(
    old_keys: np.ndarray,
    old_coeffs: np.ndarray,
    hash_keys: np.ndarray,
    hash_vals: np.ndarray,
    out: np.ndarray,
) -> None:
    for i in range(old_keys.shape[0]):
        idx = _lookup_uint64_i32(old_keys[i], hash_keys, hash_vals)
        if idx >= 0:
            out[idx] = old_coeffs[i]


# =============================================================================
# Python helpers: validation, parsing, compilation
# =============================================================================


def _as_positive_int_or_none(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a positive integer or None, got {type(value).__name__}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _as_nonnegative_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a nonnegative real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value}.")
    return value


def _as_positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}.")
    return value


def _validate_compiled_pauli_hamiltonian(
    compiled: CompiledPauliHamiltonian,
) -> CompiledPauliHamiltonian:
    """Validate the complete structural contract of a compiled Hamiltonian.

    ``CompiledPauliHamiltonian`` is public and can therefore be constructed or
    modified independently of :func:`compile_pauli_hamiltonian`. Numerical
    kernels assume the canonical grouping, dtypes, cached real coefficients,
    and uint64 mask range produced by the compiler. Validate those assumptions
    before a caller-supplied compiled object reaches a kernel.

    The validated object is returned unchanged so normal compiler output keeps
    the same arrays and numerical path.
    """
    if not isinstance(compiled, CompiledPauliHamiltonian):
        raise TypeError("compiled must be a CompiledPauliHamiltonian instance.")

    if isinstance(compiled.num_qubits, (bool, np.bool_)) or not isinstance(
        compiled.num_qubits,
        numbers.Integral,
    ):
        raise TypeError("Compiled Hamiltonian num_qubits must be an integer, not bool.")
    num_qubits = int(compiled.num_qubits)
    if num_qubits < 1 or num_qubits > 63:
        raise ValueError(
            "Compiled Hamiltonian num_qubits must be in [1, 63], "
            f"got {num_qubits}."
        )

    if isinstance(compiled.n_input_terms, (bool, np.bool_)) or not isinstance(
        compiled.n_input_terms,
        numbers.Integral,
    ):
        raise TypeError("Compiled Hamiltonian n_input_terms must be an integer, not bool.")
    if isinstance(compiled.n_compiled_terms, (bool, np.bool_)) or not isinstance(
        compiled.n_compiled_terms,
        numbers.Integral,
    ):
        raise TypeError(
            "Compiled Hamiltonian n_compiled_terms must be an integer, not bool."
        )
    n_input_terms = int(compiled.n_input_terms)
    declared_n_terms = int(compiled.n_compiled_terms)
    if n_input_terms < 0 or declared_n_terms < 0:
        raise ValueError("Compiled Hamiltonian term counts must be nonnegative.")
    if declared_n_terms > n_input_terms:
        raise ValueError(
            "Compiled Hamiltonian n_compiled_terms cannot exceed n_input_terms."
        )

    coefficient_cutoff = _as_nonnegative_float(
        "Compiled Hamiltonian coefficient_cutoff",
        compiled.coefficient_cutoff,
    )
    if not isinstance(compiled.pauli_label_convention, str):
        raise TypeError("Compiled Hamiltonian pauli_label_convention must be a string.")
    convention = compiled.pauli_label_convention.lower()
    if convention not in {
        "qiskit",
        "big_endian",
        "big-endian",
        "msb",
        "little",
        "little_endian",
        "little-endian",
        "lsb",
        "internal",
    }:
        raise ValueError(
            "Compiled Hamiltonian pauli_label_convention must be "
            "'qiskit' or 'little_endian'."
        )
    if not isinstance(compiled.is_real, bool):
        raise TypeError("Compiled Hamiltonian is_real must be bool.")

    raw_named_arrays = (
        ("group_x_masks", compiled.group_x_masks),
        ("group_offsets", compiled.group_offsets),
        ("term_z_masks", compiled.term_z_masks),
        ("term_coeffs_complex", compiled.term_coeffs_complex),
        ("term_coeffs_real", compiled.term_coeffs_real),
    )
    for name, array in raw_named_arrays:
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"Compiled Hamiltonian {name} must be a numpy.ndarray."
            )

    group_x_masks = np.asarray(compiled.group_x_masks)
    group_offsets = np.asarray(compiled.group_offsets)
    term_z_masks = np.asarray(compiled.term_z_masks)
    term_coeffs_complex = np.asarray(compiled.term_coeffs_complex)
    term_coeffs_real = np.asarray(compiled.term_coeffs_real)

    named_arrays = (
        ("group_x_masks", group_x_masks),
        ("group_offsets", group_offsets),
        ("term_z_masks", term_z_masks),
        ("term_coeffs_complex", term_coeffs_complex),
        ("term_coeffs_real", term_coeffs_real),
    )
    for name, array in named_arrays:
        if array.ndim != 1:
            raise ValueError(f"Compiled Hamiltonian {name} must be a 1D array.")

    expected_dtypes = (
        ("group_x_masks", group_x_masks, np.dtype(np.uint64)),
        ("group_offsets", group_offsets, np.dtype(np.int64)),
        ("term_z_masks", term_z_masks, np.dtype(np.uint64)),
        ("term_coeffs_complex", term_coeffs_complex, np.dtype(np.complex128)),
        ("term_coeffs_real", term_coeffs_real, np.dtype(np.float64)),
    )
    for name, array, expected_dtype in expected_dtypes:
        if array.dtype != expected_dtype:
            raise TypeError(
                f"Compiled Hamiltonian {name} must have dtype {expected_dtype}, "
                f"got {array.dtype}."
            )

    n_groups = int(group_x_masks.shape[0])
    n_terms = int(term_coeffs_complex.shape[0])
    if group_offsets.shape[0] != n_groups + 1:
        raise ValueError(
            "Compiled Hamiltonian group_offsets length must equal n_groups + 1."
        )
    if term_z_masks.shape[0] != n_terms or term_coeffs_real.shape[0] != n_terms:
        raise ValueError("Compiled Hamiltonian term arrays have inconsistent lengths.")
    if declared_n_terms != n_terms:
        raise ValueError(
            "Compiled Hamiltonian n_compiled_terms does not match its term arrays."
        )
    if int(group_offsets[0]) != 0 or int(group_offsets[-1]) != n_terms:
        raise ValueError(
            "Compiled Hamiltonian group_offsets must start at 0 and end at n_terms."
        )
    if np.any(group_offsets < 0) or np.any(group_offsets > n_terms):
        raise ValueError(
            "Compiled Hamiltonian group_offsets entries must lie within "
            "[0, n_terms]."
        )
    if n_groups == 0:
        if n_terms != 0:
            raise ValueError(
                "Compiled Hamiltonian with terms must contain at least one X-mask group."
            )
    elif np.any(np.diff(group_offsets) <= 0):
        raise ValueError(
            "Compiled Hamiltonian group_offsets must define nonempty groups in "
            "strictly increasing order."
        )

    max_mask = np.uint64((1 << num_qubits) - 1)
    if (
        (group_x_masks.size and np.any(group_x_masks > max_mask))
        or (term_z_masks.size and np.any(term_z_masks > max_mask))
    ):
        raise ValueError("Compiled Hamiltonian masks contain bits outside num_qubits.")

    # Compiler output is canonical: X-mask groups and Z masks within each group
    # are strictly ordered, with duplicate Pauli terms already merged.
    if group_x_masks.size > 1 and np.any(group_x_masks[1:] <= group_x_masks[:-1]):
        raise ValueError(
            "Compiled Hamiltonian group_x_masks must be strictly increasing."
        )
    for group_index in range(n_groups):
        start = int(group_offsets[group_index])
        stop = int(group_offsets[group_index + 1])
        group_z_masks = term_z_masks[start:stop]
        if group_z_masks.size > 1 and np.any(
            group_z_masks[1:] <= group_z_masks[:-1]
        ):
            raise ValueError(
                "Compiled Hamiltonian term_z_masks must be strictly increasing "
                f"within group {group_index}."
            )

    if not np.all(np.isfinite(term_coeffs_complex)):
        raise ValueError("Compiled Hamiltonian coefficients must be finite.")
    if not np.all(np.isfinite(term_coeffs_real)):
        raise ValueError("Compiled Hamiltonian real coefficient cache must be finite.")
    if not np.array_equal(term_coeffs_real, term_coeffs_complex.real):
        raise ValueError(
            "Compiled Hamiltonian real coefficient cache is inconsistent with "
            "term_coeffs_complex.real."
        )

    expected_is_real = bool(
        np.all(np.abs(term_coeffs_complex.imag) <= coefficient_cutoff)
    )
    if compiled.is_real != expected_is_real:
        raise ValueError(
            "Compiled Hamiltonian is_real is inconsistent with its coefficients "
            "and coefficient_cutoff."
        )

    return compiled


def _infer_num_qubits(hamiltonian: Any, num_qubits: int | None) -> int:
    explicit = _as_positive_int_or_none("num_qubits", num_qubits)
    if explicit is not None:
        out = explicit
    elif hasattr(hamiltonian, "num_qubits"):
        inferred = _as_positive_int_or_none(
            "hamiltonian.num_qubits",
            getattr(hamiltonian, "num_qubits"),
        )
        assert inferred is not None
        out = inferred
    elif hasattr(hamiltonian, "n_qubits"):
        inferred = _as_positive_int_or_none(
            "hamiltonian.n_qubits",
            getattr(hamiltonian, "n_qubits"),
        )
        assert inferred is not None
        out = inferred
    else:
        max_q = -1
        for spec, _coeff in _iter_hamiltonian_terms(hamiltonian):
            if isinstance(spec, str):
                max_q = max(max_q, len(spec) - 1)
            elif isinstance(spec, Mapping):
                indices = list(spec.keys())
                if any(
                    isinstance(q, (bool, np.bool_))
                    or not isinstance(q, numbers.Integral)
                    for q in indices
                ):
                    raise TypeError(
                        "Sparse Pauli qubit indices must be integers (not bool)."
                    )
                max_q = max(max_q, *(int(q) for q in indices)) if indices else max_q
            else:
                for q, _p in spec:
                    if isinstance(q, (bool, np.bool_)) or not isinstance(
                        q,
                        numbers.Integral,
                    ):
                        raise TypeError(
                            "Sparse Pauli qubit indices must be integers (not bool)."
                        )
                    max_q = max(max_q, int(q))
        if max_q < 0:
            raise ValueError("Cannot infer num_qubits from an empty Hamiltonian; pass num_qubits explicitly.")
        out = max_q + 1

    if out < 1:
        raise ValueError(f"num_qubits must be >= 1, got {out}.")
    if out > 63:
        raise ValueError(f"This uint64 implementation supports at most 63 qubits. Got {out}.")
    return out


def _is_one_shot_hamiltonian_iterator(hamiltonian: Any) -> bool:
    """Return whether iterating ``hamiltonian`` consumes the object itself."""
    try:
        return iter(hamiltonian) is hamiltonian
    except TypeError:
        return False


def _iter_hamiltonian_terms(hamiltonian: Any) -> list[tuple[Any, complex]]:
    if isinstance(hamiltonian, PauliTerm):
        return [(hamiltonian.pauli, complex(hamiltonian.coeff))]

    if hasattr(hamiltonian, "to_list") and callable(getattr(hamiltonian, "to_list")):
        return [(spec, complex(coeff)) for spec, coeff in hamiltonian.to_list()]

    if hasattr(hamiltonian, "paulis") and hasattr(hamiltonian, "coeffs"):
        paulis = getattr(hamiltonian, "paulis")
        coeffs = getattr(hamiltonian, "coeffs")
        if hasattr(paulis, "to_labels") and callable(getattr(paulis, "to_labels")):
            labels = paulis.to_labels()
        else:
            labels = [p.to_label() if hasattr(p, "to_label") else str(p) for p in paulis]
        return [(label, complex(coeff)) for label, coeff in zip(labels, coeffs)]

    if hasattr(hamiltonian, "terms"):
        terms = getattr(hamiltonian, "terms")
        if isinstance(terms, Mapping):
            return [(spec, complex(coeff)) for spec, coeff in terms.items()]
        return [(spec, complex(coeff)) for spec, coeff in terms]

    try:
        return [(spec, complex(coeff)) for spec, coeff in list(hamiltonian)]
    except Exception as exc:
        raise TypeError(
            "Unsupported Hamiltonian format. Expected Qiskit SparsePauliOp-like, "
            "OpenFermion QubitOperator-like, object with .terms, or iterable of "
            "(pauli_spec, coeff)."
        ) from exc


def _parse_pauli_label_to_masks(label: str, *, num_qubits: int, convention: str) -> tuple[int, int]:
    label = label.strip().replace(" ", "")
    if len(label) != num_qubits:
        raise ValueError(
            f"Pauli label length must equal num_qubits={num_qubits}. Got {label!r} length={len(label)}."
        )
    conv = convention.lower()
    qiskit_like = conv in {"qiskit", "big_endian", "big-endian", "msb"}
    little_like = conv in {"little", "little_endian", "little-endian", "lsb", "internal"}
    if not qiskit_like and not little_like:
        raise ValueError("pauli_label_convention must be 'qiskit' or 'little_endian'.")

    x_mask = 0
    z_mask = 0
    for pos, char in enumerate(label.upper()):
        q = num_qubits - 1 - pos if qiskit_like else pos
        bit = 1 << q
        if char == "I":
            continue
        if char == "X":
            x_mask |= bit
        elif char == "Z":
            z_mask |= bit
        elif char == "Y":
            x_mask |= bit
            z_mask |= bit
        else:
            raise ValueError(f"Invalid Pauli character {char!r} in label {label!r}.")
    return x_mask, z_mask


def _parse_sparse_pauli_to_masks(pauli_spec: Any, *, num_qubits: int) -> tuple[int, int]:
    if isinstance(pauli_spec, Mapping):
        items = list(pauli_spec.items())
    else:
        items = list(pauli_spec)

    x_mask = 0
    z_mask = 0
    seen: set[int] = set()
    for q_raw, op_raw in items:
        if isinstance(q_raw, (bool, np.bool_)) or not isinstance(
            q_raw,
            numbers.Integral,
        ):
            raise TypeError(
                "Sparse Pauli qubit indices must be integers (not bool), "
                f"got {q_raw!r}."
            )
        q = int(q_raw)
        if q < 0 or q >= num_qubits:
            raise ValueError(f"Pauli term references qubit {q}, outside [0, {num_qubits}).")
        if q in seen:
            raise ValueError(f"Pauli term contains multiple operators on qubit {q}.")
        seen.add(q)
        op = str(op_raw).upper()
        bit = 1 << q
        if op == "I":
            continue
        if op == "X":
            x_mask |= bit
        elif op == "Z":
            z_mask |= bit
        elif op == "Y":
            x_mask |= bit
            z_mask |= bit
        else:
            raise ValueError(f"Invalid Pauli operator {op_raw!r} on qubit {q}.")
    return x_mask, z_mask


def _pauli_spec_to_masks(spec: Any, *, num_qubits: int, pauli_label_convention: str) -> tuple[int, int]:
    if isinstance(spec, str):
        return _parse_pauli_label_to_masks(spec, num_qubits=num_qubits, convention=pauli_label_convention)
    if isinstance(spec, Mapping):
        return _parse_sparse_pauli_to_masks(spec, num_qubits=num_qubits)
    if isinstance(spec, (tuple, list)):
        if len(spec) == 0:
            return 0, 0
        if all(isinstance(x, str) and len(x) == 1 for x in spec):
            return _parse_pauli_label_to_masks(
                "".join(spec), num_qubits=num_qubits, convention=pauli_label_convention
            )
        return _parse_sparse_pauli_to_masks(spec, num_qubits=num_qubits)
    raise TypeError(
        "Unsupported Pauli spec. Use a full Pauli label string, a mapping {q: op}, "
        "or a tuple/list of (q, op) pairs."
    )


def _effective_coeff_in_key_formula(x_mask: int, z_mask: int, coeff: complex) -> complex:
    # Y = i X Z under the convention P|x> = i**nY * (-1)**popcount(z & x) |x xor x_mask>.
    n_y_mod4 = int((x_mask & z_mask).bit_count() % 4)
    phase = (1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j)[n_y_mod4]
    return coeff * phase


def _require_finite_hamiltonian_coefficient(
    coeff: complex,
    *,
    context: str,
) -> None:
    """Reject NaN and infinite Pauli coefficients before numerical kernels."""
    if not (math.isfinite(float(coeff.real)) and math.isfinite(float(coeff.imag))):
        raise ValueError(
            "Hamiltonian coefficients must be finite. "
            f"Got {coeff!r} for {context}."
        )


def _reject_duplicate_cutoff_ambiguity(
    raw_terms: Sequence[tuple[Any, complex]],
    *,
    num_qubits: int,
    coefficient_cutoff: float,
    pauli_label_convention: str,
    require_real_pauli_coefficients: bool,
) -> None:
    """Reject duplicates for which merge-before-cutoff changes the Hamiltonian.

    The projected diagonalizer historically drops individual terms before
    merging duplicate Pauli operators, while SqDRIFT historically merges first.
    Both numerical rules are retained.  Inputs for which those rules produce
    different canonical coefficients are rejected instead of allowing two
    modules to evolve and diagonalize different Hamiltonians.
    """
    grouped_coefficients: dict[tuple[int, int], list[complex]] = {}
    representative_specs: dict[tuple[int, int], Any] = {}

    for term_index, (spec, raw_coefficient) in enumerate(raw_terms):
        coefficient = complex(raw_coefficient)
        _require_finite_hamiltonian_coefficient(
            coefficient,
            context=f"input term index {term_index}, Pauli spec {spec!r}",
        )
        x_mask, z_mask = _pauli_spec_to_masks(
            spec,
            num_qubits=num_qubits,
            pauli_label_convention=pauli_label_convention,
        )
        key = (int(x_mask), int(z_mask))
        grouped_coefficients.setdefault(key, []).append(coefficient)
        representative_specs.setdefault(key, spec)

    for key, coefficients in grouped_coefficients.items():
        if len(coefficients) < 2:
            continue

        # SqDRIFT rejects such a group for non-real input independently of the
        # cutoff-order issue, so it is not a merge-vs-cutoff ambiguity.
        if any(abs(coefficient.imag) > coefficient_cutoff for coefficient in coefficients):
            continue

        individually_dropped = False
        diagonalization_sum = 0.0 + 0.0j
        sqdrift_sum = 0.0 + 0.0j
        for coefficient in coefficients:
            diagonalization_coefficient = (
                complex(coefficient.real, 0.0)
                if require_real_pauli_coefficients
                else coefficient
            )
            retained_individually = (
                abs(coefficient) > coefficient_cutoff
                and abs(diagonalization_coefficient) > coefficient_cutoff
            )
            if retained_individually:
                diagonalization_sum += diagonalization_coefficient
            else:
                individually_dropped = True
            sqdrift_sum += complex(coefficient.real, 0.0)

        if not individually_dropped:
            continue

        _require_finite_hamiltonian_coefficient(
            diagonalization_sum,
            context=f"duplicate Pauli term {representative_specs[key]!r}",
        )
        _require_finite_hamiltonian_coefficient(
            sqdrift_sum,
            context=f"duplicate Pauli term {representative_specs[key]!r}",
        )
        diagonalization_final = (
            diagonalization_sum
            if abs(diagonalization_sum) > coefficient_cutoff
            else 0.0 + 0.0j
        )
        sqdrift_final = (
            sqdrift_sum
            if abs(sqdrift_sum) > coefficient_cutoff
            else 0.0 + 0.0j
        )
        if diagonalization_final != sqdrift_final:
            raise ValueError(
                "Duplicate Pauli terms are ambiguous at the coefficient cutoff: "
                f"canonical term {representative_specs[key]!r} would have "
                f"coefficient {diagonalization_final!r} when individual terms "
                "are dropped before merging, but coefficient "
                f"{sqdrift_final!r} when duplicates are merged first. Merge "
                "duplicate terms explicitly before using diagonalization, "
                "variance, or SqDRIFT."
            )


def compile_pauli_hamiltonian(
    hamiltonian: Any,
    *,
    num_qubits: int | None = None,
    coefficient_cutoff: float = 1e-12,
    pauli_label_convention: str = "qiskit",
    require_real_pauli_coefficients: bool = True,
) -> CompiledPauliHamiltonian:
    """Compile a Pauli-sum Hamiltonian into grouped uint64 masks."""
    coefficient_cutoff = _as_nonnegative_float("coefficient_cutoff", coefficient_cutoff)
    if (
        num_qubits is None
        and getattr(hamiltonian, "num_qubits", None) is None
        and getattr(hamiltonian, "n_qubits", None) is None
        and _is_one_shot_hamiltonian_iterator(hamiltonian)
    ):
        raise ValueError(
            "num_qubits must be provided explicitly for a one-shot Hamiltonian "
            "iterator or generator; inferring it would consume the terms before "
            "compilation."
        )
    num_qubits = _infer_num_qubits(hamiltonian, num_qubits)
    raw_terms = _iter_hamiltonian_terms(hamiltonian)
    _reject_duplicate_cutoff_ambiguity(
        raw_terms,
        num_qubits=num_qubits,
        coefficient_cutoff=coefficient_cutoff,
        pauli_label_convention=pauli_label_convention,
        require_real_pauli_coefficients=require_real_pauli_coefficients,
    )
    combined: dict[tuple[int, int], complex] = {}

    for term_index, (spec, coeff) in enumerate(raw_terms):
        coeff = complex(coeff)
        _require_finite_hamiltonian_coefficient(
            coeff,
            context=f"input term index {term_index}, Pauli spec {spec!r}",
        )
        if abs(coeff) <= coefficient_cutoff:
            continue
        if require_real_pauli_coefficients:
            if abs(coeff.imag) > coefficient_cutoff:
                raise ValueError(
                    "Hermitian Pauli-sum coefficients must be real in the Pauli basis. "
                    f"Got coefficient {coeff!r} for term {spec!r}."
                )
            coeff = complex(coeff.real, 0.0)
        x_mask, z_mask = _pauli_spec_to_masks(
            spec, num_qubits=num_qubits, pauli_label_convention=pauli_label_convention
        )
        eff = _effective_coeff_in_key_formula(x_mask, z_mask, coeff)
        if abs(eff) <= coefficient_cutoff:
            continue
        key = (int(x_mask), int(z_mask))
        merged_coeff = combined.get(key, 0.0 + 0.0j) + eff
        _require_finite_hamiltonian_coefficient(
            merged_coeff,
            context=(
                "merged Pauli term with "
                f"x_mask={key[0]} and z_mask={key[1]}"
            ),
        )
        combined[key] = merged_coeff

    compact_terms: list[tuple[int, int, complex]] = []
    for (x_mask, z_mask), coeff in combined.items():
        _require_finite_hamiltonian_coefficient(
            coeff,
            context=(
                "merged Pauli term with "
                f"x_mask={x_mask} and z_mask={z_mask}"
            ),
        )
        if abs(coeff) > coefficient_cutoff:
            compact_terms.append((x_mask, z_mask, coeff))
    compact_terms.sort(key=lambda t: (t[0], t[1]))

    if not compact_terms:
        return CompiledPauliHamiltonian(
            num_qubits=num_qubits,
            group_x_masks=np.empty(0, dtype=np.uint64),
            group_offsets=np.zeros(1, dtype=np.int64),
            term_z_masks=np.empty(0, dtype=np.uint64),
            term_coeffs_complex=np.empty(0, dtype=np.complex128),
            term_coeffs_real=np.empty(0, dtype=np.float64),
            is_real=True,
            n_input_terms=len(raw_terms),
            n_compiled_terms=0,
            coefficient_cutoff=coefficient_cutoff,
            pauli_label_convention=pauli_label_convention,
        )

    group_x_masks: list[int] = []
    group_offsets: list[int] = [0]
    term_z_masks: list[int] = []
    term_coeffs: list[complex] = []
    current_x: int | None = None
    for x_mask, z_mask, coeff in compact_terms:
        if current_x is None or x_mask != current_x:
            if current_x is not None:
                group_offsets.append(len(term_z_masks))
            current_x = x_mask
            group_x_masks.append(x_mask)
        term_z_masks.append(z_mask)
        term_coeffs.append(coeff)
    group_offsets.append(len(term_z_masks))

    coeffs_complex = np.asarray(term_coeffs, dtype=np.complex128)
    is_real = bool(np.all(np.abs(coeffs_complex.imag) <= coefficient_cutoff))
    return CompiledPauliHamiltonian(
        num_qubits=num_qubits,
        group_x_masks=np.asarray(group_x_masks, dtype=np.uint64),
        group_offsets=np.asarray(group_offsets, dtype=np.int64),
        term_z_masks=np.asarray(term_z_masks, dtype=np.uint64),
        term_coeffs_complex=coeffs_complex,
        term_coeffs_real=coeffs_complex.real.astype(np.float64, copy=True),
        is_real=is_real,
        n_input_terms=len(raw_terms),
        n_compiled_terms=len(term_z_masks),
        coefficient_cutoff=coefficient_cutoff,
        pauli_label_convention=pauli_label_convention,
    )


def _require_hermitian_compiled_pauli_hamiltonian(
    compiled: CompiledPauliHamiltonian,
    *,
    context: str,
) -> None:
    """Require real coefficients in the canonical Pauli basis."""
    _validate_compiled_pauli_hamiltonian(compiled)
    cutoff = float(compiled.coefficient_cutoff)
    for group_index, x_mask_raw in enumerate(compiled.group_x_masks):
        x_mask = int(x_mask_raw)
        start = int(compiled.group_offsets[group_index])
        stop = int(compiled.group_offsets[group_index + 1])
        for term_index in range(start, stop):
            z_mask = int(compiled.term_z_masks[term_index])
            effective_coefficient = complex(compiled.term_coeffs_complex[term_index])
            n_y_mod4 = int((x_mask & z_mask).bit_count() % 4)
            phase = (
                1.0 + 0.0j,
                0.0 + 1.0j,
                -1.0 + 0.0j,
                0.0 - 1.0j,
            )[n_y_mod4]
            pauli_coefficient = effective_coefficient * np.conjugate(phase)
            if float(pauli_coefficient.imag) != 0.0:
                raise ValueError(
                    f"{context} requires a Hermitian Pauli Hamiltonian. The "
                    "final canonical Pauli coefficient at term "
                    f"{term_index} is {pauli_coefficient!r}, whose imaginary "
                    "part is nonzero. Use require_real_pauli_coefficients=True "
                    f"to canonicalize input noise up to coefficient_cutoff={cutoff}."
                )


def _compiled_pauli_hamiltonians_are_equal(
    left: CompiledPauliHamiltonian,
    right: CompiledPauliHamiltonian,
) -> bool:
    """Return whether two validated compiled objects represent the same operator."""
    _validate_compiled_pauli_hamiltonian(left)
    _validate_compiled_pauli_hamiltonian(right)
    return bool(
        int(left.num_qubits) == int(right.num_qubits)
        and np.array_equal(left.group_x_masks, right.group_x_masks)
        and np.array_equal(left.group_offsets, right.group_offsets)
        and np.array_equal(left.term_z_masks, right.term_z_masks)
        and np.array_equal(left.term_coeffs_complex, right.term_coeffs_complex)
    )


def _pack_logical_basis_uint64(basis: np.ndarray, *, bit_order: str = "qiskit") -> np.ndarray:
    """Pack logical basis rows into uint64 keys.

    bit_order="qiskit" or "big_endian":
        basis[:, 0] is the leftmost/displayed bit and maps to uint64 bit n-1.
        Example: [0, 0, 1] -> key 1.

    bit_order="little_endian" or "internal":
        basis[:, q] maps directly to uint64 bit q.
        Example: [0, 0, 1] -> key 4.
    """
    order = bit_order.lower()
    if order in {"qiskit", "big_endian", "big-endian", "msb", "display", "displayed"}:
        return _pack_logical_basis_uint64_big_endian_kernel(basis)
    if order in {"little", "little_endian", "little-endian", "lsb", "internal"}:
        return _pack_logical_basis_uint64_kernel(basis)
    raise ValueError("logical_basis_bit_order must be 'qiskit'/'big_endian' or 'little_endian'.")


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _table_size_for_num_keys(n_keys: int, max_load_factor: float) -> int:
    return max(2, _next_power_of_two(int(math.ceil(n_keys / max_load_factor))))


def _csr_memory_bytes(dim: int, nnz: int, is_real: bool) -> int:
    data_bytes = 8 if is_real else 16
    return int((dim + 1) * 8 + nnz * (4 + data_bytes))


def _validate_basis(logical_basis: np.ndarray, *, num_qubits: int, validate_bits: bool) -> np.ndarray:
    basis = np.asarray(logical_basis)
    if basis.ndim != 2:
        raise ValueError("logical_basis must be a 2D array.")
    if basis.dtype != np.uint8:
        raise TypeError(f"logical_basis must have dtype np.uint8, got {basis.dtype}.")
    if basis.shape[1] != num_qubits:
        raise ValueError(f"logical_basis has {basis.shape[1]} bits, but Hamiltonian has {num_qubits} qubits.")
    if basis.shape[0] > np.iinfo(np.int32).max:
        raise ValueError("basis dimension exceeds int32 index capacity.")
    if validate_bits and not np.all((basis == 0) | (basis == 1)):
        raise ValueError("logical_basis must contain only 0 or 1.")
    return np.ascontiguousarray(basis, dtype=np.uint8)


def _import_primme_module():
    try:
        import primme  # type: ignore
        return primme
    except Exception as exc:
        raise ImportError(
            "ProjectedPauliCSRPRIMMEDiagonalizer requires the `primme` Python package. "
            "Install it with `pip install primme`. No SciPy eigsh fallback is provided by design."
        ) from exc


# =============================================================================
# LinearOperator wrapper
# =============================================================================


def _make_numba_csr_linear_operator(csr: ProjectedCSR) -> LinearOperator:
    n = int(csr.indptr.shape[0] - 1)
    if csr.is_real:
        dtype = np.dtype(np.float64)

        def matvec(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
            return _csr_matvec_real(csr.indptr, csr.indices, csr.data, np.ascontiguousarray(x_arr))

        def matmat(X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                return matvec(X_arr).reshape(n, 1)
            return _csr_matmat_real(csr.indptr, csr.indices, csr.data, np.ascontiguousarray(X_arr))
    else:
        dtype = np.dtype(np.complex128)

        def matvec(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.complex128).reshape(-1)
            return _csr_matvec_complex(csr.indptr, csr.indices, csr.data, np.ascontiguousarray(x_arr))

        def matmat(X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=np.complex128)
            if X_arr.ndim == 1:
                return matvec(X_arr).reshape(n, 1)
            return _csr_matmat_complex(csr.indptr, csr.indices, csr.data, np.ascontiguousarray(X_arr))

    return LinearOperator((n, n), matvec=matvec, matmat=matmat, dtype=dtype)


# =============================================================================
# Main diagonalizer
# =============================================================================


class ProjectedPauliCSRPRIMMEDiagonalizer:
    """Diagonalize one fixed Pauli Hamiltonian in supplied logical subspaces.

    The Hamiltonian is compiled at construction. The first argument to
    :meth:`__call__` is retained for recovery-driver compatibility: the exact
    construction object follows an identity fast path, while a different
    object is compiled and required to represent the same Hamiltonian before
    numerical work begins. Successful calls update :attr:`last_stats`.
    """

    def __init__(
        self,
        hamiltonian: Any,
        *,
        num_qubits: int | None = None,
        max_logical_qubits: int = DEFAULT_MAX_LOGICAL_QUBITS,
        pauli_label_convention: str = "qiskit",
        logical_basis_bit_order: str = "qiskit",
        require_real_pauli_coefficients: bool = True,
        coefficient_cutoff: float = 1e-12,
        matrix_element_cutoff: float = 1e-14,
        csr_memory_limit_gib: float = 120.0,
        hash_table_max_load_factor: float = 0.65,
        eig_tol: float = 1e-8,
        maxiter: int | None = None,
        ncv: int | None = 80,
        method: Any | None = None,
        maxBlockSize: int = 0,
        minRestartSize: int = 0,
        maxPrevRetain: int = 0,
        residual_tol: float | None = None,
        residual_check: bool = True,
        num_threads: int | None = None,
        return_stats: bool = True,
        validate_basis_bits: bool = True,
        warn_if_hamiltonian_argument_differs: bool = False,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.compiled = compile_pauli_hamiltonian(
            hamiltonian,
            num_qubits=num_qubits,
            coefficient_cutoff=coefficient_cutoff,
            pauli_label_convention=pauli_label_convention,
            require_real_pauli_coefficients=require_real_pauli_coefficients,
        )
        _require_hermitian_compiled_pauli_hamiltonian(
            self.compiled,
            context="Projected eigsh diagonalization",
        )
        self.require_real_pauli_coefficients = bool(
            require_real_pauli_coefficients
        )
        max_logical_qubits_checked = _as_positive_int_or_none("max_logical_qubits", max_logical_qubits)
        assert max_logical_qubits_checked is not None
        if max_logical_qubits_checked > 63:
            raise ValueError("max_logical_qubits must be <= 63 for uint64 packing.")
        if self.compiled.num_qubits > max_logical_qubits_checked:
            raise ValueError(
                f"Hamiltonian has {self.compiled.num_qubits} qubits, exceeding max_logical_qubits="
                f"{max_logical_qubits_checked}."
            )
        self.max_logical_qubits = max_logical_qubits_checked

        logical_basis_bit_order = str(logical_basis_bit_order).lower()
        if logical_basis_bit_order not in {
            "qiskit", "big_endian", "big-endian", "msb", "display", "displayed",
            "little", "little_endian", "little-endian", "lsb", "internal",
        }:
            raise ValueError(
                "logical_basis_bit_order must be 'qiskit'/'big_endian' or 'little_endian'."
            )
        self.logical_basis_bit_order = logical_basis_bit_order

        self.matrix_element_cutoff = _as_nonnegative_float("matrix_element_cutoff", matrix_element_cutoff)
        self.csr_memory_limit_gib = _as_positive_float("csr_memory_limit_gib", csr_memory_limit_gib)
        self.hash_table_max_load_factor = float(hash_table_max_load_factor)
        if not (0.05 <= self.hash_table_max_load_factor < 0.95):
            raise ValueError("hash_table_max_load_factor must be in [0.05, 0.95).")

        self.eig_tol = _as_positive_float("eig_tol", eig_tol)
        self.maxiter = _as_positive_int_or_none("maxiter", maxiter)
        self.ncv = _as_positive_int_or_none("ncv", ncv)
        self.method = method
        self.maxBlockSize = int(maxBlockSize)
        self.minRestartSize = int(minRestartSize)
        self.maxPrevRetain = int(maxPrevRetain)
        self.residual_check = bool(residual_check)
        self.residual_tol = float(residual_tol) if residual_tol is not None else max(100.0 * self.eig_tol, 1e-8)
        if self.residual_tol <= 0.0 or not math.isfinite(self.residual_tol):
            raise ValueError("residual_tol must be finite and > 0.")
        self.return_stats = bool(return_stats)
        self.validate_basis_bits = bool(validate_basis_bits)
        self.warn_if_hamiltonian_argument_differs = bool(warn_if_hamiltonian_argument_differs)

        self.num_threads = _as_positive_int_or_none("num_threads", num_threads)
        if self.num_threads is not None:
            nb.set_num_threads(self.num_threads)

        self.last_stats: ProjectedPauliBuildStats | None = None
        self.__name__ = "ProjectedPauliCSRPRIMMEDiagonalizer"

    @property
    def num_qubits(self) -> int:
        return self.compiled.num_qubits

    def get_runtime_num_threads(self) -> int:
        """Return Numba's active worker-thread count in the current process."""
        return int(nb.get_num_threads())

    def __repr__(self) -> str:
        return (
            "ProjectedPauliCSRPRIMMEDiagonalizer("
            f"num_qubits={self.compiled.num_qubits}, "
            f"n_terms={self.compiled.n_compiled_terms}, "
            f"n_groups={self.compiled.n_groups}, "
            f"is_real={self.compiled.is_real}, "
            f"eig_tol={self.eig_tol})"
        )

    def __call__(
        self,
        hamiltonian: Any,
        logical_basis: np.ndarray,
        *,
        seed: int | None = None,
        warm_start_basis: np.ndarray | None = None,
        warm_start_keys: np.ndarray | None = None,
        warm_start_coefficients: np.ndarray | None = None,
    ) -> DiagonalizationResult:
        """Return the lowest projected Ritz pair for ``logical_basis``.

        ``warm_start_keys`` takes precedence over ``warm_start_basis`` when
        both are supplied. The ``hamiltonian`` argument must be the construction
        object or a separately constructed Hamiltonian with the same canonical
        compiled representation. The exact construction object is intentionally
        not re-read or recompiled. ``warn_if_hamiltonian_argument_differs`` is a
        retained compatibility keyword; mismatches always raise ``ValueError``.
        """
        if self.num_threads is not None:
            nb.set_num_threads(self.num_threads)

        if hamiltonian is not self.hamiltonian:
            if isinstance(hamiltonian, CompiledPauliHamiltonian):
                call_compiled = _validate_compiled_pauli_hamiltonian(hamiltonian)
            else:
                call_compiled = compile_pauli_hamiltonian(
                    hamiltonian,
                    num_qubits=self.compiled.num_qubits,
                    coefficient_cutoff=self.compiled.coefficient_cutoff,
                    pauli_label_convention=self.compiled.pauli_label_convention,
                    require_real_pauli_coefficients=(
                        self.require_real_pauli_coefficients
                    ),
                )
            if not _compiled_pauli_hamiltonians_are_equal(
                self.compiled,
                call_compiled,
            ):
                raise ValueError(
                    "The call-time Hamiltonian does not match the Hamiltonian "
                    "compiled when ProjectedPauliCSRPRIMMEDiagonalizer was "
                    "constructed."
                )

        basis = _validate_basis(logical_basis, num_qubits=self.compiled.num_qubits, validate_bits=self.validate_basis_bits)
        D = int(basis.shape[0])
        if D == 0:
            raise ValueError("logical_basis must contain at least one row.")

        t0 = time.perf_counter()
        keys = _pack_logical_basis_uint64(basis, bit_order=self.logical_basis_bit_order)
        t1 = time.perf_counter()
        if np.unique(keys).shape[0] != D:
            raise ValueError("logical_basis contains duplicate computational basis states.")

        if self.compiled.n_compiled_terms == 0:
            coeff = np.array([1.0 + 0.0j], dtype=np.complex128)
            self.last_stats = ProjectedPauliBuildStats(
                basis_dim=D,
                num_qubits=self.compiled.num_qubits,
                n_groups=0,
                n_terms=0,
                nnz=0,
                csr_memory_gib=0.0,
                pack_seconds=t1 - t0,
                residual_norm=0.0,
                relative_residual=0.0,
            )
            return DiagonalizationResult(energy=0.0, logical_basis=basis[:1].copy(), coefficients=coeff)

        csr, hash_keys, hash_vals, stats = self._assemble_projected_csr(keys, pack_seconds=t1 - t0)
        if D == 1:
            energy = self._single_state_energy_from_csr(csr)
            coeff = np.array([1.0 + 0.0j], dtype=np.complex128)
            stats.residual_norm = 0.0
            stats.relative_residual = 0.0
            self.last_stats = stats
            return DiagonalizationResult(energy=energy, logical_basis=basis.copy(), coefficients=coeff)

        A = _make_numba_csr_linear_operator(csr)
        if warm_start_keys is None:
            warm_start_keys_t, warm_start_coeffs = self._prepare_warm_start(
                warm_start_basis,
                warm_start_coefficients,
            )
        else:
            warm_start_keys_t, warm_start_coeffs = self._prepare_cached_warm_start(
                warm_start_keys,
                warm_start_coefficients,
            )
        v0, used_warm_start = self._make_initial_guess(
            D,
            csr.is_real,
            hash_keys,
            hash_vals,
            seed,
            warm_start_keys=warm_start_keys_t,
            warm_start_coefficients=warm_start_coeffs,
        )
        stats.used_warm_start = used_warm_start

        primme = _import_primme_module()
        solve_start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "k": 1,
            "which": "SA",
            "tol": self.eig_tol,
            "v0": v0,
            "return_eigenvectors": True,
            "return_stats": self.return_stats,
        }
        if self.maxiter is not None:
            kwargs["maxiter"] = self.maxiter
        if self.ncv is not None:
            kwargs["ncv"] = self.ncv
        if self.method is not None:
            kwargs["method"] = self.method
        if self.maxBlockSize > 0:
            kwargs["maxBlockSize"] = self.maxBlockSize
        if self.minRestartSize > 0:
            kwargs["minRestartSize"] = self.minRestartSize
        if self.maxPrevRetain > 0:
            kwargs["maxPrevRetain"] = self.maxPrevRetain

        out = primme.eigsh(A, **kwargs)
        solve_stop = time.perf_counter()
        if self.return_stats:
            evals, evecs, primme_stats = out
            stats.primme_stats = dict(primme_stats)
        else:
            evals, evecs = out
        stats.solve_seconds = solve_stop - solve_start

        evals = np.asarray(evals)
        evecs = np.asarray(evecs)
        if evals.ndim != 1 or evals.shape[0] < 1:
            raise RuntimeError("PRIMME returned invalid eigenvalues.")
        if evecs.ndim != 2 or evecs.shape[0] != D or evecs.shape[1] < 1:
            raise RuntimeError("PRIMME returned invalid eigenvectors.")

        energy = float(np.real(evals[0]))
        if not math.isfinite(energy):
            raise RuntimeError("PRIMME returned a non-finite eigenvalue.")
        coeffs = np.asarray(evecs[:, 0], dtype=np.complex128)
        if not np.all(np.isfinite(coeffs)):
            raise RuntimeError("PRIMME returned non-finite eigenvector coefficients.")
        norm = float(np.linalg.norm(coeffs))
        if norm <= 0.0 or not math.isfinite(norm):
            raise RuntimeError("PRIMME returned an eigenvector with invalid norm.")
        coeffs = coeffs / norm
        if csr.is_real:
            coeffs = coeffs.real.astype(np.complex128, copy=False)

        if self.residual_check:
            residual_norm, relative_residual = self._compute_residual(A, energy, coeffs)
            stats.residual_norm = residual_norm
            stats.relative_residual = relative_residual
            if relative_residual > self.residual_tol:
                raise RuntimeError(
                    "PRIMME eigenpair residual check failed: "
                    f"relative_residual={relative_residual:.3e} > residual_tol={self.residual_tol:.3e}."
                )

        self.last_stats = stats
        return DiagonalizationResult(
            energy=energy,
            logical_basis=basis.copy(),
            coefficients=coeffs.astype(np.complex128, copy=False),
        )

    def _assemble_projected_csr(self, keys: np.ndarray, *, pack_seconds: float) -> tuple[ProjectedCSR, np.ndarray, np.ndarray, ProjectedPauliBuildStats]:
        D = int(keys.shape[0])
        table_size = _table_size_for_num_keys(D, self.hash_table_max_load_factor)
        hash_start = time.perf_counter()
        hash_keys = np.empty(table_size, dtype=np.uint64)
        hash_vals = np.empty(table_size, dtype=np.int32)
        duplicate_i = _build_hash_table_uint64_i32_kernel(keys, hash_keys, hash_vals)
        if duplicate_i >= 0:
            raise ValueError(f"logical_basis contains a duplicate packed key at row {int(duplicate_i)}.")
        hash_stop = time.perf_counter()

        count_start = time.perf_counter()
        if self.compiled.is_real:
            row_counts = _count_projected_rows_real(
                keys,
                hash_keys,
                hash_vals,
                self.compiled.group_x_masks,
                self.compiled.group_offsets,
                self.compiled.term_z_masks,
                self.compiled.term_coeffs_real,
                self.matrix_element_cutoff,
            )
        else:
            row_counts = _count_projected_rows_complex(
                keys,
                hash_keys,
                hash_vals,
                self.compiled.group_x_masks,
                self.compiled.group_offsets,
                self.compiled.term_z_masks,
                self.compiled.term_coeffs_complex,
                self.matrix_element_cutoff,
            )
        count_stop = time.perf_counter()

        indptr = np.empty(D + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(row_counts, out=indptr[1:])
        nnz = int(indptr[-1])
        memory_bytes = _csr_memory_bytes(D, nnz, self.compiled.is_real)
        memory_gib = memory_bytes / (1024.0 ** 3)
        if memory_gib > self.csr_memory_limit_gib:
            raise MemoryError(
                "Projected CSR memory estimate exceeds limit: "
                f"{memory_gib:.2f} GiB > csr_memory_limit_gib={self.csr_memory_limit_gib:.2f} GiB."
            )
        if nnz > np.iinfo(np.int32).max:
            raise MemoryError(f"Projected CSR nnz={nnz} exceeds int32 capacity.")

        indices = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=np.float64 if self.compiled.is_real else np.complex128)

        fill_start = time.perf_counter()
        if self.compiled.is_real:
            _fill_projected_csr_real(
                keys,
                hash_keys,
                hash_vals,
                self.compiled.group_x_masks,
                self.compiled.group_offsets,
                self.compiled.term_z_masks,
                self.compiled.term_coeffs_real,
                self.matrix_element_cutoff,
                indptr,
                indices,
                data,
            )
        else:
            _fill_projected_csr_complex(
                keys,
                hash_keys,
                hash_vals,
                self.compiled.group_x_masks,
                self.compiled.group_offsets,
                self.compiled.term_z_masks,
                self.compiled.term_coeffs_complex,
                self.matrix_element_cutoff,
                indptr,
                indices,
                data,
            )
        fill_stop = time.perf_counter()
        if not np.all(np.isfinite(data)):
            raise ValueError(
                "Projected Hamiltonian matrix elements must remain finite; "
                "Pauli-term accumulation overflowed."
            )

        csr = ProjectedCSR(
            indptr=indptr,
            indices=indices,
            data=data,
            is_real=self.compiled.is_real,
            nnz=nnz,
            memory_bytes=memory_bytes,
        )
        stats = ProjectedPauliBuildStats(
            basis_dim=D,
            num_qubits=self.compiled.num_qubits,
            n_groups=self.compiled.n_groups,
            n_terms=self.compiled.n_compiled_terms,
            nnz=nnz,
            csr_memory_gib=memory_gib,
            pack_seconds=pack_seconds,
            hash_seconds=hash_stop - hash_start,
            count_seconds=count_stop - count_start,
            fill_seconds=fill_stop - fill_start,
        )
        return csr, hash_keys, hash_vals, stats

    def _make_initial_guess(
        self,
        D: int,
        is_real: bool,
        hash_keys: np.ndarray,
        hash_vals: np.ndarray,
        seed: int | None,
        *,
        warm_start_keys: np.ndarray | None,
        warm_start_coefficients: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        rng = np.random.default_rng(seed)
        used_warm_start = False
        if is_real:
            v = np.zeros(D, dtype=np.float64)
            if warm_start_keys is not None and warm_start_coefficients is not None:
                _fill_warm_start_real(warm_start_keys, warm_start_coefficients, hash_keys, hash_vals, v)
                used_warm_start = bool(np.linalg.norm(v) > 0.0)
            norm = float(np.linalg.norm(v))
            if norm <= 0.0 or not math.isfinite(norm):
                v = rng.standard_normal(D).astype(np.float64)
                norm = float(np.linalg.norm(v))
                used_warm_start = False
            return (v / norm).reshape(D, 1), used_warm_start

        v = np.zeros(D, dtype=np.complex128)
        if warm_start_keys is not None and warm_start_coefficients is not None:
            _fill_warm_start_complex(warm_start_keys, warm_start_coefficients, hash_keys, hash_vals, v)
            used_warm_start = bool(np.linalg.norm(v) > 0.0)
        norm = float(np.linalg.norm(v))
        if norm <= 0.0 or not math.isfinite(norm):
            v = rng.standard_normal(D) + 1.0j * rng.standard_normal(D)
            v = v.astype(np.complex128)
            norm = float(np.linalg.norm(v))
            used_warm_start = False
        return (v / norm).reshape(D, 1), used_warm_start

    def _prepare_warm_start(
        self,
        warm_start_basis: np.ndarray | None,
        warm_start_coefficients: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if warm_start_basis is None and warm_start_coefficients is None:
            return None, None
        if warm_start_basis is None or warm_start_coefficients is None:
            raise ValueError(
                "warm_start_basis and warm_start_coefficients must be provided together."
            )

        basis = _validate_basis(
            warm_start_basis,
            num_qubits=self.compiled.num_qubits,
            validate_bits=self.validate_basis_bits,
        )
        coeffs = np.asarray(warm_start_coefficients)
        if coeffs.ndim != 1:
            raise ValueError("warm_start_coefficients must be a 1D array.")
        if coeffs.shape[0] != basis.shape[0]:
            raise ValueError(
                "warm_start_coefficients length must match warm_start_basis rows."
            )
        if len(basis) == 0:
            return None, None
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("warm_start_coefficients must be finite.")

        keys = _pack_logical_basis_uint64(basis, bit_order=self.logical_basis_bit_order)
        if np.unique(keys).shape[0] != int(keys.shape[0]):
            raise ValueError("warm_start_basis contains duplicate computational basis states.")

        coeffs = np.asarray(coeffs, dtype=np.complex128)
        norm = float(np.linalg.norm(coeffs))
        if norm <= 0.0 or not math.isfinite(norm):
            return None, None
        return keys.copy(), (coeffs / norm).copy()

    def prepare_warm_start(
        self,
        warm_start_basis: np.ndarray | None,
        warm_start_coefficients: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Validate and pack a warm start once for reuse across many batches."""
        return self._prepare_warm_start(warm_start_basis, warm_start_coefficients)

    def _prepare_cached_warm_start(
        self,
        warm_start_keys: np.ndarray | None,
        warm_start_coefficients: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if warm_start_keys is None and warm_start_coefficients is None:
            return None, None
        if warm_start_keys is None or warm_start_coefficients is None:
            raise ValueError(
                "warm_start_keys and warm_start_coefficients must be provided together."
            )

        keys = np.asarray(warm_start_keys, dtype=np.uint64)
        coeffs = np.asarray(warm_start_coefficients, dtype=np.complex128)
        if keys.ndim != 1:
            raise ValueError("warm_start_keys must be a 1D array.")
        if coeffs.ndim != 1:
            raise ValueError("warm_start_coefficients must be a 1D array.")
        if coeffs.shape[0] != keys.shape[0]:
            raise ValueError(
                "warm_start_coefficients length must match warm_start_keys."
            )
        if len(keys) == 0:
            return None, None
        return keys, coeffs

    def spawn(self, *, num_threads: int | None = None) -> "ProjectedPauliCSRPRIMMEDiagonalizer":
        """Create a worker copy sharing the compiled Hamiltonian with fresh statistics."""
        clone = object.__new__(type(self))
        clone.hamiltonian = self.hamiltonian
        clone.compiled = self.compiled
        clone.require_real_pauli_coefficients = self.require_real_pauli_coefficients
        clone.max_logical_qubits = self.max_logical_qubits
        clone.logical_basis_bit_order = self.logical_basis_bit_order
        clone.matrix_element_cutoff = self.matrix_element_cutoff
        clone.csr_memory_limit_gib = self.csr_memory_limit_gib
        clone.hash_table_max_load_factor = self.hash_table_max_load_factor
        clone.eig_tol = self.eig_tol
        clone.maxiter = self.maxiter
        clone.ncv = self.ncv
        clone.method = self.method
        clone.maxBlockSize = self.maxBlockSize
        clone.minRestartSize = self.minRestartSize
        clone.maxPrevRetain = self.maxPrevRetain
        clone.residual_check = self.residual_check
        clone.residual_tol = self.residual_tol
        clone.return_stats = self.return_stats
        clone.validate_basis_bits = self.validate_basis_bits
        clone.warn_if_hamiltonian_argument_differs = self.warn_if_hamiltonian_argument_differs
        clone.num_threads = (
            self.num_threads
            if num_threads is None
            else _as_positive_int_or_none("num_threads", num_threads)
        )
        clone.last_stats = None
        clone.__name__ = self.__name__
        return clone

    @staticmethod
    def _single_state_energy_from_csr(csr: ProjectedCSR) -> float:
        if csr.nnz == 0:
            return 0.0
        energy = float(np.real(np.sum(csr.data)))
        if not math.isfinite(energy):
            raise ValueError("Single-state projected energy must remain finite.")
        return energy

    @staticmethod
    def _compute_residual(A: LinearOperator, energy: float, coeffs: np.ndarray) -> tuple[float, float]:
        if np.dtype(A.dtype).kind == "f":
            vec = np.asarray(coeffs.real, dtype=np.float64)
            Av = np.asarray(A.matvec(vec), dtype=np.float64)
            r = Av - float(energy) * vec
        else:
            vec = np.asarray(coeffs, dtype=np.complex128)
            Av = np.asarray(A.matvec(vec), dtype=np.complex128)
            r = Av - float(energy) * vec
        residual_norm = float(np.linalg.norm(r))
        relative_residual = residual_norm / max(1.0, abs(float(energy)))
        if not math.isfinite(residual_norm) or not math.isfinite(relative_residual):
            raise RuntimeError("PRIMME residual calculation returned a non-finite value.")
        return residual_norm, relative_residual

    def build_projected_csr_for_debug(self, logical_basis: np.ndarray) -> tuple[ProjectedCSR, np.ndarray]:
        """Build projected CSR without calling PRIMME; useful for small unit tests."""
        basis = _validate_basis(logical_basis, num_qubits=self.compiled.num_qubits, validate_bits=self.validate_basis_bits)
        keys = _pack_logical_basis_uint64(basis, bit_order=self.logical_basis_bit_order)
        if np.unique(keys).shape[0] != int(basis.shape[0]):
            raise ValueError("logical_basis contains duplicate computational basis states.")
        csr, _hash_keys, _hash_vals, stats = self._assemble_projected_csr(keys, pack_seconds=0.0)
        self.last_stats = stats
        return csr, keys

    def to_config(self) -> dict[str, Any]:
        return {
            "diagonalizer": "ProjectedPauliCSRPRIMMEDiagonalizer",
            "package_version": PACKAGE_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "solver": "primme.eigsh",
            "num_qubits": self.compiled.num_qubits,
            "n_input_terms": self.compiled.n_input_terms,
            "n_compiled_terms": self.compiled.n_compiled_terms,
            "n_groups": self.compiled.n_groups,
            "is_real": self.compiled.is_real,
            "logical_basis_bit_order": self.logical_basis_bit_order,
            "eig_tol": self.eig_tol,
            "ncv": self.ncv,
            "maxiter": self.maxiter,
            "matrix_element_cutoff": self.matrix_element_cutoff,
            "csr_memory_limit_gib": self.csr_memory_limit_gib,
            "hash_table_max_load_factor": self.hash_table_max_load_factor,
            "warm_start_scheme": "explicit_global_best_packed_cache",
            "residual_check": self.residual_check,
            "residual_tol": self.residual_tol,
            "num_threads": self.num_threads,
        }


def make_projected_pauli_primme_diagonalize_fn(hamiltonian: Any, **kwargs: Any) -> ProjectedPauliCSRPRIMMEDiagonalizer:
    """Convenience factory returning a callable diagonalize_fn."""
    return ProjectedPauliCSRPRIMMEDiagonalizer(hamiltonian, **kwargs)


def csr_to_dense_for_debug(csr: ProjectedCSR) -> np.ndarray:
    """Convert a ProjectedCSR to dense matrix for small tests only."""
    n = csr.indptr.shape[0] - 1
    out = np.zeros((n, n), dtype=np.complex128)
    for row in range(n):
        s, e = int(csr.indptr[row]), int(csr.indptr[row + 1])
        out[row, csr.indices[s:e]] = csr.data[s:e]
    return out


__all__ = [
    "DEFAULT_MAX_LOGICAL_QUBITS",
    "PauliTerm",
    "CompiledPauliHamiltonian",
    "ProjectedCSR",
    "ProjectedPauliBuildStats",
    "ProjectedPauliCSRPRIMMEDiagonalizer",
    "compile_pauli_hamiltonian",
    "make_projected_pauli_primme_diagonalize_fn",
    "csr_to_dense_for_debug",
]
