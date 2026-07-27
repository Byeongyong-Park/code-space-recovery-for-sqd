"""Energy-variance analysis helpers for code-space recovery results.

This module computes the full-Hamiltonian variance of an SQD Ritz state

    |psi> = sum_i coefficients[i] |basis_bitstrings[i]>

using the same Pauli-label and logical-bit conventions as the projected
diagonalizer.  The variance is computed in the full computational Hilbert space,
not only inside the projected basis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import math

import numpy as np

try:
    import numba as nb
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "energy_variance requires numba. Install the same dependencies used by "
        "the projected diagonalizer."
    ) from exc

try:  # package import
    from .diagonalization import (
        CompiledPauliHamiltonian,
        _pack_logical_basis_uint64,
        compile_pauli_hamiltonian,
    )
except Exception:  # pragma: no cover - flat-module import fallback
    from diagonalization import (  # type: ignore
        CompiledPauliHamiltonian,
        _pack_logical_basis_uint64,
        compile_pauli_hamiltonian,
    )


MODULE_VERSION = "v1.0"
ALGORITHM_VERSION = "code_space_recovery_v1.0"
__version__ = MODULE_VERSION


@dataclass(frozen=True)
class EnergyVarianceResult:
    """Full-Hamiltonian energy variance of one SQD state."""

    energy: float
    energy_imag_abs: float
    h2_expectation: float
    variance: float
    std: float
    coefficient_norm_before_normalization: float
    n_qubits: int
    n_basis: int
    n_hamiltonian_terms: int
    n_hamiltonian_groups: int
    n_hpsi_contributions: int
    n_hpsi_support: int
    inside_support_norm2: float
    outside_support_norm2: float
    outside_support_fraction: float
    estimated_contribution_workspace_gib: float
    coefficient_cutoff: float
    amplitude_cutoff: float
    pauli_label_convention: str
    logical_basis_bit_order: str
    module_version: str = MODULE_VERSION
    algorithm_version: str = ALGORITHM_VERSION
    reported_energy: float | None = None
    reported_energy_difference: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@nb.njit(cache=True, inline="always")
def _parity_u64(x: np.uint64) -> np.int64:
    p = np.int64(0)
    while x != np.uint64(0):
        p ^= np.int64(1)
        x &= x - np.uint64(1)
    return p


@nb.njit(cache=True, parallel=True)
def _fill_hpsi_group_contributions(
    basis_keys: np.ndarray,
    coefficients: np.ndarray,
    x_mask: np.uint64,
    group_start: np.int64,
    group_stop: np.int64,
    term_z_masks: np.ndarray,
    term_coeffs: np.ndarray,
    out_keys: np.ndarray,
    out_amplitudes: np.ndarray,
    out_offset: np.int64,
) -> None:
    n_basis = basis_keys.shape[0]
    for i in nb.prange(n_basis):
        key = basis_keys[i]
        matrix_element = np.complex128(0.0 + 0.0j)
        for t in range(group_start, group_stop):
            coeff = term_coeffs[t]
            if _parity_u64(term_z_masks[t] & key) == 1:
                coeff = -coeff
            matrix_element += coeff
        out_keys[out_offset + i] = key ^ x_mask
        out_amplitudes[out_offset + i] = matrix_element * coefficients[i]


def _validate_basis_and_coefficients(
    basis_bitstrings: np.ndarray,
    coefficients: np.ndarray,
    *,
    num_qubits: int,
    validate_basis_bits: bool,
    normalize_coefficients: bool,
    logical_basis_bit_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    basis_raw = np.asarray(basis_bitstrings)
    basis = np.asarray(basis_raw, dtype=np.uint8)
    if basis.ndim != 2:
        raise ValueError("basis_bitstrings must be a 2D array.")
    if basis.shape[0] == 0:
        raise ValueError("basis_bitstrings must contain at least one row.")
    if basis.shape[1] != num_qubits:
        raise ValueError(
            f"basis_bitstrings has {basis.shape[1]} bits, but Hamiltonian has "
            f"{num_qubits} qubits."
        )
    if validate_basis_bits and not np.all((basis_raw == 0) | (basis_raw == 1)):
        raise ValueError("basis_bitstrings must contain only 0/1 values.")
    basis = np.ascontiguousarray(basis, dtype=np.uint8)

    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 1:
        raise ValueError("coefficients must be a 1D array.")
    if coeffs.shape[0] != basis.shape[0]:
        raise ValueError("coefficients length must match basis_bitstrings rows.")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must be finite.")

    coeff_norm = float(np.linalg.norm(coeffs))
    if coeff_norm <= 0.0 or not math.isfinite(coeff_norm):
        raise ValueError("coefficients must have finite nonzero norm.")
    if normalize_coefficients:
        coeffs = coeffs / coeff_norm
    elif not np.isclose(coeff_norm, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "coefficients are not normalized. Pass normalize_coefficients=True "
            "or normalize them before calling this function."
        )

    basis_keys = _pack_logical_basis_uint64(basis, bit_order=logical_basis_bit_order)
    if np.unique(basis_keys).shape[0] != int(basis_keys.shape[0]):
        raise ValueError("basis_bitstrings contains duplicate computational basis states.")

    return basis, np.ascontiguousarray(coeffs), np.ascontiguousarray(basis_keys), coeff_norm


def _estimate_contribution_workspace_gib(n_contributions: int) -> float:
    # Conservative peak estimate for the current sort/reduce implementation:
    # original keys/amplitudes, argsort indices, sorted copies, and reduced
    # support arrays can briefly coexist.
    return float(n_contributions * 80 / (1024**3))


def _as_compiled_hamiltonian(
    hamiltonian: Any,
    *,
    num_qubits: int | None,
    coefficient_cutoff: float,
    pauli_label_convention: str,
    require_real_pauli_coefficients: bool,
) -> CompiledPauliHamiltonian:
    if isinstance(hamiltonian, CompiledPauliHamiltonian):
        return hamiltonian
    return compile_pauli_hamiltonian(
        hamiltonian,
        num_qubits=num_qubits,
        coefficient_cutoff=coefficient_cutoff,
        pauli_label_convention=pauli_label_convention,
        require_real_pauli_coefficients=require_real_pauli_coefficients,
    )


def compute_full_hamiltonian_variance(
    hamiltonian: Any,
    basis_bitstrings: np.ndarray,
    coefficients: np.ndarray,
    *,
    num_qubits: int | None = None,
    coefficient_cutoff: float = 1e-12,
    amplitude_cutoff: float = 0.0,
    pauli_label_convention: str = "qiskit",
    logical_basis_bit_order: str = "qiskit",
    require_real_pauli_coefficients: bool = True,
    validate_basis_bits: bool = True,
    normalize_coefficients: bool = True,
    max_workspace_gib: float | None = None,
    reported_energy: float | None = None,
) -> EnergyVarianceResult:
    """Compute full-Hamiltonian variance for a state represented in a basis.

    This applies the full Pauli Hamiltonian to the SQD state and keeps output
    amplitudes outside the projected basis.  Therefore the returned variance is
    suitable for energy-variance extrapolation; it is not merely the projected
    diagonalizer residual.
    """
    amplitude_cutoff = float(amplitude_cutoff)
    if amplitude_cutoff < 0.0:
        raise ValueError("amplitude_cutoff must be nonnegative.")
    coefficient_cutoff = float(coefficient_cutoff)
    if coefficient_cutoff < 0.0:
        raise ValueError("coefficient_cutoff must be nonnegative.")

    compiled = _as_compiled_hamiltonian(
        hamiltonian,
        num_qubits=num_qubits,
        coefficient_cutoff=coefficient_cutoff,
        pauli_label_convention=pauli_label_convention,
        require_real_pauli_coefficients=require_real_pauli_coefficients,
    )
    basis, coeffs, basis_keys, coeff_norm = _validate_basis_and_coefficients(
        basis_bitstrings,
        coefficients,
        num_qubits=compiled.num_qubits,
        validate_basis_bits=validate_basis_bits,
        normalize_coefficients=normalize_coefficients,
        logical_basis_bit_order=logical_basis_bit_order,
    )

    n_basis = int(basis.shape[0])
    n_groups = int(compiled.group_x_masks.shape[0])
    n_contributions = int(n_basis * n_groups)
    estimated_workspace_gib = _estimate_contribution_workspace_gib(n_contributions)
    if max_workspace_gib is not None and estimated_workspace_gib > float(max_workspace_gib):
        raise MemoryError(
            "Estimated energy-variance workspace exceeds max_workspace_gib: "
            f"{estimated_workspace_gib:.3f} GiB > {float(max_workspace_gib):.3f} GiB."
        )

    if n_groups == 0:
        reported_energy_difference = (
            None if reported_energy is None else -float(reported_energy)
        )
        return EnergyVarianceResult(
            energy=0.0,
            energy_imag_abs=0.0,
            h2_expectation=0.0,
            variance=0.0,
            std=0.0,
            coefficient_norm_before_normalization=coeff_norm,
            n_qubits=int(compiled.num_qubits),
            n_basis=n_basis,
            n_hamiltonian_terms=0,
            n_hamiltonian_groups=0,
            n_hpsi_contributions=0,
            n_hpsi_support=0,
            inside_support_norm2=0.0,
            outside_support_norm2=0.0,
            outside_support_fraction=0.0,
            estimated_contribution_workspace_gib=estimated_workspace_gib,
            coefficient_cutoff=coefficient_cutoff,
            amplitude_cutoff=amplitude_cutoff,
            pauli_label_convention=str(pauli_label_convention),
            logical_basis_bit_order=str(logical_basis_bit_order),
            reported_energy=None if reported_energy is None else float(reported_energy),
            reported_energy_difference=reported_energy_difference,
        )

    all_keys = np.empty(n_contributions, dtype=np.uint64)
    all_amplitudes = np.empty(n_contributions, dtype=np.complex128)

    term_coeffs = np.ascontiguousarray(compiled.term_coeffs_complex, dtype=np.complex128)
    term_z_masks = np.ascontiguousarray(compiled.term_z_masks, dtype=np.uint64)
    group_offsets = np.asarray(compiled.group_offsets, dtype=np.int64)
    group_x_masks = np.asarray(compiled.group_x_masks, dtype=np.uint64)

    for group_index in range(n_groups):
        offset = int(group_index * n_basis)
        _fill_hpsi_group_contributions(
            basis_keys,
            coeffs,
            np.uint64(group_x_masks[group_index]),
            np.int64(group_offsets[group_index]),
            np.int64(group_offsets[group_index + 1]),
            term_z_masks,
            term_coeffs,
            all_keys,
            all_amplitudes,
            np.int64(offset),
        )

    order = np.argsort(all_keys)
    sorted_keys = all_keys[order]
    sorted_amplitudes = all_amplitudes[order]
    unique_keys, first_indices = np.unique(sorted_keys, return_index=True)
    hpsi_amplitudes = np.add.reduceat(sorted_amplitudes, first_indices)

    if amplitude_cutoff > 0.0:
        keep = np.abs(hpsi_amplitudes) > amplitude_cutoff
        unique_keys = unique_keys[keep]
        hpsi_amplitudes = hpsi_amplitudes[keep]

    positions = np.searchsorted(unique_keys, basis_keys)
    in_range = positions < unique_keys.shape[0]
    inside = np.zeros(n_basis, dtype=bool)
    in_range_indices = np.nonzero(in_range)[0]
    if len(in_range_indices) > 0:
        matched = unique_keys[positions[in_range_indices]] == basis_keys[in_range_indices]
        inside[in_range_indices[matched]] = True
    hpsi_on_basis = np.zeros(n_basis, dtype=np.complex128)
    hpsi_on_basis[inside] = hpsi_amplitudes[positions[inside]]

    energy_complex = np.vdot(coeffs, hpsi_on_basis)
    energy = float(np.real(energy_complex))
    h2_expectation = float(np.real(np.vdot(hpsi_amplitudes, hpsi_amplitudes)))
    inside_support_norm2 = float(np.real(np.vdot(hpsi_on_basis, hpsi_on_basis)))
    outside_support_norm2 = max(0.0, h2_expectation - inside_support_norm2)
    outside_support_fraction = (
        outside_support_norm2 / h2_expectation if h2_expectation > 0.0 else 0.0
    )
    variance = max(0.0, h2_expectation - energy * energy)
    reported_energy_difference = (
        None if reported_energy is None else energy - float(reported_energy)
    )

    return EnergyVarianceResult(
        energy=energy,
        energy_imag_abs=float(abs(np.imag(energy_complex))),
        h2_expectation=h2_expectation,
        variance=variance,
        std=float(math.sqrt(variance)),
        coefficient_norm_before_normalization=coeff_norm,
        n_qubits=int(compiled.num_qubits),
        n_basis=n_basis,
        n_hamiltonian_terms=int(compiled.n_compiled_terms),
        n_hamiltonian_groups=n_groups,
        n_hpsi_contributions=n_contributions,
        n_hpsi_support=int(unique_keys.shape[0]),
        inside_support_norm2=inside_support_norm2,
        outside_support_norm2=outside_support_norm2,
        outside_support_fraction=float(outside_support_fraction),
        estimated_contribution_workspace_gib=estimated_workspace_gib,
        coefficient_cutoff=coefficient_cutoff,
        amplitude_cutoff=amplitude_cutoff,
        pauli_label_convention=str(pauli_label_convention),
        logical_basis_bit_order=str(logical_basis_bit_order),
        reported_energy=None if reported_energy is None else float(reported_energy),
        reported_energy_difference=reported_energy_difference,
    )


def _result_field(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        return result[name]
    return getattr(result, name)


def _first_result_field(result: Any, names: Sequence[str]) -> Any:
    errors: list[str] = []
    for name in names:
        try:
            return _result_field(result, name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise KeyError(
        "Could not find any of these result fields: "
        + ", ".join(str(name) for name in names)
        + ". Errors: "
        + "; ".join(errors)
    )


def compute_full_hamiltonian_variance_from_sqd_result(
    result: Any,
    hamiltonian: Any,
    *,
    basis_field: str | None = None,
    coefficients_field: str | None = None,
    **kwargs: Any,
) -> EnergyVarianceResult:
    """Compute variance from a CodeSpaceRecoveryResult-like object or dict."""
    reported_energy = kwargs.pop("reported_energy", None)
    if reported_energy is None:
        try:
            reported_energy = _first_result_field(result, ("best_energy", "energy"))
        except Exception:
            reported_energy = None
    basis = (
        _result_field(result, basis_field)
        if basis_field is not None
        else _first_result_field(result, ("best_logical_basis", "logical_basis", "basis"))
    )
    coeffs = (
        _result_field(result, coefficients_field)
        if coefficients_field is not None
        else _first_result_field(result, ("best_coefficients", "coefficients"))
    )
    return compute_full_hamiltonian_variance(
        hamiltonian,
        basis,
        coeffs,
        reported_energy=reported_energy,
        **kwargs,
    )


def make_energy_variance_csv_row(
    variance_result: EnergyVarianceResult | Mapping[str, Any],
    **metadata: Any,
) -> dict[str, Any]:
    """Return one CSV-friendly row with user metadata first."""
    result_dict = (
        variance_result.to_dict()
        if isinstance(variance_result, EnergyVarianceResult)
        else dict(variance_result)
    )
    return {**metadata, **result_dict}


def write_energy_variance_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    append: bool = False,
) -> None:
    """Write or append energy-variance rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    file_exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append or not file_exists:
            writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "ALGORITHM_VERSION",
    "MODULE_VERSION",
    "EnergyVarianceResult",
    "compute_full_hamiltonian_variance",
    "compute_full_hamiltonian_variance_from_sqd_result",
    "make_energy_variance_csv_row",
    "write_energy_variance_csv",
]
