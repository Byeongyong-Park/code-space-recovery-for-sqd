"""Validation and compiled-Hamiltonian provenance contracts for variance analysis."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from code_space_recovery.diagonalization import compile_pauli_hamiltonian
from code_space_recovery.energy_variance import (
    EnergyVarianceResult,
    compute_full_hamiltonian_variance,
)


_BASIS = np.array([[0]], dtype=np.uint8)
_COEFFICIENTS = np.array([1.0 + 0.0j], dtype=np.complex128)
_RAW_HAMILTONIAN = [("Z", 1.0)]


def test_new_provenance_fields_preserve_legacy_positional_binding() -> None:
    result = EnergyVarianceResult(
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1,
        1,
        1,
        1,
        1,
        1,
        1.0,
        0.0,
        0.0,
        0.0,
        1e-12,
        0.0,
        "qiskit",
        "qiskit",
        "v1.0",
        -1.25,
        0.125,
    )

    assert result.module_version == "v1.0"
    assert result.reported_energy == pytest.approx(-1.25)
    assert result.reported_energy_difference == pytest.approx(0.125)
    assert result.package_version
    assert result.algorithm_version


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
@pytest.mark.parametrize("keyword", ["coefficient_cutoff", "amplitude_cutoff"])
def test_variance_cutoffs_must_be_finite_and_nonnegative(
    keyword: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        compute_full_hamiltonian_variance(
            _RAW_HAMILTONIAN,
            _BASIS,
            _COEFFICIENTS,
            num_qubits=1,
            **{keyword: value},
        )


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_variance_workspace_limit_must_be_none_or_finite_and_positive(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="None or finite and > 0"):
        compute_full_hamiltonian_variance(
            _RAW_HAMILTONIAN,
            _BASIS,
            _COEFFICIENTS,
            num_qubits=1,
            max_workspace_gib=value,
        )


@pytest.mark.parametrize("keyword", ["coefficient_cutoff", "amplitude_cutoff"])
@pytest.mark.parametrize("value", [True, False, "0.0", 0.0 + 0.0j])
def test_variance_cutoffs_reject_bool_and_coerced_values(
    keyword: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match="real numeric value"):
        compute_full_hamiltonian_variance(
            _RAW_HAMILTONIAN,
            _BASIS,
            _COEFFICIENTS,
            num_qubits=1,
            **{keyword: value},
        )


@pytest.mark.parametrize("value", [True, False, "1.0", 1.0 + 0.0j])
def test_variance_workspace_rejects_bool_and_coerced_values(value: object) -> None:
    with pytest.raises(TypeError, match="real numeric value"):
        compute_full_hamiltonian_variance(
            _RAW_HAMILTONIAN,
            _BASIS,
            _COEFFICIENTS,
            num_qubits=1,
            max_workspace_gib=value,
        )


@pytest.mark.parametrize(
    "value",
    [True, False, "-1.0", -1.0 + 0.0j, float("nan"), float("inf")],
)
def test_reported_energy_must_be_a_finite_real_without_coercion(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="reported_energy"):
        compute_full_hamiltonian_variance(
            _RAW_HAMILTONIAN,
            _BASIS,
            _COEFFICIENTS,
            num_qubits=1,
            reported_energy=value,
        )


def test_variance_rejects_non_numeric_basis_and_coefficients_before_cast() -> None:
    with pytest.raises(TypeError, match="basis_bitstrings"):
        compute_full_hamiltonian_variance(
            _RAW_HAMILTONIAN,
            np.array([["0"]]),
            _COEFFICIENTS,
            num_qubits=1,
        )
    with pytest.raises(TypeError, match="coefficients"):
        compute_full_hamiltonian_variance(
            _RAW_HAMILTONIAN,
            _BASIS,
            np.array(["1.0"]),
            num_qubits=1,
        )


def test_variance_exact_binary_float_basis_remains_supported() -> None:
    result = compute_full_hamiltonian_variance(
        _RAW_HAMILTONIAN,
        np.array([[0.0]]),
        np.array([1.0]),
        num_qubits=1,
    )

    assert result.energy == pytest.approx(1.0)


@pytest.mark.parametrize(
    "keyword",
    [
        "require_real_pauli_coefficients",
        "validate_basis_bits",
        "normalize_coefficients",
    ],
)
def test_variance_boolean_flags_reject_truthy_strings(keyword: str) -> None:
    with pytest.raises(TypeError, match="must be bool"):
        compute_full_hamiltonian_variance(
            _RAW_HAMILTONIAN,
            _BASIS,
            _COEFFICIENTS,
            num_qubits=1,
            **{keyword: "False"},
        )


def test_raw_hamiltonian_omitted_settings_keep_historical_defaults() -> None:
    result = compute_full_hamiltonian_variance(
        _RAW_HAMILTONIAN,
        _BASIS,
        _COEFFICIENTS,
        num_qubits=1,
    )

    assert result.coefficient_cutoff == pytest.approx(1e-12)
    assert result.pauli_label_convention == "qiskit"
    assert result.energy == pytest.approx(1.0)
    assert result.variance == pytest.approx(0.0)


def test_compiled_hamiltonian_omitted_settings_inherit_actual_metadata() -> None:
    compiled = compile_pauli_hamiltonian(
        _RAW_HAMILTONIAN,
        num_qubits=1,
        coefficient_cutoff=0.125,
        pauli_label_convention="little_endian",
    )

    result = compute_full_hamiltonian_variance(
        compiled,
        _BASIS,
        _COEFFICIENTS,
    )

    assert result.n_qubits == compiled.num_qubits
    assert result.coefficient_cutoff == compiled.coefficient_cutoff
    assert result.pauli_label_convention == compiled.pauli_label_convention
    assert result.energy == pytest.approx(1.0)
    assert result.variance == pytest.approx(0.0)


def test_matching_explicit_compiled_settings_are_accepted() -> None:
    compiled = compile_pauli_hamiltonian(
        _RAW_HAMILTONIAN,
        num_qubits=1,
        coefficient_cutoff=0.125,
        pauli_label_convention="little_endian",
    )

    result = compute_full_hamiltonian_variance(
        compiled,
        _BASIS,
        _COEFFICIENTS,
        num_qubits=1,
        coefficient_cutoff=0.125,
        pauli_label_convention="internal",
    )

    assert result.coefficient_cutoff == compiled.coefficient_cutoff
    assert result.pauli_label_convention == "little_endian"


@pytest.mark.parametrize(
    ("kwargs", "setting_name"),
    [
        ({"num_qubits": 2}, "num_qubits"),
        ({"coefficient_cutoff": 0.25}, "coefficient_cutoff"),
        ({"pauli_label_convention": "qiskit"}, "pauli_label_convention"),
    ],
)
def test_explicit_settings_must_not_conflict_with_compiled_hamiltonian(
    kwargs: dict[str, object],
    setting_name: str,
) -> None:
    compiled = compile_pauli_hamiltonian(
        _RAW_HAMILTONIAN,
        num_qubits=1,
        coefficient_cutoff=0.125,
        pauli_label_convention="little_endian",
    )

    with pytest.raises(ValueError, match=setting_name):
        compute_full_hamiltonian_variance(
            compiled,
            _BASIS,
            _COEFFICIENTS,
            **kwargs,
        )


def test_variance_rejects_nonfinite_coefficients_in_compiled_hamiltonian() -> None:
    compiled = compile_pauli_hamiltonian(
        _RAW_HAMILTONIAN,
        num_qubits=1,
    )
    invalid_compiled = replace(
        compiled,
        term_coeffs_complex=np.array([complex(float("nan"), 0.0)]),
    )

    with pytest.raises(ValueError, match="Compiled Hamiltonian coefficients must be finite"):
        compute_full_hamiltonian_variance(
            invalid_compiled,
            _BASIS,
            _COEFFICIENTS,
        )


@pytest.mark.parametrize(
    ("field", "invalid_mask"),
    [
        ("group_x_masks", np.array([2], dtype=np.uint64)),
        ("term_z_masks", np.array([2], dtype=np.uint64)),
    ],
)
def test_variance_rejects_compiled_masks_outside_num_qubits(
    field: str,
    invalid_mask: np.ndarray,
) -> None:
    compiled = compile_pauli_hamiltonian(_RAW_HAMILTONIAN, num_qubits=1)
    invalid_compiled = replace(compiled, **{field: invalid_mask})

    with pytest.raises(ValueError, match="masks contain bits outside num_qubits"):
        compute_full_hamiltonian_variance(
            invalid_compiled,
            _BASIS,
            _COEFFICIENTS,
        )


def test_variance_accepts_highest_valid_uint64_mask_and_rejects_next_bit() -> None:
    compiled = compile_pauli_hamiltonian(
        [([(62, "X")], 1.0)],
        num_qubits=63,
    )
    basis = np.zeros((1, 63), dtype=np.uint8)

    result = compute_full_hamiltonian_variance(
        compiled,
        basis,
        _COEFFICIENTS,
    )

    assert result.energy == pytest.approx(0.0)
    assert result.variance == pytest.approx(1.0)

    invalid_compiled = replace(
        compiled,
        group_x_masks=np.array([1 << 63], dtype=np.uint64),
    )
    with pytest.raises(ValueError, match="masks contain bits outside num_qubits"):
        compute_full_hamiltonian_variance(
            invalid_compiled,
            basis,
            _COEFFICIENTS,
        )


@pytest.mark.parametrize(
    ("replacement", "error_type", "message"),
    [
        (
            {"group_x_masks": [np.uint64(0)]},
            TypeError,
            "group_x_masks must be a numpy.ndarray",
        ),
        (
            {"group_x_masks": np.array([0], dtype=np.int64)},
            TypeError,
            "group_x_masks must have dtype uint64",
        ),
        (
            {"term_coeffs_complex": np.array([1.0], dtype=np.complex64)},
            TypeError,
            "term_coeffs_complex must have dtype complex128",
        ),
        (
            {
                "group_x_masks": np.array([0, 1], dtype=np.uint64),
                "group_offsets": np.array([0, 0, 1], dtype=np.int64),
            },
            ValueError,
            "nonempty groups",
        ),
        (
            {"n_input_terms": 2, "n_compiled_terms": 2},
            ValueError,
            "n_compiled_terms does not match",
        ),
        (
            {"group_x_masks": np.array([[0]], dtype=np.uint64)},
            ValueError,
            "group_x_masks must be a 1D array",
        ),
    ],
)
def test_variance_rejects_invalid_compiled_array_structure(
    replacement: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    compiled = compile_pauli_hamiltonian(_RAW_HAMILTONIAN, num_qubits=1)
    invalid_compiled = replace(compiled, **replacement)

    with pytest.raises(error_type, match=message):
        compute_full_hamiltonian_variance(
            invalid_compiled,
            _BASIS,
            _COEFFICIENTS,
        )


@pytest.mark.parametrize(
    ("replacement", "error_type", "message"),
    [
        ({"is_real": 1}, TypeError, "is_real must be bool"),
        ({"is_real": False}, ValueError, "is_real is inconsistent"),
        (
            {"term_coeffs_real": np.array([2.0], dtype=np.float64)},
            ValueError,
            "real coefficient cache is inconsistent",
        ),
        (
            {"term_coeffs_real": np.array([float("nan")], dtype=np.float64)},
            ValueError,
            "real coefficient cache must be finite",
        ),
        (
            {"coefficient_cutoff": float("nan")},
            ValueError,
            "coefficient_cutoff must be finite",
        ),
        (
            {"pauli_label_convention": "unknown"},
            ValueError,
            "pauli_label_convention",
        ),
    ],
)
def test_variance_rejects_inconsistent_compiled_metadata_and_caches(
    replacement: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    compiled = compile_pauli_hamiltonian(_RAW_HAMILTONIAN, num_qubits=1)
    invalid_compiled = replace(compiled, **replacement)

    with pytest.raises(error_type, match=message):
        compute_full_hamiltonian_variance(
            invalid_compiled,
            _BASIS,
            _COEFFICIENTS,
        )


def test_compiled_nonhermitian_input_respects_require_real_contract() -> None:
    compiled = compile_pauli_hamiltonian(
        [("Z", 1.0j)],
        num_qubits=1,
        require_real_pauli_coefficients=False,
    )

    with pytest.raises(ValueError, match="require_real_pauli_coefficients=True"):
        compute_full_hamiltonian_variance(
            compiled,
            _BASIS,
            _COEFFICIENTS,
        )


@pytest.mark.parametrize("imaginary_part", [1.0, 1e-15])
def test_variance_rejects_nonhermitian_hamiltonian_even_when_real_check_disabled(
    imaginary_part: float,
) -> None:
    with pytest.raises(ValueError, match="Energy-variance.*Hermitian"):
        compute_full_hamiltonian_variance(
            [("X", complex(1.0, imaginary_part))],
            _BASIS,
            _COEFFICIENTS,
            num_qubits=1,
            require_real_pauli_coefficients=False,
        )


def test_variance_accepts_hermitian_y_with_complex_internal_coefficient() -> None:
    result = compute_full_hamiltonian_variance(
        [("Y", 1.0)],
        _BASIS,
        _COEFFICIENTS,
        num_qubits=1,
        require_real_pauli_coefficients=False,
    )

    assert result.energy == pytest.approx(0.0)
    assert result.variance == pytest.approx(1.0)


def test_variance_rejects_finite_inputs_whose_hamiltonian_action_overflows() -> None:
    compiled = compile_pauli_hamiltonian(
        [("I", 1e308), ("Z", 1e308)],
        num_qubits=1,
    )

    with pytest.raises(ValueError, match=r"H\|psi>.*finite"):
        compute_full_hamiltonian_variance(
            compiled,
            _BASIS,
            _COEFFICIENTS,
        )
