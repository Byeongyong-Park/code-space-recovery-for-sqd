"""Fail-fast contracts for Hamiltonian compilation and SqDRIFT preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from code_space_recovery.diagonalization import (
    ProjectedPauliCSRPRIMMEDiagonalizer,
    compile_pauli_hamiltonian,
)
from code_space_recovery.hamiltonians import expand_coefficients
from code_space_recovery.sqdrift import (
    _preprocess_sparse_pauli_hamiltonian,
    generate_sqdrift_logical_circuits,
)


@pytest.mark.parametrize(
    "coefficient",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        complex(1.0, float("nan")),
        complex(1.0, float("inf")),
    ],
)
def test_diagonalization_rejects_nonfinite_input_coefficients(
    coefficient: complex,
) -> None:
    with pytest.raises(ValueError, match="coefficients must be finite"):
        compile_pauli_hamiltonian(
            [("Z", coefficient)],
            num_qubits=1,
            require_real_pauli_coefficients=False,
        )


def test_diagonalization_rejects_nonfinite_merged_coefficient() -> None:
    with pytest.raises(ValueError, match="merged Pauli term"):
        compile_pauli_hamiltonian(
            [("Z", 1e308), ("Z", 1e308)],
            num_qubits=1,
        )


def test_diagonalization_finite_merge_is_numerically_unchanged() -> None:
    compiled = compile_pauli_hamiltonian(
        [("Z", 1.0), ("Z", -0.25), ("X", -0.5)],
        num_qubits=1,
    )

    assert compiled.n_input_terms == 3
    assert compiled.n_compiled_terms == 2
    np.testing.assert_allclose(compiled.term_coeffs_complex, [0.75, -0.5])


def test_one_shot_hamiltonian_requires_explicit_num_qubits_without_consuming() -> None:
    terms = iter([("Z", 1.0)])

    with pytest.raises(ValueError, match="num_qubits.*one-shot"):
        compile_pauli_hamiltonian(terms)

    assert next(terms) == ("Z", 1.0)


def test_one_shot_hamiltonian_with_explicit_num_qubits_is_consumed_once() -> None:
    terms = ((label, coefficient) for label, coefficient in [("Z", 1.0)])

    compiled = compile_pauli_hamiltonian(terms, num_qubits=1)

    assert compiled.n_input_terms == 1
    assert compiled.n_compiled_terms == 1
    np.testing.assert_allclose(compiled.term_coeffs_complex, [1.0])


def test_diagonalization_rejects_duplicate_cutoff_order_ambiguity() -> None:
    cutoff = 1e-12

    with pytest.raises(ValueError, match="ambiguous at the coefficient cutoff"):
        compile_pauli_hamiltonian(
            [("Z", 0.75 * cutoff), ("Z", 0.75 * cutoff)],
            num_qubits=1,
            coefficient_cutoff=cutoff,
        )


def test_duplicate_subcutoff_contributions_that_cancel_are_unambiguous() -> None:
    cutoff = 1e-12
    compiled = compile_pauli_hamiltonian(
        [("Z", 0.75 * cutoff), ("Z", -0.75 * cutoff)],
        num_qubits=1,
        coefficient_cutoff=cutoff,
    )

    assert compiled.n_compiled_terms == 0


@pytest.mark.parametrize("num_qubits", [True, 1.5, "1"])
def test_diagonalization_rejects_coerced_inferred_qubit_count(
    num_qubits: object,
) -> None:
    class InvalidHamiltonian:
        def __init__(self, value: object) -> None:
            self.num_qubits = value

        def to_list(self) -> list[tuple[str, float]]:
            return [("Z", 1.0)]

    with pytest.raises(TypeError, match="num_qubits"):
        compile_pauli_hamiltonian(InvalidHamiltonian(num_qubits))


@pytest.mark.parametrize("qubit", [True, 0.5, "0"])
def test_diagonalization_rejects_coerced_sparse_pauli_qubit_index(
    qubit: object,
) -> None:
    with pytest.raises(TypeError, match="qubit indices must be integers"):
        compile_pauli_hamiltonian(
            [([(qubit, "Z")], 1.0)],
            num_qubits=1,
        )


def test_projected_matrix_rejects_finite_term_accumulation_overflow() -> None:
    hamiltonian = [("I", 1e308), ("Z", 1e308)]
    diagonalizer = ProjectedPauliCSRPRIMMEDiagonalizer(
        hamiltonian,
        num_qubits=1,
        coefficient_cutoff=0.0,
        matrix_element_cutoff=0.0,
        ncv=None,
    )

    with pytest.raises(ValueError, match="matrix elements must remain finite"):
        diagonalizer(hamiltonian, np.array([[0]], dtype=np.uint8))


@pytest.mark.parametrize("warn_if_different", [False, True])
@pytest.mark.parametrize(
    "call_hamiltonian",
    [
        SparsePauliOp.from_list([("Z", 2.0)]),
        SparsePauliOp.from_list([("X", 1.0)]),
    ],
)
def test_diagonalizer_rejects_call_time_hamiltonian_mismatch_before_numerics(
    warn_if_different: bool,
    call_hamiltonian: SparsePauliOp,
) -> None:
    construction_hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])
    diagonalizer = ProjectedPauliCSRPRIMMEDiagonalizer(
        construction_hamiltonian,
        ncv=None,
        warn_if_hamiltonian_argument_differs=warn_if_different,
    )

    with pytest.raises(ValueError, match="call-time Hamiltonian does not match"):
        diagonalizer(
            call_hamiltonian,
            np.array([[0], [1]], dtype=np.uint8),
        )

    assert diagonalizer.last_stats is None


def test_semantically_identical_separate_hamiltonian_is_accepted() -> None:
    construction_hamiltonian = SparsePauliOp.from_list(
        [("Z", 0.75), ("X", -0.5)]
    )
    equivalent_hamiltonian = SparsePauliOp.from_list(
        [("X", -0.5), ("Z", 0.75)]
    )
    diagonalizer = ProjectedPauliCSRPRIMMEDiagonalizer(
        construction_hamiltonian,
        ncv=None,
    )

    result = diagonalizer(
        equivalent_hamiltonian,
        np.array([[0]], dtype=np.uint8),
    )

    assert result.energy == pytest.approx(0.75)


@pytest.mark.parametrize("imaginary_part", [1.0, 1e-15])
def test_projected_eigsh_rejects_nonhermitian_canonical_hamiltonian(
    imaginary_part: float,
) -> None:
    construction_hamiltonian = SparsePauliOp.from_list(
        [("X", complex(1.0, imaginary_part))]
    )

    with pytest.raises(ValueError, match="eigsh.*Hermitian"):
        ProjectedPauliCSRPRIMMEDiagonalizer(
            construction_hamiltonian,
            ncv=None,
            require_real_pauli_coefficients=False,
        )


def test_projected_eigsh_accepts_hermitian_y_with_complex_internal_coefficient() -> None:
    hamiltonian = SparsePauliOp.from_list([("Y", 1.0)])

    diagonalizer = ProjectedPauliCSRPRIMMEDiagonalizer(
        hamiltonian,
        ncv=None,
        require_real_pauli_coefficients=False,
    )

    assert diagonalizer.compiled.is_real is False


def test_projected_eigsh_checks_hermiticity_after_canonical_duplicate_merge() -> None:
    hamiltonian = [("Z", 1.0j), ("Z", -1.0j), ("X", 1.0)]

    diagonalizer = ProjectedPauliCSRPRIMMEDiagonalizer(
        hamiltonian,
        num_qubits=1,
        ncv=None,
        require_real_pauli_coefficients=False,
    )

    assert diagonalizer.compiled.n_compiled_terms == 1
    np.testing.assert_allclose(diagonalizer.compiled.term_coeffs_complex, [1.0])


def test_call_time_term_below_construction_cutoff_is_semantically_ignored() -> None:
    construction_hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])
    equivalent_hamiltonian = SparsePauliOp.from_list(
        [("Z", 1.0), ("X", 1e-13)]
    )
    diagonalizer = ProjectedPauliCSRPRIMMEDiagonalizer(
        construction_hamiltonian,
        coefficient_cutoff=1e-12,
        ncv=None,
    )

    result = diagonalizer(
        equivalent_hamiltonian,
        np.array([[0]], dtype=np.uint8),
    )

    assert result.energy == pytest.approx(1.0)


def test_exact_construction_object_uses_documented_identity_fast_path() -> None:
    hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])
    diagonalizer = ProjectedPauliCSRPRIMMEDiagonalizer(hamiltonian, ncv=None)
    hamiltonian.coeffs[0] = 2.0

    result = diagonalizer(hamiltonian, np.array([[0]], dtype=np.uint8))

    # The exact object is intentionally not re-read: the construction-time
    # compiled operator is used so the main recovery path has no extra compile.
    assert result.energy == pytest.approx(1.0)


def test_spawn_preserves_call_time_hamiltonian_contract() -> None:
    hamiltonian = SparsePauliOp.from_list([("Y", 1.0)])
    diagonalizer = ProjectedPauliCSRPRIMMEDiagonalizer(
        hamiltonian,
        ncv=None,
        require_real_pauli_coefficients=True,
    )
    spawned = diagonalizer.spawn()

    result = spawned(
        SparsePauliOp.from_list([("Y", 1.0)]),
        np.array([[0]], dtype=np.uint8),
    )
    assert result.energy == pytest.approx(0.0)

    with pytest.raises(ValueError, match="call-time Hamiltonian does not match"):
        spawned(
            SparsePauliOp.from_list([("Y", 2.0)]),
            np.array([[0]], dtype=np.uint8),
        )


@pytest.mark.parametrize(
    "coefficient",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        complex(1.0, float("nan")),
        complex(1.0, float("inf")),
    ],
)
def test_sqdrift_rejects_nonfinite_input_coefficients(
    coefficient: complex,
) -> None:
    hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])
    hamiltonian.coeffs[0] = coefficient

    with pytest.raises(ValueError, match="coefficients must be finite"):
        _preprocess_sparse_pauli_hamiltonian(hamiltonian)


@pytest.mark.parametrize("label", ["I", "Z"])
def test_sqdrift_rejects_nonfinite_duplicate_merge(label: str) -> None:
    hamiltonian = SparsePauliOp.from_list(
        [(label, 1e308), (label, 1e308)]
    )

    with pytest.raises(ValueError, match="Merged Hamiltonian coefficient"):
        _preprocess_sparse_pauli_hamiltonian(hamiltonian)


def test_sqdrift_rejects_duplicate_cutoff_order_ambiguity() -> None:
    cutoff = 1e-12
    hamiltonian = SparsePauliOp.from_list(
        [("Z", 0.75 * cutoff), ("Z", 0.75 * cutoff)]
    )

    with pytest.raises(ValueError, match="ambiguous at the coefficient cutoff"):
        _preprocess_sparse_pauli_hamiltonian(
            hamiltonian,
            coefficient_atol=cutoff,
        )


def test_sqdrift_accepts_duplicate_subcutoff_contributions_that_cancel() -> None:
    cutoff = 1e-12
    hamiltonian = SparsePauliOp.from_list(
        [("Z", 0.75 * cutoff), ("Z", -0.75 * cutoff)]
    )

    processed = _preprocess_sparse_pauli_hamiltonian(
        hamiltonian,
        coefficient_atol=cutoff,
    )

    assert processed["processed_labels"] == []
    assert processed["preprocessing_summary"]["num_processed_non_identity_terms"] == 0


def test_sqdrift_rejects_nonfinite_lambda() -> None:
    hamiltonian = SparsePauliOp.from_list(
        [("ZI", 1e308), ("IZ", 1e308)]
    )

    with pytest.raises(ValueError, match="qDRIFT lambda must be finite"):
        _preprocess_sparse_pauli_hamiltonian(hamiltonian)


def test_sqdrift_finite_preprocessing_is_numerically_unchanged() -> None:
    hamiltonian = SparsePauliOp.from_list(
        [("II", 0.25), ("ZI", 1.0), ("ZI", -0.25), ("IX", -0.5)]
    )

    processed = _preprocess_sparse_pauli_hamiltonian(hamiltonian)

    assert processed["identity_shift"] == pytest.approx(0.25)
    assert processed["processed_labels"] == ["IX", "ZI"]
    np.testing.assert_allclose(processed["signed_coefficients"], [-0.5, 0.75])
    assert processed["lambda"] == pytest.approx(1.25)
    np.testing.assert_allclose(processed["probabilities"], [0.4, 0.6])
    assert np.all(np.isfinite(processed["probabilities"]))
    assert float(np.sum(processed["probabilities"])) == pytest.approx(1.0)


@pytest.mark.parametrize("sort_terms", ["False", 0, 1, None])
def test_sqdrift_sort_terms_must_be_boolean(sort_terms: object) -> None:
    hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])

    with pytest.raises(TypeError, match="sort_terms must be bool"):
        _preprocess_sparse_pauli_hamiltonian(
            hamiltonian,
            sort_terms=sort_terms,
        )


def test_sqdrift_rejects_derived_evolution_time_overflow() -> None:
    hamiltonian = SparsePauliOp.from_list([("Z", 1e308)])

    with pytest.raises(ValueError, match="base step time must remain finite"):
        generate_sqdrift_logical_circuits(
            hamiltonian,
            K=2,
            N_R=1,
            N_seq=1,
            delta_t=1e308,
            include_k0=False,
            seed=7,
        )


@pytest.mark.parametrize("imaginary_part", [float("nan"), float("inf"), float("-inf")])
def test_expand_coefficients_rejects_nonfinite_imaginary_parts_before_cast(
    imaginary_part: float,
) -> None:
    values = np.array([complex(1.0, imaginary_part)], dtype=np.complex128)

    with pytest.raises(ValueError, match="must contain only finite values"):
        expand_coefficients(values, 1, name="couplings")


def test_expand_coefficients_preserves_finite_real_complex_values() -> None:
    values = np.array([complex(1.25, 0.0), complex(-0.5, 0.0)])

    expanded = expand_coefficients(values, 2, name="couplings")

    np.testing.assert_array_equal(expanded, np.array([1.25, -0.5]))
