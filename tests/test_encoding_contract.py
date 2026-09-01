"""Executable contracts for the pair-code Hamiltonian/state encoding."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from code_space_recovery.encoding import (
    DEFAULT_PAIR_PAULI_MAP,
    encode_pauli_label,
    encode_sparse_pauliop,
    encode_sparse_pauliop_with_metadata,
    summarize_encoding,
)
from code_space_recovery.state_encoding import (
    encode_state_preparation_circuit_pair_code,
)


EXPECTED_PAIR_PAULI_MAP = {
    "I": "II",
    "X": "XX",
    "Y": "YX",
    "Z": "ZI",
}


def _pair_code_isometry(n_logical: int) -> np.ndarray:
    """Return V with V|x> = |enc(x)> in Qiskit's displayed-bit order."""
    isometry = np.zeros((2 ** (2 * n_logical), 2**n_logical), dtype=complex)
    for logical_index in range(2**n_logical):
        logical_label = format(logical_index, f"0{n_logical}b")
        encoded_label = "".join(
            "01" if bit == "0" else "10" for bit in logical_label
        )
        isometry[int(encoded_label, 2), logical_index] = 1.0
    return isometry


def test_default_pair_pauli_map_is_the_documented_map() -> None:
    assert DEFAULT_PAIR_PAULI_MAP == EXPECTED_PAIR_PAULI_MAP
    assert encode_pauli_label("IXYZ") == "IIXXYXZI"
    assert encode_pauli_label("IXYZ", mode="qiskit_label") == "IIXXYXZI"


def test_default_pair_pauli_map_cannot_change_the_canonical_baseline() -> None:
    with pytest.raises(TypeError):
        DEFAULT_PAIR_PAULI_MAP["Z"] = "IZ"  # type: ignore[index]

    assert encode_pauli_label("Z") == "ZI"
    with pytest.raises(ValueError, match="custom pair_pauli_map"):
        encode_pauli_label(
            "Z",
            pair_pauli_map={"I": "II", "X": "XX", "Y": "YX", "Z": "IZ"},
        )


def test_only_the_canonical_pair_pauli_map_is_accepted() -> None:
    assert (
        encode_pauli_label(
            "Z",
            pair_pauli_map=dict(EXPECTED_PAIR_PAULI_MAP),
        )
        == "ZI"
    )
    with pytest.raises(ValueError, match="custom pair_pauli_map"):
        encode_pauli_label(
            "Z",
            pair_pauli_map={"I": "II", "X": "XX", "Y": "XY", "Z": "IZ"},
        )


@pytest.mark.parametrize(
    ("logical_bit", "expected_encoded_bitstring"),
    [(0, "01"), (1, "10")],
)
def test_state_preparation_uses_displayed_pair_code(
    logical_bit: int,
    expected_encoded_bitstring: str,
) -> None:
    logical = QuantumCircuit(1)
    if logical_bit:
        logical.x(0)

    encoded, metadata = encode_state_preparation_circuit_pair_code(
        logical,
        return_metadata=True,
    )
    probabilities = Statevector.from_instruction(encoded).probabilities_dict()

    assert set(probabilities) == {expected_encoded_bitstring}
    assert probabilities[expected_encoded_bitstring] == pytest.approx(1.0)
    assert metadata.first_rail_qubits == (1,)
    assert metadata.second_rail_qubits == (0,)
    assert "0 -> displayed pair 01" in metadata.pair_code
    assert "1 -> displayed pair 10" in metadata.pair_code


def test_encoded_pauli_operator_intertwines_with_pair_code_isometry() -> None:
    """The encoded operator A' must satisfy A' V = V A on the code space."""
    logical = SparsePauliOp.from_list(
        [
            ("II", 0.17),
            ("XI", -0.40),
            ("YZ", 0.75),
            ("ZX", -0.25),
            ("YY", 0.11),
        ]
    )
    encoded = encode_sparse_pauliop(logical)
    isometry = _pair_code_isometry(logical.num_qubits)

    np.testing.assert_allclose(
        encoded.to_matrix() @ isometry,
        isometry @ logical.to_matrix(),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "bad_coefficient",
    [
        complex(np.nan, 0.0),
        complex(np.inf, 0.0),
        complex(0.0, np.nan),
        complex(0.0, np.inf),
    ],
)
def test_encoding_rejects_nonfinite_hamiltonian_coefficients_before_output(
    bad_coefficient: complex,
) -> None:
    logical = SparsePauliOp.from_list([("Z", bad_coefficient)])

    with pytest.raises(ValueError, match="hamiltonian coefficients must be finite"):
        encode_sparse_pauliop(logical)


def test_encoding_metadata_records_the_only_supported_mode_and_map() -> None:
    logical = SparsePauliOp.from_list([("X", 2.0), ("Z", -0.5)])
    encoded, metadata = encode_sparse_pauliop_with_metadata(
        logical,
        mode="qiskit_label",
    )

    assert encoded.paulis.to_labels() == ["XX", "ZI"]
    assert metadata.mode == "qiskit_label"
    assert metadata.pair_pauli_map == EXPECTED_PAIR_PAULI_MAP
    assert metadata.input_num_qubits == 1
    assert metadata.output_num_qubits == 2


@pytest.mark.parametrize(
    "entry_point", ["label", "operator", "metadata", "summary"]
)
def test_legacy_qiskit_qubit_index_mode_fails_with_migration_guidance(
    entry_point: str,
) -> None:
    logical = SparsePauliOp.from_list([("Z", 1.0)])

    with pytest.raises(ValueError) as caught:
        if entry_point == "label":
            encode_pauli_label("Z", mode="qiskit_qubit_index")
        elif entry_point == "operator":
            encode_sparse_pauliop(logical, mode="qiskit_qubit_index")
        elif entry_point == "metadata":
            encode_sparse_pauliop_with_metadata(
                logical,
                mode="qiskit_qubit_index",
            )
        else:
            summarize_encoding(
                logical,
                encode_sparse_pauliop(logical),
                mode="qiskit_qubit_index",
            )

    message = str(caught.value).lower()
    assert "qiskit_qubit_index" in message
    assert "qiskit_label" in message
    assert any(word in message for word in ("migrat", "use", "only", "supported"))
