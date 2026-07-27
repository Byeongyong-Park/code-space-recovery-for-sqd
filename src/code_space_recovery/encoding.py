"""Encode logical Pauli Hamiltonians into pair-code form.

Used by the code-space recovery workflow and encoded SqDRIFT circuit generation.

Logical-to-pair code:
    |0>_L -> |01>
    |1>_L -> |10>

Pauli representation on the code space, written in pair order:
    I -> II
    X -> XX
    Y -> YX
    Z -> ZI

The default `mode="qiskit_label"` is intended for Qiskit SparsePauliOp labels.
Qiskit Pauli labels are displayed in descending qubit-index order. Therefore,
for one logical qubit the label "Z" is encoded as the label "ZI", which means
Z acts on the first character of the displayed two-bit pair. Equivalently, if
the physical qubits are indexed (q0, q1), the code pair |01>, |10> is read in
Qiskit's displayed bitstring order (q1 q0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
from qiskit.quantum_info import SparsePauliOp

PauliEncodingMode = Literal["qiskit_label", "qiskit_qubit_index"]

# Pair strings are written in pair tensor/display order: first rail, second rail.
DEFAULT_PAIR_PAULI_MAP: dict[str, str] = {
    "I": "II",
    "X": "XX",
    "Y": "YX",
    "Z": "ZI",
}


@dataclass(frozen=True)
class EncodingMetadata:
    """Small metadata object documenting an encoded Hamiltonian."""

    input_num_qubits: int
    output_num_qubits: int
    input_num_terms: int
    output_num_terms: int
    mode: PauliEncodingMode
    pair_pauli_map: dict[str, str]
    simplify: bool
    atol: float


def _validate_mode(mode: str) -> PauliEncodingMode:
    if mode not in {"qiskit_label", "qiskit_qubit_index"}:
        raise ValueError(
            "mode must be either 'qiskit_label' or 'qiskit_qubit_index', "
            f"got {mode!r}."
        )
    return mode  # type: ignore[return-value]


def _validate_pair_pauli_map(pair_pauli_map: Mapping[str, str]) -> dict[str, str]:
    required = {"I", "X", "Y", "Z"}
    keys = set(pair_pauli_map)
    if keys != required:
        raise ValueError(
            "pair_pauli_map must have exactly the keys {'I', 'X', 'Y', 'Z'}, "
            f"got {sorted(keys)!r}."
        )

    out: dict[str, str] = {}
    for key, value in pair_pauli_map.items():
        if not isinstance(value, str) or len(value) != 2:
            raise ValueError(
                f"pair_pauli_map[{key!r}] must be a length-2 Pauli string, "
                f"got {value!r}."
            )
        if any(ch not in required for ch in value):
            raise ValueError(
                f"pair_pauli_map[{key!r}] contains a non-Pauli character: {value!r}."
            )
        out[str(key)] = str(value)
    return out


def encode_pauli_label(
    label: str,
    *,
    mode: PauliEncodingMode = "qiskit_label",
    pair_pauli_map: Mapping[str, str] | None = None,
) -> str:
    """Encode one Pauli label from n logical qubits to 2n physical qubits.

    Parameters
    ----------
    label:
        Pauli label containing only I, X, Y, Z.
    mode:
        - "qiskit_label": expand the displayed Qiskit label character by
          character. This is the recommended default when the input and output
          are both Qiskit SparsePauliOp objects and encoded bitstrings are read
          in Qiskit's displayed order.
        - "qiskit_qubit_index": interpret input labels using Qiskit's qubit
          indexing convention and map logical qubit q to physical qubits
          (2q, 2q+1). In this mode, the first character of the pair map acts on
          physical qubit 2q and the second on physical qubit 2q+1.
    pair_pauli_map:
        Mapping for logical single-qubit Paulis. Defaults to
        I->II, X->XX, Y->YX, Z->ZI.

    Returns
    -------
    str
        Encoded 2n-qubit Pauli label.
    """
    mode = _validate_mode(mode)
    pair_map = _validate_pair_pauli_map(pair_pauli_map or DEFAULT_PAIR_PAULI_MAP)

    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label).__name__}.")
    if len(label) == 0:
        raise ValueError("label must be non-empty.")
    if any(ch not in {"I", "X", "Y", "Z"} for ch in label):
        raise ValueError(f"label contains a non-Pauli character: {label!r}.")

    if mode == "qiskit_label":
        return "".join(pair_map[ch] for ch in label)

    # Qiskit label position pos acts on qubit q = n - 1 - pos.
    # We explicitly place the encoded pair on physical qubits (2q, 2q+1).
    n_logical = len(label)
    n_physical = 2 * n_logical
    encoded_chars = ["I"] * n_physical

    for pos, ch in enumerate(label):
        logical_q = n_logical - 1 - pos
        pair = pair_map[ch]

        physical_q0 = 2 * logical_q
        physical_q1 = 2 * logical_q + 1

        encoded_chars[n_physical - 1 - physical_q0] = pair[0]
        encoded_chars[n_physical - 1 - physical_q1] = pair[1]

    return "".join(encoded_chars)


def encode_sparse_pauliop(
    hamiltonian: SparsePauliOp,
    *,
    mode: PauliEncodingMode = "qiskit_label",
    pair_pauli_map: Mapping[str, str] | None = None,
    simplify: bool = True,
    atol: float = 1e-12,
) -> SparsePauliOp:
    """Encode a SparsePauliOp Hamiltonian by replacing each Pauli label termwise.

    Coefficients are copied unchanged. Duplicate encoded labels are merged when
    `simplify=True`.
    """
    if not isinstance(hamiltonian, SparsePauliOp):
        raise TypeError(
            "hamiltonian must be a qiskit.quantum_info.SparsePauliOp object."
        )
    mode = _validate_mode(mode)
    pair_map = _validate_pair_pauli_map(pair_pauli_map or DEFAULT_PAIR_PAULI_MAP)
    if not isinstance(simplify, bool):
        raise TypeError(f"simplify must be bool, got {type(simplify).__name__}.")
    atol = float(atol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be a finite non-negative float, got {atol}.")

    input_labels = list(hamiltonian.paulis.to_labels())
    coeffs = np.asarray(hamiltonian.coeffs, dtype=complex)
    n_logical = int(hamiltonian.num_qubits)

    if len(input_labels) == 0:
        return SparsePauliOp.from_list([("I" * (2 * n_logical), 0.0)])

    encoded_terms = [
        (
            encode_pauli_label(
                label,
                mode=mode,
                pair_pauli_map=pair_map,
            ),
            coeff,
        )
        for label, coeff in zip(input_labels, coeffs, strict=True)
    ]

    encoded = SparsePauliOp.from_list(encoded_terms)
    if simplify:
        encoded = encoded.simplify(atol=atol)
    return encoded


def encode_sparse_pauliop_with_metadata(
    hamiltonian: SparsePauliOp,
    *,
    mode: PauliEncodingMode = "qiskit_label",
    pair_pauli_map: Mapping[str, str] | None = None,
    simplify: bool = True,
    atol: float = 1e-12,
) -> tuple[SparsePauliOp, EncodingMetadata]:
    """Same as `encode_sparse_pauliop`, but also return metadata.
    """
    pair_map = _validate_pair_pauli_map(pair_pauli_map or DEFAULT_PAIR_PAULI_MAP)
    encoded = encode_sparse_pauliop(
        hamiltonian,
        mode=mode,
        pair_pauli_map=pair_map,
        simplify=simplify,
        atol=atol,
    )
    metadata = EncodingMetadata(
        input_num_qubits=int(hamiltonian.num_qubits),
        output_num_qubits=int(encoded.num_qubits),
        input_num_terms=len(hamiltonian),
        output_num_terms=len(encoded),
        mode=mode,
        pair_pauli_map=dict(pair_map),
        simplify=simplify,
        atol=float(atol),
    )
    return encoded, metadata


def summarize_encoding(
    hamiltonian: SparsePauliOp,
    encoded_hamiltonian: SparsePauliOp,
    *,
    mode: PauliEncodingMode,
) -> dict[str, Any]:
    """Summarize an encoding that uses ``DEFAULT_PAIR_PAULI_MAP``.

    Use ``encode_sparse_pauliop_with_metadata`` to record a custom pair map.
    """
    return {
        "input_num_qubits": int(hamiltonian.num_qubits),
        "output_num_qubits": int(encoded_hamiltonian.num_qubits),
        "input_num_terms": int(len(hamiltonian)),
        "output_num_terms": int(len(encoded_hamiltonian)),
        "mode": mode,
        "pair_pauli_map": dict(DEFAULT_PAIR_PAULI_MAP),
    }
