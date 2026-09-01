"""Encode logical Pauli Hamiltonians into pair-code form.

Used by the code-space recovery workflow and encoded SqDRIFT circuit generation.

Logical-to-pair code:
    |0>_L -> |01>
    |1>_L -> |10>

Pauli representation on the code space, written in displayed pair order
``[first rail, second rail]``:
    I -> II
    X -> XX
    Y -> YX
    Z -> ZI

Only ``mode="qiskit_label"`` is supported. Qiskit Pauli labels are displayed
in descending qubit-index order, so each logical label character is expanded
directly to one adjacent displayed pair. For logical qubit ``q``, the first
displayed rail is physical qubit ``2q+1`` and the second is physical qubit
``2q``. Thus logical ``Z`` becomes displayed label ``ZI`` and acts on the
first rail, consistently with state preparation and recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
from qiskit.quantum_info import SparsePauliOp

try:  # package import
    from ._version import ALGORITHM_VERSION, PACKAGE_VERSION
except ImportError:  # pragma: no cover - flat-module compatibility
    from _version import ALGORITHM_VERSION, PACKAGE_VERSION  # type: ignore


PauliEncodingMode = Literal["qiskit_label"]

# Pair strings are written in pair tensor/display order: first rail, second rail.
_CANONICAL_PAIR_PAULI_MAP: Mapping[str, str] = MappingProxyType({
    "I": "II",
    "X": "XX",
    "Y": "YX",
    "Z": "ZI",
})
# Expose a separate read-only view so callers cannot mutate the package's
# private canonical baseline through the public constant.
DEFAULT_PAIR_PAULI_MAP: Mapping[str, str] = MappingProxyType(
    dict(_CANONICAL_PAIR_PAULI_MAP)
)


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
    package_version: str = PACKAGE_VERSION
    algorithm_version: str = ALGORITHM_VERSION


def _validate_mode(mode: str) -> PauliEncodingMode:
    if mode == "qiskit_qubit_index":
        raise ValueError(
            "mode='qiskit_qubit_index' was removed in code-space-recovery 2.0 "
            "because its physical-rail placement is incompatible with the package's "
            "state-encoding and recovery convention. Use mode='qiskit_label'. "
            "Use code-space-recovery 1.x only to read or reproduce a legacy "
            "physical-qubit-index artifact."
        )
    if mode != "qiskit_label":
        raise ValueError(f"mode must be 'qiskit_label', got {mode!r}.")
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
    if out != _CANONICAL_PAIR_PAULI_MAP:
        raise ValueError(
            "custom pair_pauli_map values are not supported in "
            "code-space-recovery 2.x because they can violate the package-wide "
            "state-encoding and recovery rail convention. Use the canonical map "
            "{'I': 'II', 'X': 'XX', 'Y': 'YX', 'Z': 'ZI'}."
        )
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
        Must be ``"qiskit_label"``. The displayed Qiskit label is expanded
        character by character. The first character of each pair map acts on
        physical qubit ``2q+1`` and the second on physical qubit ``2q``.
        The legacy ``"qiskit_qubit_index"`` mode was removed in version 2.0.
    pair_pauli_map:
        Optional explicit copy of the canonical mapping
        I->II, X->XX, Y->YX, Z->ZI. Other mappings are rejected because they
        are incompatible with the package-wide rail convention.

    Returns
    -------
    str
        Encoded 2n-qubit Pauli label.
    """
    mode = _validate_mode(mode)
    pair_map = _validate_pair_pauli_map(
        _CANONICAL_PAIR_PAULI_MAP if pair_pauli_map is None else pair_pauli_map
    )

    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label).__name__}.")
    if len(label) == 0:
        raise ValueError("label must be non-empty.")
    if any(ch not in {"I", "X", "Y", "Z"} for ch in label):
        raise ValueError(f"label contains a non-Pauli character: {label!r}.")

    return "".join(pair_map[ch] for ch in label)


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
    pair_map = _validate_pair_pauli_map(
        _CANONICAL_PAIR_PAULI_MAP if pair_pauli_map is None else pair_pauli_map
    )
    if not isinstance(simplify, bool):
        raise TypeError(f"simplify must be bool, got {type(simplify).__name__}.")
    atol = float(atol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be a finite non-negative float, got {atol}.")

    input_labels = list(hamiltonian.paulis.to_labels())
    coeffs = np.asarray(hamiltonian.coeffs, dtype=complex)
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("hamiltonian coefficients must be finite.")
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
    pair_map = _validate_pair_pauli_map(
        _CANONICAL_PAIR_PAULI_MAP if pair_pauli_map is None else pair_pauli_map
    )
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
    """Summarize an encoding that uses the canonical pair Pauli map."""
    mode = _validate_mode(mode)
    return {
        "input_num_qubits": int(hamiltonian.num_qubits),
        "output_num_qubits": int(encoded_hamiltonian.num_qubits),
        "input_num_terms": int(len(hamiltonian)),
        "output_num_terms": int(len(encoded_hamiltonian)),
        "mode": mode,
        "pair_pauli_map": dict(_CANONICAL_PAIR_PAULI_MAP),
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
    }


__all__ = [
    "PauliEncodingMode",
    "DEFAULT_PAIR_PAULI_MAP",
    "EncodingMetadata",
    "encode_pauli_label",
    "encode_sparse_pauliop",
    "encode_sparse_pauliop_with_metadata",
    "summarize_encoding",
]
