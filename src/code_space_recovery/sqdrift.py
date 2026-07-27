"""SqDRIFT circuit-generation utilities for Pauli Hamiltonians.

Circuits are generated in the qubit space of the supplied Hamiltonian, which
may be logical or encoded. Measurement and submission are handled by
``sampling.py``.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


def _validate_positive_int(name: str, value: Any) -> int:
    """Validate that value is an integer >= 1."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _validate_positive_real(name: str, value: Any) -> float:
    """Validate that value is a finite real number > 0."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def _validate_seed(seed: Any) -> int | None:
    """Validate seed. Currently accepts None or a non-negative integer."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be None or a non-negative integer, got {type(seed).__name__}.")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")
    return seed


def _pair_seed(seed: int, k: int, r: int) -> int:
    """Derive a deterministic seed for each (k, r).

    This makes the sampled sequence for a fixed (k, r) independent of whether
    k=0 is included or not.
    """
    payload = f"sqdrift:{seed}:k={k}:r={r}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _preprocess_sparse_pauli_hamiltonian(
    hamiltonian: SparsePauliOp,
    *,
    coefficient_atol: float = 1e-12,
    sort_terms: bool = True,
) -> dict[str, Any]:
    """Preprocess SparsePauliOp Hamiltonian for Pauli-basis qDRIFT.

    Input:
        H = sum_j alpha_j P_j

    Processing:
        1. Require coefficients to be real up to coefficient_atol.
        2. Merge duplicate Pauli labels.
        3. Separate identity shift.
        4. Drop zero or near-zero non-identity terms.
        5. Build qDRIFT decomposition:
               a_i = signed coefficient
               c_i = |a_i|
               sign_i = sign(a_i)
               p_i = c_i / lambda
               lambda = sum_i c_i

    Important:
        preprocessed_hamiltonian is the signed effective Hamiltonian H_eff.
        It is not the positive-coefficient qDRIFT decomposition.
    """
    if not isinstance(hamiltonian, SparsePauliOp):
        raise TypeError(
            "hamiltonian must be a qiskit.quantum_info.SparsePauliOp object."
        )

    coefficient_atol = _validate_positive_real("coefficient_atol", coefficient_atol)

    n_qubits = hamiltonian.num_qubits
    identity_label = "I" * n_qubits

    labels = list(hamiltonian.paulis.to_labels())
    coeffs = np.asarray(hamiltonian.coeffs, dtype=complex)

    merged_coeffs: dict[str, float] = {}
    source_indices: dict[str, list[int]] = {}

    for original_index, (label, coeff) in enumerate(zip(labels, coeffs)):
        if abs(float(np.imag(coeff))) > coefficient_atol:
            raise ValueError(
                "Hamiltonian coefficients must be real for a Hermitian Pauli Hamiltonian. "
                f"Term index {original_index}, label {label}, coefficient {coeff} has "
                f"imaginary part larger than coefficient_atol={coefficient_atol}."
            )

        real_coeff = float(np.real(coeff))
        merged_coeffs[label] = merged_coeffs.get(label, 0.0) + real_coeff
        source_indices.setdefault(label, []).append(original_index)

    num_unique_labels_after_merge = len(merged_coeffs)

    identity_shift = float(merged_coeffs.pop(identity_label, 0.0))
    identity_source_indices = source_indices.get(identity_label, [])

    if abs(identity_shift) <= coefficient_atol:
        identity_shift = 0.0

    kept_items: list[tuple[str, float]] = []
    dropped_zero_labels: list[str] = []

    for label, coeff in merged_coeffs.items():
        coeff = float(coeff)
        if abs(coeff) <= coefficient_atol:
            dropped_zero_labels.append(label)
        else:
            kept_items.append((label, coeff))

    if sort_terms:
        kept_items.sort(key=lambda item: item[0])

    processed_labels = [label for label, _ in kept_items]
    signed_coefficients = np.array([coeff for _, coeff in kept_items], dtype=float)

    num_processed_terms = len(processed_labels)

    if num_processed_terms > 0:
        preprocessed_hamiltonian = SparsePauliOp.from_list(
            [(label, coeff) for label, coeff in kept_items]
        )
        signs = np.sign(signed_coefficients).astype(float)
        weights = np.abs(signed_coefficients).astype(float)
        lambda_value = float(np.sum(weights))
        probabilities = weights / lambda_value
    else:
        # Zero effective Hamiltonian. Qiskit still needs a valid SparsePauliOp object.
        preprocessed_hamiltonian = SparsePauliOp.from_list([(identity_label, 0.0)])
        signs = np.array([], dtype=float)
        weights = np.array([], dtype=float)
        probabilities = np.array([], dtype=float)
        lambda_value = 0.0

    qdrift_terms: list[dict[str, Any]] = []
    for i, label in enumerate(processed_labels):
        qdrift_terms.append(
            {
                "term_index": i,
                "pauli_label": label,
                "signed_coefficient": float(signed_coefficients[i]),
                "sign": float(signs[i]),
                "weight": float(weights[i]),
                "probability": float(probabilities[i]),
                "source_indices": list(source_indices.get(label, [])),
            }
        )

    return {
        "preprocessed_hamiltonian": preprocessed_hamiltonian,
        "identity_shift": identity_shift,
        "identity_label": identity_label,
        "identity_source_indices": identity_source_indices,
        "processed_labels": processed_labels,
        "signed_coefficients": signed_coefficients,
        "signs": signs,
        "weights": weights,
        "probabilities": probabilities,
        "lambda": lambda_value,
        "qdrift_terms": qdrift_terms,
        "preprocessing_summary": {
            "input_num_terms": len(labels),
            "num_unique_labels_after_merge": num_unique_labels_after_merge,
            "num_processed_non_identity_terms": num_processed_terms,
            "dropped_zero_labels": dropped_zero_labels,
            "coefficient_atol": coefficient_atol,
            "sort_terms": sort_terms,
            "identity_removed_from_sampling_pool": True,
        },
    }


def _prepare_initial_state(
    initial_state: QuantumCircuit | None,
    *,
    n_qubits: int,
) -> tuple[QuantumCircuit, bool]:
    """Prepare the initial-state circuit.

    If initial_state is None, return an empty n-qubit circuit preparing |00...0>.
    """
    if initial_state is None:
        qc_initial = QuantumCircuit(n_qubits, name="psi0")
        return qc_initial, True

    if not isinstance(initial_state, QuantumCircuit):
        raise TypeError(
            "initial_state must be None or a qiskit.QuantumCircuit object."
        )

    if initial_state.num_qubits != n_qubits:
        raise ValueError(
            "initial_state.num_qubits must match hamiltonian.num_qubits. "
            f"Got initial_state.num_qubits={initial_state.num_qubits}, "
            f"hamiltonian.num_qubits={n_qubits}."
        )

    if initial_state.num_clbits != 0:
        raise ValueError(
            "initial_state should be a pure state-preparation circuit with no classical bits. "
            f"Got initial_state.num_clbits={initial_state.num_clbits}."
        )

    qc_initial = initial_state.copy()
    return qc_initial, False


def _sample_qdrift_indices(
    *,
    probabilities: np.ndarray,
    num_terms: int,
    N_seq: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample qDRIFT term indices with replacement.

    Returned indices refer to the processed qDRIFT term table, not the original
    input Hamiltonian term indices.
    """
    if num_terms == 0:
        return []

    sampled = rng.choice(
        num_terms,
        size=N_seq,
        replace=True,
        p=probabilities,
    )
    return [int(x) for x in sampled]


def _make_single_pauli_evolution_gate(
    *,
    pauli_label: str,
    signed_time: float,
) -> PauliEvolutionGate:
    """Build a high-level PauliEvolutionGate for exp(-i * signed_time * P).

    The Pauli operator is unsigned, coefficient +1.
    The sign of the Hamiltonian coefficient is absorbed into signed_time.
    """
    unsigned_pauli_operator = SparsePauliOp.from_list([(pauli_label, 1.0)])
    return PauliEvolutionGate(unsigned_pauli_operator, time=float(signed_time))


def generate_sqdrift_logical_circuits(
    hamiltonian: SparsePauliOp,
    initial_state: QuantumCircuit | None = None,
    *,
    K: int,
    include_k0: bool,
    N_R: int,
    N_seq: int,
    delta_t: float,
    seed: int | None = None,
    coefficient_atol: float = 1e-12,
    sort_terms: bool = True,
) -> dict[str, Any]:
    """Generate unmeasured SqDRIFT circuits for a Pauli-basis Hamiltonian.

    The supplied Hamiltonian and initial state determine whether the circuit
    represents a logical or encoded qubit space.

    Convention:
        k = 0, 1, ..., K-1.

    Circuit-generation policy:
        - If include_k0=True, create exactly one k=0 record with r=0.
          This circuit is only the initial-state preparation circuit.
        - If include_k0=False, no k=0 record is created.
        - For k >= 1, create N_R independent qDRIFT randomized realizations.

    qDRIFT step:
        H_eff = sum_i a_i P_i
        c_i = |a_i|
        sign_i = sign(a_i)
        lambda = sum_i c_i
        p_i = c_i / lambda

        For sampled index i at Krylov time tau_k = k * delta_t, append

            exp(-i * sign_i * P_i * lambda * tau_k / N_seq)

        implemented as

            PauliEvolutionGate(P_i, time=sign_i * lambda * tau_k / N_seq)

    Output format:
        result = {
            "metadata": {...},
            "circuit_records": [
                {
                    "record_id": ...,
                    "k": ...,
                    "r": ...,
                    "circuit": QuantumCircuit,
                    "sampled_term_indices": [...],
                    "sampled_pauli_labels": [...],
                    "evolution_time": ...,
                    "qdrift_base_step_time": ...,
                },
                ...
            ],
        }

    Notes:
        - No measurement is added.
        - No physical layout or backend transpilation is performed.
        - sampled_term_indices refer to metadata["qdrift_decomposition"]["terms"].
        - If the effective Hamiltonian is zero after removing identity and zero terms,
          k>=1 records are still created, but they contain only the initial-state circuit.
    """
    if not isinstance(hamiltonian, SparsePauliOp):
        raise TypeError(
            "hamiltonian must be a qiskit.quantum_info.SparsePauliOp object."
        )

    if not isinstance(include_k0, bool):
        raise TypeError(
            f"include_k0 must be bool, got {type(include_k0).__name__}."
        )

    K = _validate_positive_int("K", K)
    N_R = _validate_positive_int("N_R", N_R)
    N_seq = _validate_positive_int("N_seq", N_seq)
    delta_t = _validate_positive_real("delta_t", delta_t)
    seed = _validate_seed(seed)
    coefficient_atol = _validate_positive_real("coefficient_atol", coefficient_atol)

    n_qubits = hamiltonian.num_qubits

    preprocessing = _preprocess_sparse_pauli_hamiltonian(
        hamiltonian,
        coefficient_atol=coefficient_atol,
        sort_terms=sort_terms,
    )

    qc_initial, initial_state_was_none = _prepare_initial_state(
        initial_state,
        n_qubits=n_qubits,
    )

    processed_labels: list[str] = preprocessing["processed_labels"]
    signs: np.ndarray = preprocessing["signs"]
    probabilities: np.ndarray = preprocessing["probabilities"]
    lambda_value: float = preprocessing["lambda"]
    num_terms = len(processed_labels)

    circuit_records: list[dict[str, Any]] = []

    # For seed=None, use one entropy-based RNG sequentially.
    # For seed=int, use deterministic per-(k,r) RNGs.
    master_rng = np.random.default_rng(None) if seed is None else None

    # k = 0 record: compact policy, one initial-state circuit only.
    if include_k0:
        qc0 = qc_initial.copy()
        qc0.name = "sqdrift_k000_r000"

        circuit_records.append(
            {
                "record_id": qc0.name,
                "k": 0,
                "r": 0,
                "circuit": qc0,
                "sampled_term_indices": [],
                "sampled_pauli_labels": [],
                "evolution_time": 0.0,
                "qdrift_base_step_time": 0.0,
            }
        )

    # k >= 1 randomized qDRIFT circuits.
    for k in range(1, K):
        tau_k = float(k * delta_t)
        qdrift_base_step_time = (
            float(lambda_value * tau_k / N_seq) if lambda_value > 0.0 else 0.0
        )

        for r in range(N_R):
            record_id = f"sqdrift_k{k:03d}_r{r:03d}"

            if seed is None:
                assert master_rng is not None
                rng = master_rng
            else:
                rng = np.random.default_rng(_pair_seed(seed, k, r))

            sampled_term_indices = _sample_qdrift_indices(
                probabilities=probabilities,
                num_terms=num_terms,
                N_seq=N_seq,
                rng=rng,
            )

            qc = qc_initial.copy()
            qc.name = record_id

            sampled_pauli_labels: list[str] = []

            for term_index in sampled_term_indices:
                pauli_label = processed_labels[term_index]
                signed_time = float(signs[term_index] * qdrift_base_step_time)

                gate = _make_single_pauli_evolution_gate(
                    pauli_label=pauli_label,
                    signed_time=signed_time,
                )

                qc.append(gate, qargs=list(range(n_qubits)))
                sampled_pauli_labels.append(pauli_label)

            circuit_records.append(
                {
                    "record_id": record_id,
                    "k": k,
                    "r": r,
                    "circuit": qc,
                    "sampled_term_indices": sampled_term_indices,
                    "sampled_pauli_labels": sampled_pauli_labels,
                    "evolution_time": tau_k,
                    "qdrift_base_step_time": qdrift_base_step_time,
                }
            )

    expected_num_records = (1 if include_k0 else 0) + (K - 1) * N_R

    result = {
        "metadata": {
            "hamiltonian": hamiltonian.copy(),
            "preprocessed_hamiltonian": preprocessing["preprocessed_hamiltonian"],
            "identity_shift": preprocessing["identity_shift"],
            "identity_label": preprocessing["identity_label"],
            "identity_source_indices": preprocessing["identity_source_indices"],
            "qdrift_decomposition": {
                "lambda": preprocessing["lambda"],
                "terms": preprocessing["qdrift_terms"],
                "index_convention": (
                    "sampled_term_indices refer to qdrift_decomposition['terms'][term_index], "
                    "not to the original input Hamiltonian term indices."
                ),
            },
            "preprocessing_summary": preprocessing["preprocessing_summary"],
            "initial_state": qc_initial.copy(),
            "initial_state_was_none": initial_state_was_none,
            "n_qubits": n_qubits,
            "K": K,
            "K_convention": "k = 0, 1, ..., K-1",
            "include_k0": include_k0,
            "N_R": N_R,
            "N_seq": N_seq,
            "delta_t": delta_t,
            "seed": seed,
            "record_format": "list_of_records",
            "expected_num_records": expected_num_records,
            "actual_num_records": len(circuit_records),
            "measurements_added": False,
            "transpiled": False,
            "pauli_label_qubit_order_note": (
                "Qiskit Pauli-label convention is used as-is. "
                "Be careful when matching labels and bitstrings later."
            ),
        },
        "circuit_records": circuit_records,
    }

    if len(circuit_records) != expected_num_records:
        raise RuntimeError(
            "Internal error: unexpected number of circuit records. "
            f"Expected {expected_num_records}, got {len(circuit_records)}."
        )

    return result
