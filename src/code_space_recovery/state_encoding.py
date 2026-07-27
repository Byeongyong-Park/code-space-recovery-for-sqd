"""State-preparation helpers for pair-code encoded circuits.

The displayed pair order is ``|0> -> |01>`` and ``|1> -> |10>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class PairEncodedStateCircuitMetadata:
    """Metadata describing the physical rail layout of a pair-encoded state circuit."""

    n_logical: int
    n_encoded: int
    first_rail_qubits: tuple[int, ...]
    second_rail_qubits: tuple[int, ...]
    logical_to_encoded_qubits: dict[int, tuple[int, int]]
    pair_code: str
    qiskit_order_note: str


def pair_code_rail_qubits(n_logical: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return pair-code rail qubit indices under Qiskit's displayed bitstring convention.

        logical 0 -> displayed pair 01
        logical 1 -> displayed pair 10

    For logical qubit q:
        first rail  = physical qubit 2*q + 1
        second rail = physical qubit 2*q

    Example for n_logical=2:
        logical q0 -> pair displayed as encoded qubits (q1, q0)
                    -> first rail physical 1, second rail physical 0
        logical q1 -> pair displayed as encoded qubits (q3, q2)
                    -> first rail physical 3, second rail physical 2

        displayed encoded bitstring order is therefore:
            q3 q2 q1 q0
            ^  ^  ^  ^
            |  |  |  |
            q1 first/second, q0 first/second
    """
    if not isinstance(n_logical, int) or isinstance(n_logical, bool):
        raise TypeError(f"n_logical must be an integer, got {type(n_logical).__name__}.")
    if n_logical < 1:
        raise ValueError(f"n_logical must be >= 1, got {n_logical}.")

    first_rails = tuple(2 * q + 1 for q in range(n_logical))
    second_rails = tuple(2 * q for q in range(n_logical))
    return first_rails, second_rails


def encode_state_preparation_circuit_pair_code(
    logical_state_circuit: QuantumCircuit,
    *,
    name: str | None = None,
    add_barriers: bool = False,
    return_metadata: bool = False,
) -> QuantumCircuit | tuple[QuantumCircuit, PairEncodedStateCircuitMetadata]:
    """Convert an n-qubit logical state-preparation circuit to a 2n-qubit pair-encoded circuit.

    Pair code:
        logical |0> -> |01>
        logical |1> -> |10>

    Input:
        logical_state_circuit:
            A Qiskit QuantumCircuit preparing the logical state from |00...0>.
            It must be a pure state-preparation circuit with no classical bits.

    Output:
        encoded_circuit:
            A 2n-qubit QuantumCircuit preparing the encoded state.

    Construction:
        1. Start from all-zero 2n-qubit state.
        2. Set every second rail to |1>.
        3. Compose the logical state-preparation circuit onto the first rails.
        4. Apply CX(first_rail -> second_rail) for each pair.

    This maps

        sum_x alpha_x |x>_logical

    to

        sum_x alpha_x |enc(x)>,

    where each displayed logical bit b becomes displayed encoded pair [b, 1-b].

    This function does not add measurements, so the returned circuit can be
    passed as ``initial_state`` to the SqDRIFT generator.
    """
    if not isinstance(logical_state_circuit, QuantumCircuit):
        raise TypeError(
            "logical_state_circuit must be a qiskit.QuantumCircuit object. "
            f"Got {type(logical_state_circuit).__name__}."
        )

    n_logical = int(logical_state_circuit.num_qubits)
    if n_logical < 1:
        raise ValueError("logical_state_circuit must contain at least one qubit.")

    if logical_state_circuit.num_clbits != 0:
        raise ValueError(
            "logical_state_circuit must have no classical bits. "
            "Use this helper only for state-preparation circuits, not measured circuits. "
            f"Got num_clbits={logical_state_circuit.num_clbits}."
        )

    n_encoded = 2 * n_logical
    first_rails, second_rails = pair_code_rail_qubits(n_logical)

    circuit_name = name
    if circuit_name is None:
        source_name = logical_state_circuit.name or "logical_state"
        circuit_name = f"pair_encoded_{source_name}"

    encoded = QuantumCircuit(n_encoded, name=circuit_name)

    # Prepare |0_L ... 0_L> = |01 01 ... 01> in displayed pair order.
    # Since each pair is displayed as [first rail, second rail], the second rail
    # starts at physical qubit 2*q and is flipped to |1>.
    for q_second in second_rails:
        encoded.x(q_second)

    if add_barriers:
        encoded.barrier()

    # Prepare the logical state on the first rails.
    # logical qubit q is mapped to physical first rail 2*q + 1.
    encoded.compose(
        logical_state_circuit,
        qubits=list(first_rails),
        inplace=True,
    )

    if add_barriers:
        encoded.barrier()

    # Entangle the complement rail:
    # second = 1 XOR first.
    #
    # If first=0 -> pair is 01.
    # If first=1 -> pair is 10.
    for q_first, q_second in zip(first_rails, second_rails, strict=True):
        encoded.cx(q_first, q_second)

    metadata = PairEncodedStateCircuitMetadata(
        n_logical=n_logical,
        n_encoded=n_encoded,
        first_rail_qubits=first_rails,
        second_rail_qubits=second_rails,
        logical_to_encoded_qubits={
            q: (first_rails[q], second_rails[q]) for q in range(n_logical)
        },
        pair_code="logical 0 -> displayed pair 01; logical 1 -> displayed pair 10",
        qiskit_order_note=(
            "Qiskit displays qubit 0 as the rightmost bit. "
            "For logical qubit q, first rail is physical qubit 2*q+1 and "
            "second rail is physical qubit 2*q."
        ),
    )

    encoded.metadata = dict(encoded.metadata or {})
    encoded.metadata.update(
        {
            "state_encoding": "pair_code",
            "pair_code": metadata.pair_code,
            "source_logical_circuit_name": logical_state_circuit.name,
            "n_logical": metadata.n_logical,
            "n_encoded": metadata.n_encoded,
            "first_rail_qubits": metadata.first_rail_qubits,
            "second_rail_qubits": metadata.second_rail_qubits,
            "logical_to_encoded_qubits": metadata.logical_to_encoded_qubits,
            "qiskit_order_note": metadata.qiskit_order_note,
        }
    )

    if return_metadata:
        return encoded, metadata
    return encoded
