"""Logical benchmark Hamiltonians for code-space recovery and encoded SqDRIFT.

Main convention
---------------
All builders return logical n-qubit Hamiltonians as Qiskit SparsePauliOp objects.
They do not return pair-encoded 2n-qubit Hamiltonians. Use
`encoding.encode_sparse_pauliop` to encode the returned logical Hamiltonian
when generating encoded SqDRIFT circuits.

Qiskit Pauli-label convention
-----------------------------
SparsePauliOp labels are written in Qiskit's displayed order: the rightmost
label character acts on qubit 0. Therefore, for num_qubits=3:

    X on qubit 0        -> "IIX"
    Z on qubit 2        -> "ZII"
    Z on qubits 0 and 1 -> "IZZ"

Public quick-use builders
-------------------------
    make_1d_tfim_sparse_pauliop
    make_1d_mfim_sparse_pauliop
    make_2d_tfim_sparse_pauliop

Metadata-returning builders
---------------------------
    make_1d_tfim_benchmark
    make_1d_mfim_benchmark
    make_2d_tfim_benchmark

Generic graph-Ising builder
---------------------------
    make_ising_graph_sparse_pauliop
    make_ising_graph_benchmark

Hamiltonian formulas
--------------------
1D TFIM:
    H = zz_coeff * sum_{(i,j) in edges} Z_i Z_j
      + x_coeff  * sum_i X_i
      + identity_coeff * I

1D MFIM:
    H = zz_coeff * sum_{(i,j) in edges} Z_i Z_j
      + x_coeff  * sum_i X_i
      + z_coeff  * sum_i Z_i
      + identity_coeff * I

2D TFIM:
    H = zz_coeff * sum_{<i,j> in square-lattice edges} Z_i Z_j
      + x_coeff  * sum_i X_i
      + identity_coeff * I
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
import json
import math
import numbers

import numpy as np

try:  # Keep the module importable even in environments without Qiskit.
    from qiskit.quantum_info import SparsePauliOp  # type: ignore
except Exception as _exc:  # pragma: no cover - depends on user environment
    SparsePauliOp = None  # type: ignore[assignment]
    _QISKIT_IMPORT_ERROR: Exception | None = _exc
else:  # pragma: no cover - trivial
    _QISKIT_IMPORT_ERROR = None


MODULE_VERSION = "v1.0"
__version__ = MODULE_VERSION
QISKIT_QUBIT_INDEX_CONVENTION = (
    "Qiskit SparsePauliOp displayed labels are big-endian: the rightmost "
    "Pauli-label character acts on qubit 0. Logical basis rows in code-space "
    "recovery are intended to use the same displayed-bit convention."
)


@dataclass(frozen=True)
class PauliTermRecord:
    """Structured record for one Pauli term before SparsePauliOp construction."""

    label: str
    coeff: complex
    kind: str
    sites: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "coeff": _jsonify_coefficient(self.coeff),
            "kind": self.kind,
            "sites": list(self.sites),
            "metadata": _jsonify(self.metadata),
        }


@dataclass(frozen=True)
class HamiltonianMetadata:
    """JSON-friendly metadata for benchmark reproducibility and audit logs."""

    family: str
    num_qubits: int
    formula: str
    parameters: dict[str, Any]
    boundary_conditions: dict[str, Any]
    geometry: dict[str, Any]
    term_counts: dict[str, int]
    pauli_label_convention: str = "qiskit"
    qubit_index_convention: str = QISKIT_QUBIT_INDEX_CONVENTION
    simplify: bool = True
    atol: float = 1e-12
    module_version: str = MODULE_VERSION
    terms_preview: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(frozen=True)
class BenchmarkHamiltonian:
    """Hamiltonian plus reproducibility metadata."""

    hamiltonian: Any
    metadata: HamiltonianMetadata

    @property
    def num_qubits(self) -> int:
        return int(getattr(self.hamiltonian, "num_qubits"))

    @property
    def num_terms(self) -> int:
        return int(len(self.hamiltonian))

    def to_dict(self, *, max_terms: int = 12) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "hamiltonian_summary": summarize_sparse_pauliop(
                self.hamiltonian,
                max_terms=max_terms,
            ),
        }


# =============================================================================
# Small validation / JSON helpers
# =============================================================================


def _require_sparse_pauliop_class() -> Any:
    if SparsePauliOp is None:
        raise ImportError(
            "code_space_recovery.hamiltonians requires qiskit.quantum_info.SparsePauliOp "
            "to build Hamiltonians. Install Qiskit, for example with `pip install qiskit`."
        ) from _QISKIT_IMPORT_ERROR
    return SparsePauliOp


def _jsonify_coefficient(value: complex, *, atol: float = 1e-14) -> float | dict[str, float]:
    z = complex(value)
    if abs(z.imag) <= atol:
        return float(z.real)
    return {"real": float(z.real), "imag": float(z.imag)}


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return _jsonify_coefficient(value)
    if isinstance(value, tuple):
        return [_jsonify(x) for x in value]
    if isinstance(value, list):
        return [_jsonify(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return value


def _as_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a positive integer, got {type(value).__name__}.")
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {out}.")
    return out


def _as_nonnegative_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a nonnegative real number, got {type(value).__name__}.")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {out}.")
    return out


def _as_finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite real number, got {type(value).__name__}.")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out}.")
    return out


def _as_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {type(value).__name__}.")
    return bool(value)


def _validate_pauli_char(pauli: str) -> str:
    if not isinstance(pauli, str):
        raise TypeError(f"Pauli operator must be a string, got {type(pauli).__name__}.")
    op = pauli.upper()
    if op not in {"I", "X", "Y", "Z"}:
        raise ValueError(f"Invalid Pauli operator {pauli!r}; expected one of I, X, Y, Z.")
    return op


def _validate_atol(atol: Any) -> float:
    return _as_nonnegative_float("atol", atol)


# =============================================================================
# Pauli labels and coefficient expansion
# =============================================================================


def make_qiskit_pauli_label(num_qubits: int, ops: Mapping[int, str]) -> str:
    """Create a Qiskit SparsePauliOp label from a qubit-index -> Pauli map.

    Parameters
    ----------
    num_qubits:
        Number of logical qubits.
    ops:
        Mapping from Qiskit qubit index to Pauli char. Qubit 0 is represented by
        the rightmost label character.

    Examples
    --------
    >>> make_qiskit_pauli_label(3, {0: "X"})
    'IIX'
    >>> make_qiskit_pauli_label(3, {2: "Z"})
    'ZII'
    >>> make_qiskit_pauli_label(3, {0: "Z", 1: "Z"})
    'IZZ'
    """
    n = _as_positive_int("num_qubits", num_qubits)
    if not isinstance(ops, Mapping):
        raise TypeError(f"ops must be a mapping from qubit index to Pauli char, got {type(ops).__name__}.")

    chars = ["I"] * n
    for q_raw, pauli_raw in ops.items():
        if isinstance(q_raw, bool) or not isinstance(q_raw, numbers.Integral):
            raise TypeError(f"Qubit index must be an integer, got {q_raw!r}.")
        q = int(q_raw)
        if q < 0 or q >= n:
            raise ValueError(f"Qubit index {q} outside valid range [0, {n}).")
        op = _validate_pauli_char(pauli_raw)
        chars[n - 1 - q] = op
    return "".join(chars)


def expand_coefficients(value: float | Sequence[float] | np.ndarray, length: int, *, name: str) -> np.ndarray:
    """Broadcast or validate real coefficients to a 1D float64 array.

    A scalar is broadcast to `length`. A sequence/array must have exactly
    `length` entries. All entries must be finite real numbers.
    """
    if isinstance(length, bool) or not isinstance(length, numbers.Integral):
        raise TypeError(f"length must be an integer, got {type(length).__name__}.")
    length = int(length)
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}.")

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar or a sequence of real scalars, got bool.")
    if isinstance(value, numbers.Real):
        scalar = _as_finite_real(name, value)
        return np.full(length, scalar, dtype=np.float64)

    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D sequence, got shape {arr.shape}.")
    if arr.shape[0] != length:
        raise ValueError(f"{name} has length {arr.shape[0]}, expected {length}.")
    if np.iscomplexobj(arr):
        imag = np.max(np.abs(np.imag(arr))) if arr.size else 0.0
        if imag > 0.0:
            raise ValueError(f"{name} must be real; got nonzero imaginary component.")
        arr = np.real(arr)
    out = arr.astype(np.float64, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values.")
    return out


# =============================================================================
# Geometry helpers
# =============================================================================


def _canonical_edge(edge: tuple[int, int]) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a < b else (b, a)


def _validate_edges(
    edges: Sequence[tuple[int, int]],
    *,
    num_qubits: int,
    reject_duplicate_edges: bool = True,
) -> list[tuple[int, int]]:
    if not isinstance(edges, Sequence):
        raise TypeError(f"edges must be a sequence of (i, j) pairs, got {type(edges).__name__}.")

    canonical: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for idx, edge in enumerate(edges):
        if not isinstance(edge, Sequence) or len(edge) != 2:
            raise TypeError(f"edges[{idx}] must be a length-2 pair, got {edge!r}.")
        q0_raw, q1_raw = edge
        if (
            isinstance(q0_raw, bool)
            or isinstance(q1_raw, bool)
            or not isinstance(q0_raw, numbers.Integral)
            or not isinstance(q1_raw, numbers.Integral)
        ):
            raise TypeError(f"edges[{idx}] must contain integer qubit indices, got {edge!r}.")
        q0 = int(q0_raw)
        q1 = int(q1_raw)
        if q0 < 0 or q0 >= num_qubits or q1 < 0 or q1 >= num_qubits:
            raise ValueError(f"edges[{idx}]={edge!r} references qubits outside [0, {num_qubits}).")
        if q0 == q1:
            raise ValueError(f"edges[{idx}]={edge!r} is a self-loop; ZZ self-loops are not allowed.")
        e = _canonical_edge((q0, q1))
        if e in seen and reject_duplicate_edges:
            raise ValueError(
                f"Duplicate undirected edge {e!r} found at edges[{idx}]. "
                "Set reject_duplicate_edges=False only if you intentionally want repeated couplings."
            )
        seen.add(e)
        canonical.append(e)
    return canonical


def make_1d_edges(num_qubits: int, *, periodic: bool = False) -> list[tuple[int, int]]:
    """Return deterministic 1D nearest-neighbor edges using qubit indices."""
    n = _as_positive_int("num_qubits", num_qubits)
    periodic = _as_bool("periodic", periodic)
    if periodic and n < 3:
        raise ValueError(
            "periodic=True for a 1D chain requires num_qubits >= 3 to avoid duplicate undirected edges."
        )

    edges = [(i, i + 1) for i in range(n - 1)]
    if periodic:
        edges.append((0, n - 1))
    return edges


def coordinate_to_qubit_index(x: int, y: int, Lx: int, Ly: int, *, site_order: str = "row_major") -> int:
    """Map a 2D coordinate to a logical qubit index."""
    Lx = _as_positive_int("Lx", Lx)
    Ly = _as_positive_int("Ly", Ly)
    if isinstance(x, bool) or not isinstance(x, numbers.Integral):
        raise TypeError("x must be an integer.")
    if isinstance(y, bool) or not isinstance(y, numbers.Integral):
        raise TypeError("y must be an integer.")
    x = int(x)
    y = int(y)
    if x < 0 or x >= Lx or y < 0 or y >= Ly:
        raise ValueError(f"Coordinate ({x}, {y}) outside lattice [0,{Lx}) x [0,{Ly}).")

    order = str(site_order).lower()
    if order in {"row_major", "row-major", "row"}:
        return y * Lx + x
    if order in {"column_major", "column-major", "col_major", "col"}:
        return x * Ly + y
    raise ValueError("site_order must be 'row_major' or 'column_major'.")


def make_2d_square_lattice_edges(
    Lx: int,
    Ly: int,
    *,
    periodic_x: bool = False,
    periodic_y: bool = False,
    site_order: str = "row_major",
) -> list[tuple[int, int]]:
    """Return nearest-neighbor edges for an Lx-by-Ly square lattice.

    Edge order is deterministic: all horizontal edges row by row, then all
    vertical edges row by row. Periodic edges are appended after open-boundary
    edges in each direction.
    """
    Lx = _as_positive_int("Lx", Lx)
    Ly = _as_positive_int("Ly", Ly)
    periodic_x = _as_bool("periodic_x", periodic_x)
    periodic_y = _as_bool("periodic_y", periodic_y)

    if periodic_x and Lx < 3:
        raise ValueError("periodic_x=True requires Lx >= 3 to avoid duplicate undirected edges.")
    if periodic_y and Ly < 3:
        raise ValueError("periodic_y=True requires Ly >= 3 to avoid duplicate undirected edges.")

    def q(x: int, y: int) -> int:
        return coordinate_to_qubit_index(x, y, Lx, Ly, site_order=site_order)

    edges: list[tuple[int, int]] = []

    # Horizontal nearest neighbors.
    for y in range(Ly):
        for x in range(Lx - 1):
            edges.append(_canonical_edge((q(x, y), q(x + 1, y))))
        if periodic_x:
            edges.append(_canonical_edge((q(Lx - 1, y), q(0, y))))

    # Vertical nearest neighbors.
    for y in range(Ly - 1):
        for x in range(Lx):
            edges.append(_canonical_edge((q(x, y), q(x, y + 1))))
    if periodic_y:
        for x in range(Lx):
            edges.append(_canonical_edge((q(x, Ly - 1), q(x, 0))))

    # Safety: periodic constraints above should prevent duplicates, but validate anyway.
    return _validate_edges(edges, num_qubits=Lx * Ly, reject_duplicate_edges=True)


# =============================================================================
# Generic graph-Ising construction
# =============================================================================


def _build_ising_graph_term_records(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    *,
    zz_coeff: float | Sequence[float] | np.ndarray = -1.0,
    x_coeff: float | Sequence[float] | np.ndarray = 0.0,
    z_coeff: float | Sequence[float] | np.ndarray = 0.0,
    identity_coeff: float = 0.0,
    atol: float = 1e-12,
    reject_duplicate_edges: bool = True,
) -> tuple[list[PauliTermRecord], list[tuple[int, int]], dict[str, Any]]:
    n = _as_positive_int("num_qubits", num_qubits)
    atol = _validate_atol(atol)
    edges_valid = _validate_edges(edges, num_qubits=n, reject_duplicate_edges=reject_duplicate_edges)

    zz = expand_coefficients(zz_coeff, len(edges_valid), name="zz_coeff")
    x = expand_coefficients(x_coeff, n, name="x_coeff")
    z = expand_coefficients(z_coeff, n, name="z_coeff")
    identity = _as_finite_real("identity_coeff", identity_coeff)

    records: list[PauliTermRecord] = []

    for edge_index, ((q0, q1), coeff) in enumerate(zip(edges_valid, zz, strict=True)):
        if abs(float(coeff)) <= atol:
            continue
        label = make_qiskit_pauli_label(n, {q0: "Z", q1: "Z"})
        records.append(
            PauliTermRecord(
                label=label,
                coeff=complex(float(coeff), 0.0),
                kind="ZZ",
                sites=(q0, q1),
                metadata={"edge_index": edge_index},
            )
        )

    for q, coeff in enumerate(x):
        if abs(float(coeff)) <= atol:
            continue
        label = make_qiskit_pauli_label(n, {q: "X"})
        records.append(
            PauliTermRecord(
                label=label,
                coeff=complex(float(coeff), 0.0),
                kind="X",
                sites=(q,),
                metadata={"site_index": q},
            )
        )

    for q, coeff in enumerate(z):
        if abs(float(coeff)) <= atol:
            continue
        label = make_qiskit_pauli_label(n, {q: "Z"})
        records.append(
            PauliTermRecord(
                label=label,
                coeff=complex(float(coeff), 0.0),
                kind="Z",
                sites=(q,),
                metadata={"site_index": q},
            )
        )

    if abs(identity) > atol:
        records.append(
            PauliTermRecord(
                label="I" * n,
                coeff=complex(identity, 0.0),
                kind="I",
                sites=(),
                metadata={"identity_shift": True},
            )
        )

    coefficient_summary = {
        "zz_coeff": _coefficient_input_summary(zz_coeff, zz),
        "x_coeff": _coefficient_input_summary(x_coeff, x),
        "z_coeff": _coefficient_input_summary(z_coeff, z),
        "identity_coeff": float(identity),
    }
    return records, edges_valid, coefficient_summary


def _coefficient_input_summary(original: Any, expanded: np.ndarray) -> dict[str, Any]:
    is_scalar = isinstance(original, numbers.Real) and not isinstance(original, bool)
    if len(expanded) == 0:
        return {"kind": "scalar" if is_scalar else "array", "length": 0}
    out: dict[str, Any] = {
        "kind": "scalar" if is_scalar else "array",
        "length": int(len(expanded)),
        "min": float(np.min(expanded)),
        "max": float(np.max(expanded)),
    }
    if is_scalar:
        out["value"] = float(expanded[0])
    else:
        preview_len = min(8, len(expanded))
        out["preview"] = [float(x) for x in expanded[:preview_len]]
        if len(expanded) > preview_len:
            out["preview_truncated"] = True
    return out


def _make_sparse_pauliop_from_records(
    records: Sequence[PauliTermRecord],
    *,
    num_qubits: int,
    simplify: bool = True,
    atol: float = 1e-12,
) -> Any:
    SparsePauliOpClass = _require_sparse_pauliop_class()
    n = _as_positive_int("num_qubits", num_qubits)
    simplify = _as_bool("simplify", simplify)
    atol = _validate_atol(atol)

    if len(records) == 0:
        op = SparsePauliOpClass.from_list([("I" * n, 0.0)])
    else:
        op = SparsePauliOpClass.from_list([(r.label, r.coeff) for r in records])
    if simplify:
        op = op.simplify(atol=atol)
    return op


def make_ising_graph_sparse_pauliop(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    *,
    zz_coeff: float | Sequence[float] | np.ndarray = -1.0,
    x_coeff: float | Sequence[float] | np.ndarray = 0.0,
    z_coeff: float | Sequence[float] | np.ndarray = 0.0,
    identity_coeff: float = 0.0,
    simplify: bool = True,
    atol: float = 1e-12,
    reject_duplicate_edges: bool = True,
) -> Any:
    """Build a logical graph-Ising SparsePauliOp.

    Formula:
        H = sum_e zz_coeff[e] Z_i Z_j + sum_i x_coeff[i] X_i
          + sum_i z_coeff[i] Z_i + identity_coeff * I
    """
    records, _edges_valid, _coeff_summary = _build_ising_graph_term_records(
        num_qubits,
        edges,
        zz_coeff=zz_coeff,
        x_coeff=x_coeff,
        z_coeff=z_coeff,
        identity_coeff=identity_coeff,
        atol=atol,
        reject_duplicate_edges=reject_duplicate_edges,
    )
    return _make_sparse_pauliop_from_records(
        records,
        num_qubits=num_qubits,
        simplify=simplify,
        atol=atol,
    )


def make_ising_graph_benchmark(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    *,
    family: str = "ising_graph",
    zz_coeff: float | Sequence[float] | np.ndarray = -1.0,
    x_coeff: float | Sequence[float] | np.ndarray = 0.0,
    z_coeff: float | Sequence[float] | np.ndarray = 0.0,
    identity_coeff: float = 0.0,
    simplify: bool = True,
    atol: float = 1e-12,
    geometry: Mapping[str, Any] | None = None,
    boundary_conditions: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
    reject_duplicate_edges: bool = True,
    max_terms_preview: int = 12,
) -> BenchmarkHamiltonian:
    """Build a graph-Ising Hamiltonian plus metadata."""
    n = _as_positive_int("num_qubits", num_qubits)
    records, edges_valid, coefficient_summary = _build_ising_graph_term_records(
        n,
        edges,
        zz_coeff=zz_coeff,
        x_coeff=x_coeff,
        z_coeff=z_coeff,
        identity_coeff=identity_coeff,
        atol=atol,
        reject_duplicate_edges=reject_duplicate_edges,
    )
    H = _make_sparse_pauliop_from_records(records, num_qubits=n, simplify=simplify, atol=atol)

    term_counts = _count_records_by_kind(records)
    term_counts["total_before_simplify"] = int(len(records))
    term_counts["total_after_simplify"] = int(len(H))

    geom = dict(geometry or {})
    geom.setdefault("edges", [list(e) for e in edges_valid])
    geom.setdefault("num_edges", int(len(edges_valid)))

    metadata = HamiltonianMetadata(
        family=str(family),
        num_qubits=n,
        formula=(
            "H = sum_e zz_coeff[e] Z_i Z_j + sum_i x_coeff[i] X_i "
            "+ sum_i z_coeff[i] Z_i + identity_coeff * I"
        ),
        parameters=coefficient_summary,
        boundary_conditions=dict(boundary_conditions or {}),
        geometry=_jsonify(geom),
        term_counts=term_counts,
        simplify=bool(simplify),
        atol=float(atol),
        terms_preview=[r.to_dict() for r in records[:max_terms_preview]],
        notes=list(notes),
    )
    return BenchmarkHamiltonian(hamiltonian=H, metadata=metadata)


def _count_records_by_kind(records: Sequence[PauliTermRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.kind] = counts.get(record.kind, 0) + 1
    for key in ("ZZ", "X", "Z", "I"):
        counts.setdefault(key, 0)
    counts["total"] = int(len(records))
    return counts


# =============================================================================
# Named benchmark families
# =============================================================================


def make_1d_tfim_sparse_pauliop(
    num_qubits: int,
    *,
    zz_coeff: float | Sequence[float] | np.ndarray = -1.0,
    x_coeff: float | Sequence[float] | np.ndarray = -1.0,
    periodic: bool = False,
    identity_coeff: float = 0.0,
    simplify: bool = True,
    atol: float = 1e-12,
) -> Any:
    """Build the logical 1D transverse-field Ising model as SparsePauliOp."""
    edges = make_1d_edges(num_qubits, periodic=periodic)
    return make_ising_graph_sparse_pauliop(
        num_qubits,
        edges,
        zz_coeff=zz_coeff,
        x_coeff=x_coeff,
        z_coeff=0.0,
        identity_coeff=identity_coeff,
        simplify=simplify,
        atol=atol,
    )


def make_1d_tfim_benchmark(
    num_qubits: int,
    *,
    zz_coeff: float | Sequence[float] | np.ndarray = -1.0,
    x_coeff: float | Sequence[float] | np.ndarray = -1.0,
    periodic: bool = False,
    identity_coeff: float = 0.0,
    simplify: bool = True,
    atol: float = 1e-12,
    max_terms_preview: int = 12,
) -> BenchmarkHamiltonian:
    """Build the logical 1D TFIM benchmark Hamiltonian with metadata."""
    edges = make_1d_edges(num_qubits, periodic=periodic)
    return make_ising_graph_benchmark(
        num_qubits,
        edges,
        family="1d_tfim",
        zz_coeff=zz_coeff,
        x_coeff=x_coeff,
        z_coeff=0.0,
        identity_coeff=identity_coeff,
        simplify=simplify,
        atol=atol,
        geometry={
            "dimension": 1,
            "lattice": "chain",
            "site_order": "linear_qiskit_index",
        },
        boundary_conditions={"periodic": bool(periodic)},
        notes=["Logical Hamiltonian; call encode_sparse_pauliop separately for pair-encoded circuits."],
        max_terms_preview=max_terms_preview,
    )


def make_1d_mfim_sparse_pauliop(
    num_qubits: int,
    *,
    zz_coeff: float | Sequence[float] | np.ndarray = -1.0,
    x_coeff: float | Sequence[float] | np.ndarray = -1.0,
    z_coeff: float | Sequence[float] | np.ndarray = -0.5,
    periodic: bool = False,
    identity_coeff: float = 0.0,
    simplify: bool = True,
    atol: float = 1e-12,
) -> Any:
    """Build the logical 1D mixed-field Ising model as SparsePauliOp."""
    edges = make_1d_edges(num_qubits, periodic=periodic)
    return make_ising_graph_sparse_pauliop(
        num_qubits,
        edges,
        zz_coeff=zz_coeff,
        x_coeff=x_coeff,
        z_coeff=z_coeff,
        identity_coeff=identity_coeff,
        simplify=simplify,
        atol=atol,
    )


def make_1d_mfim_benchmark(
    num_qubits: int,
    *,
    zz_coeff: float | Sequence[float] | np.ndarray = -1.0,
    x_coeff: float | Sequence[float] | np.ndarray = -1.0,
    z_coeff: float | Sequence[float] | np.ndarray = -0.5,
    periodic: bool = False,
    identity_coeff: float = 0.0,
    simplify: bool = True,
    atol: float = 1e-12,
    max_terms_preview: int = 12,
) -> BenchmarkHamiltonian:
    """Build the logical 1D MFIM benchmark Hamiltonian with metadata."""
    edges = make_1d_edges(num_qubits, periodic=periodic)
    return make_ising_graph_benchmark(
        num_qubits,
        edges,
        family="1d_mfim",
        zz_coeff=zz_coeff,
        x_coeff=x_coeff,
        z_coeff=z_coeff,
        identity_coeff=identity_coeff,
        simplify=simplify,
        atol=atol,
        geometry={
            "dimension": 1,
            "lattice": "chain",
            "site_order": "linear_qiskit_index",
        },
        boundary_conditions={"periodic": bool(periodic)},
        notes=[
            "MFIM here means Ising ZZ coupling plus transverse X field plus longitudinal Z field.",
            "Logical Hamiltonian; call encode_sparse_pauliop separately for pair-encoded circuits.",
        ],
        max_terms_preview=max_terms_preview,
    )


def make_2d_tfim_sparse_pauliop(
    Lx: int,
    Ly: int,
    *,
    zz_coeff: float | Sequence[float] | np.ndarray = -1.0,
    x_coeff: float | Sequence[float] | np.ndarray = -1.0,
    periodic_x: bool = False,
    periodic_y: bool = False,
    site_order: str = "row_major",
    identity_coeff: float = 0.0,
    simplify: bool = True,
    atol: float = 1e-12,
) -> Any:
    """Build the logical 2D square-lattice TFIM as SparsePauliOp."""
    Lx = _as_positive_int("Lx", Lx)
    Ly = _as_positive_int("Ly", Ly)
    edges = make_2d_square_lattice_edges(
        Lx,
        Ly,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
        site_order=site_order,
    )
    return make_ising_graph_sparse_pauliop(
        Lx * Ly,
        edges,
        zz_coeff=zz_coeff,
        x_coeff=x_coeff,
        z_coeff=0.0,
        identity_coeff=identity_coeff,
        simplify=simplify,
        atol=atol,
    )


def make_2d_tfim_benchmark(
    Lx: int,
    Ly: int,
    *,
    zz_coeff: float | Sequence[float] | np.ndarray = -1.0,
    x_coeff: float | Sequence[float] | np.ndarray = -1.0,
    periodic_x: bool = False,
    periodic_y: bool = False,
    site_order: str = "row_major",
    identity_coeff: float = 0.0,
    simplify: bool = True,
    atol: float = 1e-12,
    max_terms_preview: int = 12,
) -> BenchmarkHamiltonian:
    """Build the logical 2D square-lattice TFIM benchmark Hamiltonian with metadata."""
    Lx = _as_positive_int("Lx", Lx)
    Ly = _as_positive_int("Ly", Ly)
    edges = make_2d_square_lattice_edges(
        Lx,
        Ly,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
        site_order=site_order,
    )
    return make_ising_graph_benchmark(
        Lx * Ly,
        edges,
        family="2d_tfim",
        zz_coeff=zz_coeff,
        x_coeff=x_coeff,
        z_coeff=0.0,
        identity_coeff=identity_coeff,
        simplify=simplify,
        atol=atol,
        geometry={
            "dimension": 2,
            "lattice": "square",
            "Lx": int(Lx),
            "Ly": int(Ly),
            "site_order": str(site_order),
            "coordinate_to_qubit_index": (
                "row_major: q = y * Lx + x; column_major: q = x * Ly + y"
            ),
        },
        boundary_conditions={
            "periodic_x": bool(periodic_x),
            "periodic_y": bool(periodic_y),
        },
        notes=["Logical Hamiltonian; call encode_sparse_pauliop separately for pair-encoded circuits."],
        max_terms_preview=max_terms_preview,
    )


# =============================================================================
# Summary and serialization utilities
# =============================================================================


def classify_pauli_label(label: str) -> tuple[str, tuple[int, ...]]:
    """Classify a Qiskit Pauli label and return (kind, sites)."""
    if not isinstance(label, str) or len(label) == 0:
        raise ValueError("label must be a non-empty Pauli string.")
    n = len(label)
    non_id: list[tuple[int, str]] = []
    for pos, char in enumerate(label.upper()):
        if char not in {"I", "X", "Y", "Z"}:
            raise ValueError(f"Invalid Pauli character {char!r} in label {label!r}.")
        if char != "I":
            q = n - 1 - pos
            non_id.append((q, char))
    if len(non_id) == 0:
        return "I", ()
    if len(non_id) == 1:
        q, op = non_id[0]
        return op, (q,)
    if len(non_id) == 2 and all(op == "Z" for _q, op in non_id):
        sites = tuple(sorted(q for q, _op in non_id))
        return "ZZ", sites
    kind = "".join(op for _q, op in sorted(non_id, key=lambda t: t[0]))
    sites = tuple(q for q, _op in sorted(non_id, key=lambda t: t[0]))
    return f"OTHER_{kind}", sites


def summarize_sparse_pauliop(hamiltonian: Any, *, max_terms: int = 12) -> dict[str, Any]:
    """Return a JSON-friendly summary of a SparsePauliOp-like Hamiltonian."""
    if not hasattr(hamiltonian, "num_qubits") or not hasattr(hamiltonian, "paulis"):
        raise TypeError("hamiltonian must be SparsePauliOp-like with num_qubits, paulis, and coeffs.")
    labels = list(hamiltonian.paulis.to_labels())
    coeffs = np.asarray(hamiltonian.coeffs, dtype=complex)

    counts: dict[str, int] = {}
    preview: list[dict[str, Any]] = []
    for idx, (label, coeff) in enumerate(zip(labels, coeffs, strict=True)):
        kind, sites = classify_pauli_label(label)
        counts[kind] = counts.get(kind, 0) + 1
        if idx < max_terms:
            preview.append(
                {
                    "term_index": idx,
                    "label": label,
                    "coeff": _jsonify_coefficient(coeff),
                    "kind": kind,
                    "sites": list(sites),
                }
            )

    return {
        "num_qubits": int(hamiltonian.num_qubits),
        "num_terms": int(len(labels)),
        "term_counts_by_kind": counts,
        "coefficients_real_up_to_1e-12": bool(np.all(np.abs(coeffs.imag) <= 1e-12)),
        "terms_preview": preview,
        "preview_truncated": bool(len(labels) > max_terms),
        "pauli_label_convention": "qiskit",
        "qubit_index_convention": QISKIT_QUBIT_INDEX_CONVENTION,
    }


def benchmark_to_json(benchmark: BenchmarkHamiltonian, *, indent: int = 2, max_terms: int = 12) -> str:
    """Serialize BenchmarkHamiltonian metadata and summary to JSON."""
    return json.dumps(benchmark.to_dict(max_terms=max_terms), indent=indent, sort_keys=True)


__all__ = [
    "MODULE_VERSION",
    "__version__",
    "QISKIT_QUBIT_INDEX_CONVENTION",
    "PauliTermRecord",
    "HamiltonianMetadata",
    "BenchmarkHamiltonian",
    "make_qiskit_pauli_label",
    "expand_coefficients",
    "make_1d_edges",
    "coordinate_to_qubit_index",
    "make_2d_square_lattice_edges",
    "make_ising_graph_sparse_pauliop",
    "make_ising_graph_benchmark",
    "make_1d_tfim_sparse_pauliop",
    "make_1d_tfim_benchmark",
    "make_1d_mfim_sparse_pauliop",
    "make_1d_mfim_benchmark",
    "make_2d_tfim_sparse_pauliop",
    "make_2d_tfim_benchmark",
    "classify_pauli_label",
    "summarize_sparse_pauliop",
    "benchmark_to_json",
]


if __name__ == "__main__":  # pragma: no cover - convenience smoke check
    if SparsePauliOp is None:
        print(
            json.dumps(
                {
                    "module": "code_space_recovery.hamiltonians",
                    "status": "SKIP",
                    "reason": "Qiskit is not installed; builders require qiskit.quantum_info.SparsePauliOp.",
                },
                indent=2,
            )
        )
    else:
        bench = make_1d_tfim_benchmark(num_qubits=3, zz_coeff=-1.0, x_coeff=-0.5)
        print(benchmark_to_json(bench, indent=2, max_terms=20))
