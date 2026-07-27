"""Sampling and post-processing for encoded SqDRIFT and code-space recovery.

The workflow covers submission metadata, realization-level post-processing,
and merging the resulting records into clustering inputs.

Conventions
-----------
- Final measurement bitstrings are Qiskit displayed-order strings, converted to
  np.uint8 rows with the leftmost displayed bit in column 0.
- Code-space recovery samples contain encoded-system bits only.
- Circuits contain only the final measurement register; no reset probe or
  reset postselection is used.
- Two post-processing variants are used. Their historical ``*_reset_off``
  identifiers are retained for saved-artifact compatibility:
    m3_off_reset_off
    m3_on_reset_off
- M3-on `.npz` files always save the raw quasi distribution as well as the
  nonnegative SQD/clustering weights.
- QPY circuits are saved as plain `.qpy`, never as `.qpy.gz`, to avoid Qiskit
  backward-seek failures on gzip write streams.
- M3 is applied only to the final `meas` register.
- Per-realization active-physical-qubit count statistics are recorded in
  compiled summaries and printed logs.
- Compilation is separated from M3 calibration and sampling submission.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import gzip
import io
import json
import math
import shutil

import numpy as np

try:  # Keep this module importable in non-Qiskit analysis environments.
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, qpy  # type: ignore
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    QuantumCircuit = None  # type: ignore[assignment]
    ClassicalRegister = None  # type: ignore[assignment]
    QuantumRegister = None  # type: ignore[assignment]
    qpy = None  # type: ignore[assignment]
    generate_preset_pass_manager = None  # type: ignore[assignment]

try:
    from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    Batch = None  # type: ignore[assignment]
    Sampler = None  # type: ignore[assignment]

try:
    import mthree  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    mthree = None  # type: ignore[assignment]


MODULE_VERSION = "v1.0"
__version__ = MODULE_VERSION

POSTPROCESSING_BRANCHES = (
    "m3_off_reset_off",
    "m3_on_reset_off",
)


# =============================================================================
# Small import / JSON helpers
# =============================================================================


def _require_qiskit() -> None:
    if QuantumCircuit is None or ClassicalRegister is None or QuantumRegister is None:
        raise ImportError("sampling requires qiskit for circuit construction.")


def _require_runtime() -> None:
    if Batch is None or Sampler is None or generate_preset_pass_manager is None:
        raise ImportError(
            "sampling submission requires qiskit-ibm-runtime "
            "and qiskit transpiler utilities."
        )


def _require_mthree() -> None:
    if mthree is None:
        raise ImportError("sampling requires mthree for M3 calibration/correction.")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonify(value: Any) -> Any:
    """Best-effort JSON conversion for experiment logs."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if hasattr(value, "job_id") and callable(value.job_id):
        try:
            return {"job_id": value.job_id(), "class": value.__class__.__name__}
        except Exception:
            pass
    if QuantumCircuit is not None and isinstance(value, QuantumCircuit):
        return {
            "class": "QuantumCircuit",
            "name": value.name,
            "num_qubits": int(value.num_qubits),
            "num_clbits": int(value.num_clbits),
            "depth": int(value.depth()),
            "metadata": _jsonify(value.metadata or {}),
        }
    return repr(value)


def _write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(data), f, indent=2, sort_keys=True, ensure_ascii=False)
    return path


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _job_id(job: Any) -> str:
    return str(job.job_id() if hasattr(job, "job_id") and callable(job.job_id) else job)


def _job_ids(jobs: Any) -> list[str]:
    if jobs is None:
        return []
    if isinstance(jobs, (str, bytes)):
        return [str(jobs)]
    try:
        return [_job_id(job) for job in jobs]
    except TypeError:
        return [_job_id(jobs)]


def _backend_name(backend: Any) -> str | None:
    name = getattr(backend, "name", None)
    if callable(name):
        try:
            name = name()
        except Exception:
            name = None
    return None if name is None else str(name)


# =============================================================================
# Circuit and submission helpers
# =============================================================================


def make_logical_ghz_initial_state(n_logical: int):
    """Create an ``n_logical``-qubit GHZ state-preparation circuit."""
    _require_qiskit()
    n_logical = int(n_logical)
    logical_initial_state = QuantumCircuit(n_logical)
    logical_initial_state.h(0)
    for i in range(n_logical - 1):
        logical_initial_state.cx(0, i + 1)
    return logical_initial_state


def add_final_measurement(circuit):
    """Add only a final measurement register to a SqDRIFT circuit.

    The circuit contains no initial probe measurement. Its only classical
    register is:
        meas: measured after the circuit body
    """
    _require_qiskit()
    n = circuit.num_qubits

    qr = QuantumRegister(n, "q")
    meas = ClassicalRegister(n, "meas")

    out = QuantumCircuit(qr, meas, name=(circuit.name or "circuit") + "_sample")
    out.compose(circuit, qubits=range(n), inplace=True)
    out.measure(range(n), meas)
    return out


def _mapping_values(mapping: Any) -> list[int]:
    """Return physical qubits in classical-bit order for a measurement mapping.

    `mthree.utils.final_measurement_mapping` can return either a dict or a list.
    If it is a dict, keys are sorted so the returned list is ordered by the
    classical bit index.
    """
    if isinstance(mapping, Mapping):
        return [int(mapping[k]) for k in sorted(mapping, key=lambda x: int(x))]
    return [int(q) for q in list(mapping)]


def _classical_register_indices(circ: Any, reg_name: str) -> list[int]:
    """Return global classical-bit indices for a named ClassicalRegister."""
    out: list[int] = []
    for creg in getattr(circ, "cregs", []):
        if getattr(creg, "name", None) != reg_name:
            continue
        for bit in creg:
            try:
                out.append(int(circ.find_bit(bit).index))
            except Exception:
                pass
        break
    return out


def _mapping_values_for_cbits(mapping: Any, cbit_indices: Sequence[int]) -> list[int]:
    """Return mapping values restricted to explicit classical-bit indices."""
    idx = [int(i) for i in cbit_indices]
    if isinstance(mapping, Mapping):
        return [int(mapping[i]) if i in mapping else int(mapping[str(i)]) for i in idx]

    values = [int(q) for q in list(mapping)]
    if len(idx) == 0:
        return []
    if max(idx) < len(values):
        return [int(values[i]) for i in idx]

    raise ValueError(
        "Cannot restrict a list-style measurement mapping by classical-bit index: "
        f"max index {max(idx)} but mapping length {len(values)}."
    )


def _final_meas_mapping_for_circuit(circ: Any, full_mapping: Any | None = None) -> list[int]:
    """Return the M3 mapping for the final `meas` register only.

    M3 correction is applied to counts from `meas.get_counts()`, so the qubit
    list passed to M3 must follow that register's classical-bit order. Explicit
    register selection also keeps legacy multi-register artifacts readable.
    """
    if full_mapping is None:
        if mthree is None:
            raise ImportError("mthree is required to infer measurement mappings.")
        full_mapping = mthree.utils.final_measurement_mapping(circ)

    meas_indices = _classical_register_indices(circ, "meas")
    if len(meas_indices) > 0:
        return _mapping_values_for_cbits(full_mapping, meas_indices)

    # Fallback for circuits without a named `meas` register.
    return _mapping_values(full_mapping)


def _m3_qubits_for_counts(mapping: Any, n_bits: int) -> list[int]:
    """Return a qubit list whose length matches an n-bit counts dictionary.

    Current mappings contain only final measurements. Legacy saved mappings may
    contain additional leading classical registers; in those artifacts the
    final `meas` register occupies the trailing ``n_bits`` entries.
    """
    n_bits = int(n_bits)
    values = _mapping_values(mapping)

    if len(values) == n_bits:
        return values

    if len(values) > n_bits:
        # Legacy artifacts stored the full mapping while counts contained only
        # the final `meas` register.
        return values[-n_bits:]

    raise ValueError(
        "M3 mapping length is shorter than the counts bitstring length. "
        f"mapping length={len(values)}, counts bits={n_bits}."
    )


def _circuit_qubit_index(circ: Any, qubit: Any) -> int:
    try:
        return int(circ.find_bit(qubit).index)
    except Exception:
        return int(qubit)


def _summary_stats(values: Sequence[float], prefix: str) -> dict[str, float]:
    """Return min/median/mean/max stats with a common key prefix."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _simple_compiled_summary(isa_circuits: Sequence[Any], mappings: Sequence[Any]) -> dict[str, Any]:
    """Basic per-circuit information that can be read directly from compiled circuits.

    `num_active_physical_qubits` is the per-realization physical-qubit count used
    for reporting. It is computed as

        non-measure quantum-operation qubits UNION final measured qubits

    and deliberately ignores barriers/delays so that an all-qubit barrier or idle
    delay does not falsely count the whole backend as used.
    """
    per_circuit: list[dict[str, Any]] = []

    for i, circ in enumerate(isa_circuits):
        mapping_qubits = sorted(int(q) for q in _mapping_values(mappings[i]))
        measured_qubits = set(mapping_qubits)
        ops = {str(k): int(v) for k, v in dict(circ.count_ops()).items()}

        oneq_count = 0
        twoq_count = 0
        operation_qubits: set[int] = set()

        for inst in circ.data:
            op = inst.operation
            qargs = inst.qubits
            qinds = [_circuit_qubit_index(circ, q) for q in qargs]

            if op.name in {"barrier", "delay", "measure"}:
                continue

            operation_qubits.update(int(q) for q in qinds)
            if len(qargs) == 1:
                oneq_count += 1
            elif len(qargs) == 2:
                twoq_count += 1

        active_qubits = set(operation_qubits) | measured_qubits
        active_sorted = sorted(active_qubits)
        operation_sorted = sorted(operation_qubits)

        metadata = dict(circ.metadata or {})
        per_circuit.append(
            {
                "circuit_index": int(i),
                "name": circ.name,
                "record_id": metadata.get("record_id"),
                "k": metadata.get("k"),
                "r": metadata.get("r"),
                "evolution_time": metadata.get("evolution_time"),
                "depth": int(circ.depth()),
                "num_qubits_in_isa_circuit": int(circ.num_qubits),
                "operation_physical_qubits": operation_sorted,
                "active_physical_qubits": active_sorted,
                "measured_physical_qubits": mapping_qubits,
                "num_operation_physical_qubits": int(len(operation_sorted)),
                "num_active_physical_qubits": int(len(active_sorted)),
                "num_measured_physical_qubits": int(len(mapping_qubits)),
                "op_counts": ops,
                "oneq_count_excluding_measure": int(oneq_count),
                "twoq_count": int(twoq_count),
            }
        )

    if len(per_circuit) == 0:
        summary = {
            "num_circuits": 0,
            "active_physical_qubits": [],
            "num_active_physical_qubits": 0,
            "num_active_physical_qubits_total_union": 0,
            "operation_physical_qubits": [],
            "num_operation_physical_qubits": 0,
            "num_operation_physical_qubits_total_union": 0,
            "measured_physical_qubits": [],
            "num_measured_physical_qubits": 0,
            "num_measured_physical_qubits_total_union": 0,
        }
        return {"summary": summary, "per_circuit": per_circuit}

    depths = np.array([x["depth"] for x in per_circuit], dtype=float)
    oneq_counts = np.array([x["oneq_count_excluding_measure"] for x in per_circuit], dtype=float)
    twoq_counts = np.array([x["twoq_count"] for x in per_circuit], dtype=float)
    active_qubit_counts = np.array([len(x["active_physical_qubits"]) for x in per_circuit], dtype=float)
    operation_qubit_counts = np.array([len(x["operation_physical_qubits"]) for x in per_circuit], dtype=float)
    measured_qubit_counts = np.array([len(x["measured_physical_qubits"]) for x in per_circuit], dtype=float)
    all_active_qubits = sorted({q for x in per_circuit for q in x["active_physical_qubits"]})
    all_operation_qubits = sorted({q for x in per_circuit for q in x["operation_physical_qubits"]})
    all_measured_qubits = sorted({q for x in per_circuit for q in x["measured_physical_qubits"]})

    summary = {
        "num_circuits": int(len(isa_circuits)),

        # Union over all compiled realization circuits.
        "active_physical_qubits": all_active_qubits,
        "num_active_physical_qubits": int(len(all_active_qubits)),
        "operation_physical_qubits": all_operation_qubits,
        "num_operation_physical_qubits": int(len(all_operation_qubits)),
        "measured_physical_qubits": all_measured_qubits,
        "num_measured_physical_qubits": int(len(all_measured_qubits)),

        # Per-realization qubit-count statistics.
        "active_qubits_per_circuit_min": float(active_qubit_counts.min()),
        "active_qubits_per_circuit_mean": float(active_qubit_counts.mean()),
        "active_qubits_per_circuit_median": float(np.median(active_qubit_counts)),
        "active_qubits_per_circuit_max": float(active_qubit_counts.max()),
        "operation_qubits_per_circuit_min": float(operation_qubit_counts.min()),
        "operation_qubits_per_circuit_mean": float(operation_qubit_counts.mean()),
        "operation_qubits_per_circuit_median": float(np.median(operation_qubit_counts)),
        "operation_qubits_per_circuit_max": float(operation_qubit_counts.max()),
        "measured_qubits_per_circuit_min": float(measured_qubit_counts.min()),
        "measured_qubits_per_circuit_mean": float(measured_qubit_counts.mean()),
        "measured_qubits_per_circuit_median": float(np.median(measured_qubit_counts)),
        "measured_qubits_per_circuit_max": float(measured_qubit_counts.max()),

        # Backward-compatible aliases with the longer physical-qubit wording.
        "active_physical_qubits_per_circuit_min": float(active_qubit_counts.min()),
        "active_physical_qubits_per_circuit_mean": float(active_qubit_counts.mean()),
        "active_physical_qubits_per_circuit_median": float(np.median(active_qubit_counts)),
        "active_physical_qubits_per_circuit_max": float(active_qubit_counts.max()),
        "operation_physical_qubits_per_circuit_min": float(operation_qubit_counts.min()),
        "operation_physical_qubits_per_circuit_mean": float(operation_qubit_counts.mean()),
        "operation_physical_qubits_per_circuit_median": float(np.median(operation_qubit_counts)),
        "operation_physical_qubits_per_circuit_max": float(operation_qubit_counts.max()),
        "measured_physical_qubits_per_circuit_min": float(measured_qubit_counts.min()),
        "measured_physical_qubits_per_circuit_mean": float(measured_qubit_counts.mean()),
        "measured_physical_qubits_per_circuit_median": float(np.median(measured_qubit_counts)),
        "measured_physical_qubits_per_circuit_max": float(measured_qubit_counts.max()),

        "depth_min": float(depths.min()),
        "depth_mean": float(depths.mean()),
        "depth_median": float(np.median(depths)),
        "depth_max": float(depths.max()),
        "oneq_count_min": float(oneq_counts.min()),
        "oneq_count_mean": float(oneq_counts.mean()),
        "oneq_count_median": float(np.median(oneq_counts)),
        "oneq_count_max": float(oneq_counts.max()),
        "twoq_count_min": float(twoq_counts.min()),
        "twoq_count_mean": float(twoq_counts.mean()),
        "twoq_count_median": float(np.median(twoq_counts)),
        "twoq_count_max": float(twoq_counts.max()),
    }
    return {"summary": summary, "per_circuit": per_circuit}


def _print_compiled_summary(compiled_info: Mapping[str, Any]) -> None:
    """Print the compiled-circuit summary plus qubit statistics."""
    print("compiled circuit summary:")
    print(compiled_info["summary"])

    summary = compiled_info["summary"]
    print(
        "per-realization active physical qubits: "
        f"mean={summary.get('active_physical_qubits_per_circuit_mean')}, "
        f"median={summary.get('active_physical_qubits_per_circuit_median')}, "
        f"min={summary.get('active_physical_qubits_per_circuit_min')}, "
        f"max={summary.get('active_physical_qubits_per_circuit_max')}"
    )
    print(
        "per-realization operation physical qubits: "
        f"mean={summary.get('operation_physical_qubits_per_circuit_mean')}, "
        f"median={summary.get('operation_physical_qubits_per_circuit_median')}, "
        f"min={summary.get('operation_physical_qubits_per_circuit_min')}, "
        f"max={summary.get('operation_physical_qubits_per_circuit_max')}"
    )
    print(
        "per-realization final measured physical qubits: "
        f"mean={summary.get('measured_physical_qubits_per_circuit_mean')}, "
        f"median={summary.get('measured_physical_qubits_per_circuit_median')}, "
        f"min={summary.get('measured_physical_qubits_per_circuit_min')}, "
        f"max={summary.get('measured_physical_qubits_per_circuit_max')}"
    )


def compile_sampling_circuits(
    sqdrift_circuits: Mapping[str, Any],
    backend: Any,
    optimization_level: int = 3,
    *,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Compile SqDRIFT sampling circuits with final measurements only.

    This stage does not build M3 calibration circuits or submit Sampler jobs.

    This is the compile stage:
        sqdrift circuit records -> final-measurement circuits -> ISA/transpiled circuits
        -> final-measurement-only M3 mappings -> compiled summary

    Returns
    -------
    compiled_run_info
        A run-info-like dictionary containing `records`, `sampling_circuits`,
        `isa_circuits`, `mappings`, `all_mappings`, `used_qubits_for_m3`, and
        `compiled_info`.  Pass this object to `run_sampling_jobs_from_compiled`.
    """
    _require_runtime()
    _require_mthree()

    records = list(sqdrift_circuits["circuit_records"])

    circuits = []
    for record in records:
        qc = add_final_measurement(record["circuit"])
        qc.metadata = {
            "record_id": record["record_id"],
            "k": int(record["k"]),
            "r": int(record["r"]),
            "evolution_time": float(record["evolution_time"]),
        }
        if "qdrift_base_step_time" in record:
            qc.metadata["qdrift_base_step_time"] = float(record["qdrift_base_step_time"])
        circuits.append(qc)

    print("number of sampling circuits:", len(circuits))
    print("compiling sampling circuits...")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_circuits = pm.run(circuits)
    if not isinstance(isa_circuits, list):
        isa_circuits = [isa_circuits]

    for isa, src in zip(isa_circuits, circuits, strict=True):
        isa.metadata = dict(src.metadata or {})

    # M3 is applied only to the final `meas` register.  We compute mappings now
    # because they are properties of the compiled circuit layout, but we do not
    # run M3 calibration until the sampling-submission stage.
    all_mappings = [mthree.utils.final_measurement_mapping(circ) for circ in isa_circuits]
    mappings = [
        _final_meas_mapping_for_circuit(circ, full_mapping)
        for circ, full_mapping in zip(isa_circuits, all_mappings, strict=True)
    ]
    used_qubits = sorted({int(q) for mapping in mappings for q in _mapping_values(mapping)})

    print("M3 final-measurement calibration qubits:", used_qubits)

    compiled_info = _simple_compiled_summary(isa_circuits, mappings)
    if print_summary:
        _print_compiled_summary(compiled_info)

    compiled_run_info = {
        "records": records,
        "sampling_circuits": circuits,
        "isa_circuits": isa_circuits,
        "mappings": mappings,
        "all_mappings": all_mappings,
        "mapping_note": "mappings contain only the final `meas` register; no reset-probe register is present",
        "used_qubits_for_m3": used_qubits,
        "compiled_info": compiled_info,
        "optimization_level": int(optimization_level),
        "backend_name": _backend_name(backend),
        "reset_probe": False,
        "compiled_at_utc": _utc_now_iso(),
        "sampling_submitted": False,
        "module_version": MODULE_VERSION,
    }
    return compiled_run_info


# Semantic alias.
compile_sqdrift_sampling_circuits = compile_sampling_circuits


def run_sampling_jobs_from_compiled(
    compiled_run_info: Mapping[str, Any],
    backend: Any,
    shots: int,
    m3_file: str = "m3_cals.json",
    chunk_size: int = 100,
    max_circuits_per_batch: int = 500,
    batch_max_time: str = "10m",
    *,
    m3_method: str = "balanced",
    dd_enabled: bool = True,
    dd_sequence_type: str = "XpXm",
):
    """Submit already-compiled ISA sampling circuits.

    This is the sampling stage. It performs M3 calibration on the compiled
    final-measurement physical qubits, then submits the ISA circuits to IBM
    Runtime Sampler in Batch mode.

    Parameters
    ----------
    compiled_run_info:
        Output of `compile_sampling_circuits`.
    backend:
        IBM Runtime backend matching the compiled circuits; used here for M3
        calibration and sampling submission.
    shots:
        Shots per circuit.
    m3_file:
        Calibration file path.  M3 calibration is performed here, not in the
        compile stage.
    """
    _require_runtime()
    _require_mthree()

    isa_circuits = list(compiled_run_info["isa_circuits"])
    mappings = list(compiled_run_info["mappings"])
    used_qubits = sorted(
        int(q)
        for q in compiled_run_info.get(
            "used_qubits_for_m3",
            sorted({int(q) for mapping in mappings for q in _mapping_values(mapping)}),
        )
    )

    print("sampling from precompiled ISA circuits:", len(isa_circuits))
    print("M3 final-measurement calibration qubits:", used_qubits)

    mit = mthree.M3Mitigation(backend)
    m3_jobs = mit.cals_from_system(
        qubits=used_qubits,
        method=str(m3_method),
        cals_file=m3_file,
        async_cal=False,
    )

    print("M3 calibration finished.")
    print("M3 calibration file:", m3_file)

    jobs = []
    job_slices: list[tuple[int, int]] = []
    n_circuits = len(isa_circuits)

    for batch_start in range(0, n_circuits, max_circuits_per_batch):
        batch_stop = min(batch_start + max_circuits_per_batch, n_circuits)
        print()
        print(f"opening batch for circuits [{batch_start}, {batch_stop})")

        with Batch(backend=backend, max_time=batch_max_time) as batch:
            sampler = Sampler(mode=batch)
            sampler.options.dynamical_decoupling.enable = bool(dd_enabled)
            if dd_enabled:
                sampler.options.dynamical_decoupling.sequence_type = str(dd_sequence_type)

            for start in range(batch_start, batch_stop, chunk_size):
                stop = min(start + chunk_size, batch_stop)
                chunk = isa_circuits[start:stop]
                job = sampler.run(chunk, shots=int(shots))
                jobs.append(job)
                job_slices.append((int(start), int(stop)))
                print("submitted sampling job:", job.job_id(), "| circuits:", f"[{start}, {stop})")

    run_info = dict(compiled_run_info)
    run_info.update(
        {
            "mitigator": mit,
            "m3_file": m3_file,
            "m3_jobs": m3_jobs,
            "m3_method": str(m3_method),
            "shots": int(shots),
            "sampling_jobs": jobs,
            "job_slices": job_slices,
            "chunk_size": int(chunk_size),
            "max_circuits_per_batch": int(max_circuits_per_batch),
            "batch_max_time": batch_max_time,
            "backend_name": run_info.get("backend_name", _backend_name(backend)),
            "dd_enabled": bool(dd_enabled),
            "dd_sequence_type": str(dd_sequence_type) if dd_enabled else None,
            "created_at_utc": _utc_now_iso(),
            "sampling_submitted": True,
            "module_version": MODULE_VERSION,
        }
    )
    return jobs, run_info


# Semantic aliases.
run_sampling_from_compiled = run_sampling_jobs_from_compiled
run_compiled_sampling_jobs_batch = run_sampling_jobs_from_compiled


def run_sampling_jobs_batch(
    sqdrift_circuits: Mapping[str, Any],
    backend: Any,
    shots: int,
    optimization_level: int = 3,
    m3_file: str = "m3_cals.json",
    chunk_size: int = 100,
    max_circuits_per_batch: int = 500,
    batch_max_time: str = "10m",
):
    """Backward-compatible wrapper.

    Equivalent to:
        compiled = compile_sampling_circuits(...)
        jobs, run_info = run_sampling_jobs_from_compiled(compiled, ...)

    Use the two explicit functions above when you want to inspect/save compiled
    ISA circuits before submitting sampling jobs.
    """
    compiled_run_info = compile_sampling_circuits(
        sqdrift_circuits=sqdrift_circuits,
        backend=backend,
        optimization_level=optimization_level,
        print_summary=True,
    )
    return run_sampling_jobs_from_compiled(
        compiled_run_info,
        backend=backend,
        shots=shots,
        m3_file=m3_file,
        chunk_size=chunk_size,
        max_circuits_per_batch=max_circuits_per_batch,
        batch_max_time=batch_max_time,
    )


# =============================================================================
# Submission and run metadata
# =============================================================================


def _save_qpy(circuits: Sequence[Any], path: str | Path) -> Path:
    """Save Qiskit circuits in plain, uncompressed QPY format.

    QPY is never written through gzip. Qiskit's qpy.dump() seeks
    backwards while finalizing the file table, and gzip write streams cannot
    safely support that. If a caller passes a `.qpy.gz` or `.gz` path, this
    function silently changes it to a plain `.qpy` path before writing.

    The write is atomic: dump to a temporary seekable file, then replace.
    """
    if qpy is None:
        raise ImportError("QPY saving requires qiskit.qpy.")
    path = Path(path)

    # Never save QPY with gzip. This specifically prevents:
    # OSError: Negative seek in write mode
    name = path.name
    if name.endswith(".qpy.gz"):
        path = path.with_name(name[:-3])  # remove only '.gz' -> '.qpy'
    elif path.suffix == ".gz":
        path = path.with_suffix(".qpy")
    elif path.suffix != ".qpy":
        path = path.with_suffix(".qpy")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        qpy.dump(list(circuits), f)
    tmp.replace(path)
    return path


def load_qpy_circuits(path: str | Path) -> list[Any]:
    """Load circuits saved by `_save_qpy` / `save_sampling_submission`.

    Current runs save plain ``.qpy`` files. Existing legacy ``.qpy.gz`` files are also
    readable: the bytes are decompressed into an in-memory seekable buffer before
    calling ``qpy.load``.
    """
    if qpy is None:
        raise ImportError("QPY loading requires qiskit.qpy.")
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
        return list(qpy.load(buffer))
    with path.open("rb") as f:
        return list(qpy.load(f))


def _safe_backend_properties(backend: Any) -> Any | None:
    props_attr = getattr(backend, "properties", None)
    if callable(props_attr):
        try:
            return props_attr()
        except Exception:
            return None
    return props_attr


def _safe_call(obj: Any, name: str, *args: Any) -> Any | None:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    try:
        out = fn(*args)
        if isinstance(out, np.generic):
            out = out.item()
        return out
    except Exception:
        return None


def _target_instruction_property(backend: Any, gate_name: str, qargs: Sequence[int], attr: str) -> Any | None:
    target = getattr(backend, "target", None)
    if target is None:
        return None
    try:
        inst_map = target[gate_name]
        prop = inst_map.get(tuple(qargs), None) if hasattr(inst_map, "get") else inst_map[tuple(qargs)]
        if prop is None:
            return None
        return getattr(prop, attr, None)
    except Exception:
        return None


def _gate_error_duration(backend: Any, props: Any, gate_name: str, qargs: Sequence[int]) -> tuple[float | None, float | None]:
    error = _safe_call(props, "gate_error", gate_name, list(qargs)) if props is not None else None
    duration = _safe_call(props, "gate_length", gate_name, list(qargs)) if props is not None else None
    if error is None:
        error = _target_instruction_property(backend, gate_name, qargs, "error")
    if duration is None:
        duration = _target_instruction_property(backend, gate_name, qargs, "duration")
    return (
        None if error is None else float(error),
        None if duration is None else float(duration),
    )


def _qubit_snapshot(backend: Any, props: Any, qubit: int) -> dict[str, Any]:
    readout_error = _safe_call(props, "readout_error", qubit) if props is not None else None
    if readout_error is None:
        readout_error = _target_instruction_property(backend, "measure", (qubit,), "error")
    measure_duration = _safe_call(props, "gate_length", "measure", [qubit]) if props is not None else None
    if measure_duration is None:
        measure_duration = _target_instruction_property(backend, "measure", (qubit,), "duration")

    return {
        "qubit": int(qubit),
        "T1": _safe_call(props, "t1", qubit) if props is not None else None,
        "T2": _safe_call(props, "t2", qubit) if props is not None else None,
        "frequency": _safe_call(props, "frequency", qubit) if props is not None else None,
        "readout_error": None if readout_error is None else float(readout_error),
        "measure_duration": None if measure_duration is None else float(measure_duration),
    }


def _finite_stats(values: Sequence[float | None]) -> dict[str, float | None]:
    arr = np.array([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def build_hardware_snapshot(
    backend: Any,
    isa_circuits: Sequence[Any],
    mappings: Sequence[Any],
) -> dict[str, Any]:
    """Snapshot the calibration-like information needed for later reporting.

    The snapshot contains available per-qubit, per-edge, and per-circuit
    calibration statistics. Its ``active_physical_qubits`` includes every qubit
    referenced by an instruction, including barriers, delays, and measurements.
    """
    props = _safe_backend_properties(backend)
    backend_name = _backend_name(backend)
    props_date = getattr(props, "last_update_date", None) if props is not None else None

    used_qubits: set[int] = set()
    measured_by_circuit: list[list[int]] = []
    active_by_circuit: list[list[int]] = []
    per_circuit: list[dict[str, Any]] = []
    gate_occurrences: Counter[tuple[str, tuple[int, ...]]] = Counter()
    gate_prop_cache: dict[tuple[str, tuple[int, ...]], tuple[float | None, float | None]] = {}

    for i, circ in enumerate(isa_circuits):
        measured = sorted(_mapping_values(mappings[i]))
        active: set[int] = set()
        oneq_errors: list[float | None] = []
        twoq_errors: list[float | None] = []
        oneq_durations: list[float | None] = []
        twoq_durations: list[float | None] = []
        gate_counts: Counter[str] = Counter()

        for inst in circ.data:
            op = inst.operation
            qinds = tuple(_circuit_qubit_index(circ, q) for q in inst.qubits)
            active.update(qinds)
            if op.name in {"barrier", "delay", "measure"}:
                continue

            gate_counts[str(op.name)] += 1
            key = (str(op.name), tuple(int(q) for q in qinds))
            gate_occurrences[key] += 1
            if key not in gate_prop_cache:
                gate_prop_cache[key] = _gate_error_duration(backend, props, key[0], key[1])
            err, dur = gate_prop_cache[key]

            if len(qinds) == 1:
                oneq_errors.append(err)
                oneq_durations.append(dur)
            elif len(qinds) == 2:
                twoq_errors.append(err)
                twoq_durations.append(dur)

        active_sorted = sorted(active)
        measured_by_circuit.append(measured)
        active_by_circuit.append(active_sorted)
        used_qubits.update(active_sorted)
        used_qubits.update(measured)

        readout_by_q = [_qubit_snapshot(backend, props, q).get("readout_error") for q in measured]
        t1_by_q = [_qubit_snapshot(backend, props, q).get("T1") for q in active_sorted]
        t2_by_q = [_qubit_snapshot(backend, props, q).get("T2") for q in active_sorted]

        per_circuit.append(
            {
                "circuit_index": int(i),
                "name": circ.name,
                "active_physical_qubits": active_sorted,
                "num_active_physical_qubits": int(len(active_sorted)),
                "measured_physical_qubits": measured,
                "num_measured_physical_qubits": int(len(measured)),
                "gate_counts_by_name": dict(gate_counts),
                "readout_error": _finite_stats(readout_by_q),
                "T1": _finite_stats(t1_by_q),
                "T2": _finite_stats(t2_by_q),
                "oneq_gate_error": _finite_stats(oneq_errors),
                "twoq_gate_error": _finite_stats(twoq_errors),
                "oneq_gate_duration": _finite_stats(oneq_durations),
                "twoq_gate_duration": _finite_stats(twoq_durations),
            }
        )

    per_qubit = [_qubit_snapshot(backend, props, q) for q in sorted(used_qubits)]
    per_gate_or_edge = []
    for (gate, qargs), count in sorted(gate_occurrences.items(), key=lambda x: (x[0][0], x[0][1])):
        err, dur = gate_prop_cache[(gate, qargs)]
        per_gate_or_edge.append(
            {
                "gate": gate,
                "qubits": list(qargs),
                "used_count_total": int(count),
                "gate_error": err,
                "gate_duration": dur,
            }
        )

    return {
        "snapshot_created_at_utc": _utc_now_iso(),
        "backend_name": _backend_name(backend),
        "backend_version": getattr(backend, "backend_version", None),
        "properties_last_update_date": _jsonify(props_date),
        "used_physical_qubits_union": sorted(used_qubits),
        "measured_physical_qubits_union": sorted({q for qs in measured_by_circuit for q in qs}),
        "per_qubit": per_qubit,
        "per_gate_or_edge": per_gate_or_edge,
        "per_circuit_hardware_summary": per_circuit,
    }


def build_record_table(run_info: Mapping[str, Any], jobs: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    """Combine SqDRIFT records, compiled summaries, and job slices per circuit."""
    records = list(run_info.get("records", []))
    compiled_rows = {
        int(row["circuit_index"]): row
        for row in run_info.get("compiled_info", {}).get("per_circuit", [])
    }
    jobs_eff = list(jobs if jobs is not None else run_info.get("sampling_jobs", []))
    job_slices = list(run_info.get("job_slices", []))

    job_lookup: dict[int, dict[str, Any]] = {}
    for job_pos, (job, slc) in enumerate(zip(jobs_eff, job_slices)):
        start, stop = int(slc[0]), int(slc[1])
        for local_idx, circuit_idx in enumerate(range(start, stop)):
            job_lookup[int(circuit_idx)] = {
                "job_id": _job_id(job),
                "job_position": int(job_pos),
                "job_local_index": int(local_idx),
                "job_slice_start": start,
                "job_slice_stop": stop,
            }

    table: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        row = {
            "circuit_index": int(i),
            "record_id": record.get("record_id"),
            "k": record.get("k"),
            "r": record.get("r"),
            "evolution_time": record.get("evolution_time"),
            "qdrift_base_step_time": record.get("qdrift_base_step_time"),
            "sampled_term_indices": record.get("sampled_term_indices", []),
            "sampled_pauli_labels": record.get("sampled_pauli_labels", []),
        }
        row.update(job_lookup.get(i, {}))
        if i in compiled_rows:
            row["compiled"] = compiled_rows[i]
        table.append(_jsonify(row))
    return table


def save_sampling_submission(
    run_dir: str | Path,
    jobs: Sequence[Any],
    run_info: Mapping[str, Any],
    *,
    backend: Any | None = None,
    sqdrift_circuits: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    save_qpy: bool = True,
    save_hardware_snapshot: bool = True,
    fail_on_qpy_error: bool = False,
) -> dict[str, str]:
    """Save sampling job metadata and optional run artifacts.

    The saved artifacts include job IDs, serializable run info, record table,
    compiled info, optional hardware snapshot, M3 calibration file copy, and QPY
    circuits.

    QPY files are saved uncompressed by default (`.qpy`, not
    `.qpy.gz`) to avoid Qiskit QPY backward-seek failures on gzip streams.
    QPY save failures are logged to `qpy_save_errors.json` and do not abort
    metadata saving unless `fail_on_qpy_error=True`.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    circuits_dir = run_dir / "circuits"
    saved: dict[str, str] = {}

    sampling_job_ids = _job_ids(jobs)
    m3_job_ids = _job_ids(run_info.get("m3_jobs"))

    manifest = {
        "module": "sampling",
        "module_version": MODULE_VERSION,
        "created_at_utc": _utc_now_iso(),
        "backend_name": run_info.get("backend_name", _backend_name(backend)),
        "shots": run_info.get("shots"),
        "optimization_level": run_info.get("optimization_level"),
        "chunk_size": run_info.get("chunk_size"),
        "max_circuits_per_batch": run_info.get("max_circuits_per_batch"),
        "batch_max_time": run_info.get("batch_max_time"),
        "dd_enabled": run_info.get("dd_enabled"),
        "dd_sequence_type": run_info.get("dd_sequence_type"),
        "reset_probe": run_info.get("reset_probe"),
        "m3_file": run_info.get("m3_file"),
        "sampling_job_ids": sampling_job_ids,
        "m3_job_ids": m3_job_ids,
        "job_slices": run_info.get("job_slices"),
        "extra_metadata": dict(extra_metadata or {}),
        "qpy_format": "plain_seekable_qpy_no_gzip",
    }
    saved["manifest"] = str(_write_json(run_dir / "manifest.json", manifest))

    record_table = build_record_table(run_info, jobs)
    saved["record_table"] = str(_write_json(run_dir / "record_table.json", record_table))
    saved["compiled_info"] = str(_write_json(run_dir / "compiled_info.json", run_info.get("compiled_info", {})))

    serializable_run_info = {
        k: v
        for k, v in run_info.items()
        if k not in {"mitigator", "sampling_jobs", "m3_jobs", "sampling_circuits", "isa_circuits"}
    }
    serializable_run_info["sampling_job_ids"] = sampling_job_ids
    serializable_run_info["m3_job_ids"] = m3_job_ids
    saved["run_info_serializable"] = str(_write_json(run_dir / "run_info_serializable.json", serializable_run_info))

    # Copy M3 calibration file if it exists.
    m3_file = run_info.get("m3_file")
    if m3_file is not None:
        src = Path(str(m3_file))
        if src.exists():
            dst = run_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            saved["m3_cals"] = str(dst)

    if save_hardware_snapshot and backend is not None and "isa_circuits" in run_info and "mappings" in run_info:
        snapshot = build_hardware_snapshot(backend, run_info["isa_circuits"], run_info["mappings"])
        saved["hardware_snapshot"] = str(_write_json(run_dir / "hardware_snapshot.json", snapshot))

    if save_qpy:
        qpy_errors: list[dict[str, Any]] = []

        def _attempt_qpy(key: str, circuits_obj: Sequence[Any], file_name: str) -> None:
            try:
                saved[key] = str(_save_qpy(circuits_obj, circuits_dir / file_name))
            except Exception as exc:  # intentionally keep submission metadata safe
                err = {
                    "key": key,
                    "file_name": file_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                qpy_errors.append(err)
                if fail_on_qpy_error:
                    raise

        if sqdrift_circuits is not None and "circuit_records" in sqdrift_circuits:
            src_circuits = [r["circuit"] for r in sqdrift_circuits["circuit_records"]]
            _attempt_qpy("sqdrift_circuits_qpy", src_circuits, "sqdrift_circuits_no_measure.qpy")

        sampling_circuits = run_info.get("sampling_circuits")
        if sampling_circuits is None and "records" in run_info:
            sampling_circuits = []
            for record in run_info["records"]:
                if "circuit" in record:
                    qc = add_final_measurement(record["circuit"])
                    qc.metadata = {
                        "record_id": record.get("record_id"),
                        "k": record.get("k"),
                        "r": record.get("r"),
                        "evolution_time": record.get("evolution_time"),
                    }
                    sampling_circuits.append(qc)
        if sampling_circuits is not None:
            _attempt_qpy("sampling_circuits_qpy", sampling_circuits, "sampling_circuits_final_meas_only.qpy")

        if "isa_circuits" in run_info:
            _attempt_qpy("isa_circuits_qpy", run_info["isa_circuits"], "isa_circuits_transpiled.qpy")

        if qpy_errors:
            saved["qpy_save_errors"] = str(_write_json(run_dir / "qpy_save_errors.json", qpy_errors))

    saved["saved_paths"] = str(_write_json(run_dir / "saved_paths.json", saved))
    return saved



def save_compiled_sampling_run(
    run_dir: str | Path,
    compiled_run_info: Mapping[str, Any],
    *,
    backend: Any | None = None,
    sqdrift_circuits: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    save_qpy: bool = True,
    save_hardware_snapshot: bool = True,
    fail_on_qpy_error: bool = False,
) -> dict[str, str]:
    """Save a compile-only checkpoint before M3 calibration / sampling submission.

    A checkpoint can be reloaded for submission only when its ISA circuits were
    successfully saved as QPY.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    circuits_dir = run_dir / "circuits"
    saved: dict[str, str] = {}

    manifest = {
        "module": "sampling",
        "module_version": MODULE_VERSION,
        "stage": "compiled_only",
        "created_at_utc": _utc_now_iso(),
        "backend_name": compiled_run_info.get("backend_name", _backend_name(backend)),
        "optimization_level": compiled_run_info.get("optimization_level"),
        "reset_probe": compiled_run_info.get("reset_probe"),
        "used_qubits_for_m3": compiled_run_info.get("used_qubits_for_m3"),
        "sampling_submitted": False,
        "extra_metadata": dict(extra_metadata or {}),
        "qpy_format": "plain_seekable_qpy_no_gzip",
    }
    saved["compiled_manifest"] = str(_write_json(run_dir / "compiled_manifest.json", manifest))

    record_table = build_record_table(compiled_run_info, jobs=[])
    saved["record_table"] = str(_write_json(run_dir / "record_table.json", record_table))
    saved["compiled_info"] = str(_write_json(run_dir / "compiled_info.json", compiled_run_info.get("compiled_info", {})))

    serializable = {
        k: v
        for k, v in compiled_run_info.items()
        if k not in {"sampling_circuits", "isa_circuits"}
    }
    saved["run_info_compiled_serializable"] = str(_write_json(run_dir / "run_info_compiled_serializable.json", serializable))

    if save_hardware_snapshot and backend is not None and "isa_circuits" in compiled_run_info and "mappings" in compiled_run_info:
        snapshot = build_hardware_snapshot(backend, compiled_run_info["isa_circuits"], compiled_run_info["mappings"])
        saved["hardware_snapshot"] = str(_write_json(run_dir / "hardware_snapshot.json", snapshot))

    if save_qpy:
        qpy_errors: list[dict[str, Any]] = []

        def _attempt_qpy(key: str, circuits_obj: Sequence[Any], file_name: str) -> None:
            try:
                saved[key] = str(_save_qpy(circuits_obj, circuits_dir / file_name))
            except Exception as exc:
                err = {
                    "key": key,
                    "file_name": file_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                qpy_errors.append(err)
                if fail_on_qpy_error:
                    raise

        if sqdrift_circuits is not None and "circuit_records" in sqdrift_circuits:
            src_circuits = [r["circuit"] for r in sqdrift_circuits["circuit_records"]]
            _attempt_qpy("sqdrift_circuits_qpy", src_circuits, "sqdrift_circuits_no_measure.qpy")
        if "sampling_circuits" in compiled_run_info:
            _attempt_qpy("sampling_circuits_qpy", compiled_run_info["sampling_circuits"], "sampling_circuits_final_meas_only.qpy")
        if "isa_circuits" in compiled_run_info:
            _attempt_qpy("isa_circuits_qpy", compiled_run_info["isa_circuits"], "isa_circuits_transpiled.qpy")
        if qpy_errors:
            saved["qpy_save_errors"] = str(_write_json(run_dir / "qpy_save_errors.json", qpy_errors))

    saved["saved_paths_compiled"] = str(_write_json(run_dir / "saved_paths_compiled.json", saved))
    return saved


def load_compiled_run_info(run_dir: str | Path) -> dict[str, Any]:
    """Load a compile checkpoint that contains a saved ISA-circuit QPY file."""
    run_dir = Path(run_dir)
    info_path = run_dir / "run_info_compiled_serializable.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"No compile-only run info found at {info_path}. "
            "Call save_compiled_sampling_run(...) after compile_sampling_circuits(...)."
        )
    info = _read_json(info_path)
    info["records"] = _read_json(run_dir / "record_table.json")

    saved_paths_path = run_dir / "saved_paths_compiled.json"
    if saved_paths_path.exists():
        saved_paths = _read_json(saved_paths_path)
    else:
        saved_paths = {}

    isa_path = saved_paths.get("isa_circuits_qpy", str(run_dir / "circuits" / "isa_circuits_transpiled.qpy"))
    if Path(isa_path).exists():
        info["isa_circuits"] = load_qpy_circuits(isa_path)
    else:
        raise FileNotFoundError(f"Compiled ISA QPY file not found: {isa_path}")

    sampling_path = saved_paths.get("sampling_circuits_qpy", str(run_dir / "circuits" / "sampling_circuits_final_meas_only.qpy"))
    if Path(sampling_path).exists():
        info["sampling_circuits"] = load_qpy_circuits(sampling_path)

    return info


def load_sampling_jobs_from_manifest(service: Any, run_dir_or_manifest: str | Path) -> tuple[list[Any], dict[str, Any]]:
    """Load sampling jobs from `manifest.json` using `QiskitRuntimeService.job`."""
    p = Path(run_dir_or_manifest)
    manifest_path = p / "manifest.json" if p.is_dir() else p
    manifest = _read_json(manifest_path)
    jobs = [service.job(job_id) for job_id in manifest["sampling_job_ids"]]
    return jobs, manifest


def load_saved_run_info(run_dir: str | Path) -> dict[str, Any]:
    """Load the minimal serializable run_info needed for result post-processing.

    This is intended for later sessions where the original in-memory `run_info`
    object is unavailable. For M3-on processing you still need either a live
    `run_info['mitigator']` or to pass `backend`/`m3_file` to the processing
    function so the mitigator can be reconstructed.
    """
    run_dir = Path(run_dir)
    info = _read_json(run_dir / "run_info_serializable.json")
    record_table = _read_json(run_dir / "record_table.json")
    info["records"] = record_table
    return info


# =============================================================================
# Job results to realization-level post-processing records
# =============================================================================


def _get_databin_register(pub_result: Any, name: str) -> Any:
    data = pub_result.data
    if hasattr(data, name):
        return getattr(data, name)
    try:
        return data[name]
    except Exception as exc:
        raise KeyError(f"Sampler result data has no classical register {name!r}.") from exc


def _normalise_bit_key(key: Any, n_bits: int) -> str:
    if isinstance(key, (int, np.integer)):
        return format(int(key), f"0{n_bits}b")
    s = str(key).replace(" ", "")
    if s.startswith("0x"):
        return format(int(s, 16), f"0{n_bits}b")
    if len(s) < n_bits:
        s = s.zfill(n_bits)
    if len(s) != n_bits or any(ch not in "01" for ch in s):
        raise ValueError(f"Invalid {n_bits}-bit key: {key!r}")
    return s


def _bitstrings_to_uint8(bitstrings: Sequence[str], n_bits: int) -> np.ndarray:
    if len(bitstrings) == 0:
        return np.empty((0, n_bits), dtype=np.uint8)
    return np.asarray([[1 if ch == "1" else 0 for ch in s] for s in bitstrings], dtype=np.uint8)


def _counts_to_arrays(counts: Mapping[Any, Any], n_bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    items = [(_normalise_bit_key(k, n_bits), float(v)) for k, v in counts.items() if float(v) != 0.0]
    if len(items) == 0:
        return (
            np.empty((0, n_bits), dtype=np.uint8),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    keys = [k for k, _ in items]
    raw_values = np.asarray([v for _, v in items], dtype=np.float64)
    weights = raw_values.copy()
    total = float(np.sum(weights))
    if total > 0.0:
        weights /= total
    else:
        weights[:] = 0.0
    return _bitstrings_to_uint8(keys, n_bits), weights, raw_values


def _distribution_to_dict(dist: Any, n_bits: int) -> dict[str, float]:
    if dist is None:
        return {}
    if hasattr(dist, "items"):
        items = dist.items()
    elif hasattr(dist, "binary_probabilities") and callable(dist.binary_probabilities):
        items = dist.binary_probabilities().items()
    else:
        items = dict(dist).items()

    out: dict[str, float] = {}
    for k, v in items:
        val = float(v)
        if math.isfinite(val) and val != 0.0:
            out[_normalise_bit_key(k, n_bits)] = val
    return out


def _project_quasi_to_probability(raw_quasi: Any, n_bits: int) -> tuple[dict[str, float], dict[str, Any]]:
    raw = _distribution_to_dict(raw_quasi, n_bits)
    negative_mass = float(sum(-v for v in raw.values() if v < 0.0))

    if hasattr(raw_quasi, "nearest_probability_distribution") and callable(raw_quasi.nearest_probability_distribution):
        projected_obj = raw_quasi.nearest_probability_distribution()
        projected = _distribution_to_dict(projected_obj, n_bits)
        method = "nearest_probability_distribution"
    else:
        projected = {k: max(0.0, v) for k, v in raw.items()}
        method = "clip_and_renormalize"

    total = float(sum(projected.values()))
    if total > 0.0:
        projected = {k: v / total for k, v in projected.items() if v > 0.0}
    else:
        projected = {}

    all_keys = set(raw) | set(projected)
    l1_change = float(sum(abs(projected.get(k, 0.0) - raw.get(k, 0.0)) for k in all_keys))
    meta = {
        "projection_method": method,
        "negative_mass": negative_mass,
        "clip_l1_change": l1_change,
        "raw_quasi_sum": float(sum(raw.values())),
        "projected_sum": float(sum(projected.values())),
    }
    return projected, meta


def _prob_dict_to_arrays(prob: Mapping[Any, Any], n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    items = [(_normalise_bit_key(k, n_bits), float(v)) for k, v in prob.items() if float(v) != 0.0]
    if len(items) == 0:
        return np.empty((0, n_bits), dtype=np.uint8), np.empty(0, dtype=np.float64)
    keys = [k for k, _ in items]
    vals = np.asarray([v for _, v in items], dtype=np.float64)
    return _bitstrings_to_uint8(keys, n_bits), vals


def load_m3_mitigator(backend: Any, m3_file: str | Path):
    """Reconstruct an M3 mitigator from a saved calibration file."""
    _require_mthree()
    mit = mthree.M3Mitigation(backend)
    for method_name in ("cals_from_file", "load_cals_from_file", "readout_cals_from_file"):
        method = getattr(mit, method_name, None)
        if callable(method):
            method(str(m3_file))
            return mit
    raise AttributeError(
        "This mthree version does not expose a recognized M3 calibration-file loader. "
        "Pass the live run_info['mitigator'] instead."
    )


def _make_branch_record(
    *,
    variant: str,
    circuit_index: int,
    record: Mapping[str, Any],
    job_id: str,
    job_local_index: int,
    counts: Mapping[Any, Any],
    n_bits: int,
    shots: int | None,
    accepted_shots: int | None,
    mapping: Any,
    mitigator: Any | None,
) -> dict[str, Any]:
    m3_enabled = variant.startswith("m3_on")

    count_bits, count_weights, count_values = _counts_to_arrays(counts, n_bits)
    raw_quasi_bits = np.empty((0, n_bits), dtype=np.uint8)
    raw_quasi_weights = np.empty(0, dtype=np.float64)
    m3_meta: dict[str, Any] = {}

    if m3_enabled:
        if mitigator is None:
            raise ValueError("M3-on branch requested but no mitigator is available.")
        raw_quasi = mitigator.apply_correction(dict(counts), _m3_qubits_for_counts(mapping, n_bits)) if len(counts) else {}
        raw_quasi_dict = _distribution_to_dict(raw_quasi, n_bits)
        raw_quasi_bits, raw_quasi_weights = _prob_dict_to_arrays(raw_quasi_dict, n_bits)
        projected, m3_meta = _project_quasi_to_probability(raw_quasi, n_bits)
        bitstrings, weights = _prob_dict_to_arrays(projected, n_bits)
    else:
        bitstrings = count_bits
        weights = count_weights

    metadata = {
        "variant": variant,
        "circuit_index": int(circuit_index),
        "record_id": record.get("record_id"),
        "k": record.get("k"),
        "r": record.get("r"),
        "evolution_time": record.get("evolution_time"),
        "qdrift_base_step_time": record.get("qdrift_base_step_time"),
        "sampled_term_indices": record.get("sampled_term_indices", []),
        "sampled_pauli_labels": record.get("sampled_pauli_labels", []),
        "job_id": job_id,
        "job_local_index": int(job_local_index),
        "n_bits": int(n_bits),
        "shots": None if shots is None else int(shots),
        "accepted_shots": None if accepted_shots is None else int(accepted_shots),
        # Retained in the artifact schema; current circuits are never reset-postselected.
        "reset_postselection": False,
        "m3_enabled": bool(m3_enabled),
        "m3": m3_meta,
        "created_at_utc": _utc_now_iso(),
    }

    return {
        "bitstrings": np.ascontiguousarray(bitstrings, dtype=np.uint8),
        "weights": np.ascontiguousarray(weights, dtype=np.float64),
        "counts_bitstrings": np.ascontiguousarray(count_bits, dtype=np.uint8),
        "counts_values": np.ascontiguousarray(count_values, dtype=np.float64),
        "raw_quasi_bitstrings": np.ascontiguousarray(raw_quasi_bits, dtype=np.uint8),
        "raw_quasi_weights": np.ascontiguousarray(raw_quasi_weights, dtype=np.float64),
        "metadata": metadata,
    }


def make_postprocessing_branches_from_batch_jobs(
    jobs: Sequence[Any],
    run_info: Mapping[str, Any],
    *,
    backend: Any | None = None,
    m3_file: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Convert IBM Runtime SamplerV2 jobs into two post-processing branches.

    Branches (identifiers retained for artifact compatibility):
        m3_off_reset_off: normalized unmitigated empirical probabilities
        m3_on_reset_off:  M3 quasi-distribution projected to probabilities

    Raw integer counts are stored separately in each branch record.
    """
    records = list(run_info["records"])
    mappings = list(run_info["mappings"])
    job_slices = list(run_info["job_slices"])
    shots = run_info.get("shots")

    mitigator = run_info.get("mitigator")
    if mitigator is None and backend is not None:
        m3_path = m3_file or run_info.get("m3_file")
        if m3_path is not None:
            mitigator = load_m3_mitigator(backend, m3_path)

    branches: dict[str, list[dict[str, Any]]] = {name: [] for name in POSTPROCESSING_BRANCHES}

    for job, slc in zip(jobs, job_slices, strict=True):
        start, stop = int(slc[0]), int(slc[1])
        result = job.result()
        pub_results = list(result)
        if len(pub_results) != stop - start:
            raise ValueError(
                f"Job {_job_id(job)} returned {len(pub_results)} PUB results, "
                f"but job slice [{start}, {stop}) has {stop - start} circuits."
            )

        for local_idx, pub in enumerate(pub_results):
            circuit_index = start + local_idx
            record = records[circuit_index]
            mapping = mappings[circuit_index]
            meas = _get_databin_register(pub, "meas")
            n_bits = int(meas.num_bits)

            raw_counts = meas.get_counts()
            total_shots = int(shots) if shots is not None else int(sum(int(v) for v in raw_counts.values()))

            for variant, counts, accepted in (
                ("m3_off_reset_off", raw_counts, total_shots),
                ("m3_on_reset_off", raw_counts, total_shots),
            ):
                branches[variant].append(
                    _make_branch_record(
                        variant=variant,
                        circuit_index=circuit_index,
                        record=record,
                        job_id=_job_id(job),
                        job_local_index=local_idx,
                        counts=counts,
                        n_bits=n_bits,
                        shots=shots,
                        accepted_shots=accepted,
                        mapping=mapping,
                        mitigator=mitigator,
                    )
                )

    return branches


def _safe_filename_piece(text: Any) -> str:
    s = str(text if text is not None else "record")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


def save_branch_npz(
    branches: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_root: str | Path = "processed",
    prefix: str = "sqdrift",
) -> dict[str, list[str]]:
    """Save realization-level branch records to branch folders of `.npz` files."""
    output_root = Path(output_root)
    saved: dict[str, list[str]] = {}

    for variant, records in branches.items():
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        saved[variant] = []

        for rec in records:
            meta = dict(rec["metadata"])
            idx = int(meta["circuit_index"])
            record_id = _safe_filename_piece(meta.get("record_id", f"circuit_{idx:04d}"))
            filename = f"{prefix}_circuit_{idx:04d}_{record_id}.npz"
            path = variant_dir / filename

            np.savez_compressed(
                path,
                bitstrings=np.asarray(rec["bitstrings"], dtype=np.uint8),
                weights=np.asarray(rec["weights"], dtype=np.float64),
                counts_bitstrings=np.asarray(rec["counts_bitstrings"], dtype=np.uint8),
                counts_values=np.asarray(rec["counts_values"], dtype=np.float64),
                raw_quasi_bitstrings=np.asarray(rec["raw_quasi_bitstrings"], dtype=np.uint8),
                raw_quasi_weights=np.asarray(rec["raw_quasi_weights"], dtype=np.float64),
                metadata_json=np.array(json.dumps(_jsonify(meta), ensure_ascii=False)),
            )
            saved[variant].append(str(path))

    _write_json(output_root / "saved_branch_files.json", saved)
    return saved


def _branch_npz_path(
    output_root: str | Path,
    variant: str,
    rec: Mapping[str, Any],
    *,
    prefix: str,
) -> Path:
    """Return the deterministic file path for one realization-level branch record."""
    meta = dict(rec["metadata"])
    idx = int(meta["circuit_index"])
    record_id = _safe_filename_piece(meta.get("record_id", f"circuit_{idx:04d}"))
    filename = f"{prefix}_circuit_{idx:04d}_{record_id}.npz"
    return Path(output_root) / variant / filename


def _save_single_branch_record_npz(
    rec: Mapping[str, Any],
    *,
    output_root: str | Path,
    variant: str,
    prefix: str,
) -> Path:
    """Save one realization-level branch record immediately and return its path."""
    path = _branch_npz_path(output_root, variant, rec, prefix=prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(rec["metadata"])
    np.savez_compressed(
        path,
        bitstrings=np.asarray(rec["bitstrings"], dtype=np.uint8),
        weights=np.asarray(rec["weights"], dtype=np.float64),
        counts_bitstrings=np.asarray(rec["counts_bitstrings"], dtype=np.uint8),
        counts_values=np.asarray(rec["counts_values"], dtype=np.float64),
        raw_quasi_bitstrings=np.asarray(rec["raw_quasi_bitstrings"], dtype=np.uint8),
        raw_quasi_weights=np.asarray(rec["raw_quasi_weights"], dtype=np.float64),
        metadata_json=np.array(json.dumps(_jsonify(meta), ensure_ascii=False)),
    )
    return path


def iter_postprocessing_branch_records_from_batch_jobs(
    jobs: Sequence[Any],
    run_info: Mapping[str, Any],
    *,
    backend: Any | None = None,
    m3_file: str | Path | None = None,
):
    """Yield (variant, record) pairs without keeping all processed results in memory."""
    records = list(run_info["records"])
    mappings = list(run_info["mappings"])
    job_slices = list(run_info["job_slices"])
    shots = run_info.get("shots")

    mitigator = run_info.get("mitigator")
    if mitigator is None and backend is not None:
        m3_path = m3_file or run_info.get("m3_file")
        if m3_path is not None:
            mitigator = load_m3_mitigator(backend, m3_path)

    for job, slc in zip(jobs, job_slices, strict=True):
        start, stop = int(slc[0]), int(slc[1])
        result = job.result()
        pub_results = list(result)
        if len(pub_results) != stop - start:
            raise ValueError(
                f"Job {_job_id(job)} returned {len(pub_results)} PUB results, "
                f"but job slice [{start}, {stop}) has {stop - start} circuits."
            )

        for local_idx, pub in enumerate(pub_results):
            circuit_index = start + local_idx
            record = records[circuit_index]
            mapping = mappings[circuit_index]
            meas = _get_databin_register(pub, "meas")
            n_bits = int(meas.num_bits)

            raw_counts = meas.get_counts()
            total_shots = int(shots) if shots is not None else int(sum(int(v) for v in raw_counts.values()))

            for variant, counts, accepted in (
                ("m3_off_reset_off", raw_counts, total_shots),
                ("m3_on_reset_off", raw_counts, total_shots),
            ):
                yield variant, _make_branch_record(
                    variant=variant,
                    circuit_index=circuit_index,
                    record=record,
                    job_id=_job_id(job),
                    job_local_index=local_idx,
                    counts=counts,
                    n_bits=n_bits,
                    shots=shots,
                    accepted_shots=accepted,
                    mapping=mapping,
                    mitigator=mitigator,
                )


def process_and_save_sampling_jobs(
    jobs: Sequence[Any],
    run_info: Mapping[str, Any],
    *,
    output_root: str | Path = "processed",
    prefix: str = "sqdrift",
    backend: Any | None = None,
    m3_file: str | Path | None = None,
) -> dict[str, list[str]]:
    """Stream job results to two branch folders as `.npz` files.

    Saving each decoded record immediately reduces peak array memory relative to
    materializing all branch records first.
    """
    output_root = Path(output_root)
    saved: dict[str, list[str]] = {name: [] for name in POSTPROCESSING_BRANCHES}

    for variant, rec in iter_postprocessing_branch_records_from_batch_jobs(
        jobs,
        run_info,
        backend=backend,
        m3_file=m3_file,
    ):
        path = _save_single_branch_record_npz(
            rec,
            output_root=output_root,
            variant=variant,
            prefix=prefix,
        )
        saved.setdefault(variant, []).append(str(path))

    _write_json(output_root / "saved_branch_files.json", saved)
    return saved


def read_branch_npz_metadata(path: str | Path) -> dict[str, Any]:
    """Read the JSON metadata stored in one realization-level `.npz` file."""
    with np.load(path, allow_pickle=False) as data:
        raw = data["metadata_json"]
        text = str(raw.item() if hasattr(raw, "item") else raw)
    return json.loads(text)


# =============================================================================
# Branch folders to clustering-input files
# =============================================================================


def _row_keys_uint8(bitstrings: np.ndarray) -> np.ndarray:
    bitstrings = np.ascontiguousarray(bitstrings, dtype=np.uint8)
    if bitstrings.ndim != 2:
        raise ValueError("bitstrings must be a 2D array.")
    row_dtype = np.dtype((np.void, bitstrings.dtype.itemsize * bitstrings.shape[1]))
    return bitstrings.view(row_dtype).reshape(-1)


def _merge_duplicate_rows_sum_weights(bitstrings: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bitstrings = np.asarray(bitstrings, dtype=np.uint8)
    weights = np.asarray(weights, dtype=np.float64)
    if bitstrings.ndim != 2:
        raise ValueError("bitstrings must be 2D.")
    if weights.ndim != 1 or len(weights) != len(bitstrings):
        raise ValueError("weights must be 1D with the same row count as bitstrings.")
    if len(bitstrings) == 0:
        return bitstrings.copy(), weights.copy()

    bits_c = np.ascontiguousarray(bitstrings, dtype=np.uint8)
    keys = _row_keys_uint8(bits_c)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    out_w = np.zeros(len(first), dtype=np.float64)
    np.add.at(out_w, inverse, weights)
    return bits_c[first].astype(np.uint8, copy=True), out_w


def make_clustering_input_from_npz_folder(
    folder: str | Path,
    *,
    output_file: str | Path | None = None,
    normalize: bool = True,
    min_weight: float = 0.0,
) -> dict[str, Any]:
    """Merge realization-level branch `.npz` files into one clustering input file.

    Duplicate rows are merged before the strict ``weight > min_weight`` filter.
    Empty realization files are skipped. The output stores ``bitstrings``,
    ``probabilities`` (unnormalized weights when ``normalize=False``), source
    files, counts, total retained weight, and the normalization flag.
    """
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*.npz") if not p.name.startswith("clustering_input"))
    if len(files) == 0:
        raise FileNotFoundError(f"No realization .npz files found in {folder}.")

    bits_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    source_files: list[str] = []

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            bits = np.asarray(data["bitstrings"], dtype=np.uint8)
            weights = np.asarray(data["weights"], dtype=np.float64)
        if len(bits) == 0:
            continue
        bits_list.append(bits)
        weights_list.append(weights)
        source_files.append(str(path))

    if len(bits_list) == 0:
        raise ValueError(f"All realization files in {folder} were empty.")

    bitstrings = np.vstack(bits_list).astype(np.uint8, copy=False)
    weights = np.concatenate(weights_list).astype(np.float64, copy=False)

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain non-finite values.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative. Check M3 projection/clipping.")

    bitstrings, weights = _merge_duplicate_rows_sum_weights(bitstrings, weights)
    keep = weights > float(min_weight)
    bitstrings = bitstrings[keep]
    weights = weights[keep]

    total_before_normalization = float(np.sum(weights))
    if total_before_normalization <= 0.0:
        raise ValueError("Merged branch has zero total weight.")

    probabilities = weights / total_before_normalization if normalize else weights
    probabilities = np.ascontiguousarray(probabilities, dtype=np.float64)
    bitstrings = np.ascontiguousarray(bitstrings, dtype=np.uint8)

    if output_file is None:
        output_file = folder / "clustering_input.npz"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        bitstrings=bitstrings,
        probabilities=probabilities,
        source_files=np.asarray(source_files, dtype=str),
        n_realization_files=np.array(len(source_files), dtype=np.int64),
        n_unique=np.array(len(bitstrings), dtype=np.int64),
        total_weight_before_normalization=np.array(total_before_normalization, dtype=np.float64),
        normalized=np.array(bool(normalize)),
    )

    return {
        "output_file": str(output_file),
        "bitstrings": bitstrings,
        "probabilities": probabilities,
        "n_unique": int(len(bitstrings)),
        "total_weight_before_normalization": total_before_normalization,
        "source_files": source_files,
    }


def make_all_clustering_inputs(
    processed_root: str | Path = "processed",
    *,
    output_dir: str | Path | None = None,
    variants: Sequence[str] = POSTPROCESSING_BRANCHES,
) -> dict[str, dict[str, Any]]:
    """Create one clustering-input `.npz` for each branch folder."""
    processed_root = Path(processed_root)
    if output_dir is None:
        output_dir = processed_root.parent / "clustering_inputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, dict[str, Any]] = {}
    for variant in variants:
        folder = processed_root / variant
        out_file = output_dir / f"{variant}.npz"
        out[variant] = make_clustering_input_from_npz_folder(folder, output_file=out_file)
    _write_json(output_dir / "clustering_inputs_summary.json", {k: {kk: vv for kk, vv in v.items() if kk not in {"bitstrings", "probabilities"}} for k, v in out.items()})
    return out


def load_clustering_input(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load arrays ready for `assign_clusters_bmm(bitstrings, probabilities, k=...)`."""
    with np.load(path, allow_pickle=False) as data:
        return (
            np.ascontiguousarray(data["bitstrings"], dtype=np.uint8),
            np.ascontiguousarray(data["probabilities"], dtype=np.float64),
        )


# Backward-compatible aliases for the staged workflow.
process_sampling_jobs_to_npz = process_and_save_sampling_jobs
make_clustering_inputs_from_processed = make_all_clustering_inputs


def merge_npz_folder_for_clustering(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for ``make_clustering_input_from_npz_folder``."""
    return make_clustering_input_from_npz_folder(*args, **kwargs)


def build_all_clustering_inputs(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Alias for ``make_all_clustering_inputs``."""
    return make_all_clustering_inputs(*args, **kwargs)


def load_clustering_input_npz(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Alias for ``load_clustering_input`` retaining the `.npz` naming convention."""
    return load_clustering_input(*args, **kwargs)


__all__ = [
    "POSTPROCESSING_BRANCHES",
    "make_logical_ghz_initial_state",
    "add_final_measurement",
    "_mapping_values",
    "_simple_compiled_summary",
    "compile_sampling_circuits",
    "compile_sqdrift_sampling_circuits",
    "run_sampling_jobs_from_compiled",
    "run_sampling_from_compiled",
    "run_compiled_sampling_jobs_batch",
    "run_sampling_jobs_batch",
    "save_sampling_submission",
    "save_compiled_sampling_run",
    "load_compiled_run_info",
    "load_sampling_jobs_from_manifest",
    "load_saved_run_info",
    "load_qpy_circuits",
    "build_hardware_snapshot",
    "build_record_table",
    "load_m3_mitigator",
    "make_postprocessing_branches_from_batch_jobs",
    "iter_postprocessing_branch_records_from_batch_jobs",
    "save_branch_npz",
    "process_and_save_sampling_jobs",
    "read_branch_npz_metadata",
    "process_sampling_jobs_to_npz",
    "make_clustering_input_from_npz_folder",
    "merge_npz_folder_for_clustering",
    "make_all_clustering_inputs",
    "make_clustering_inputs_from_processed",
    "load_clustering_input",
    "load_clustering_input_npz",
]
