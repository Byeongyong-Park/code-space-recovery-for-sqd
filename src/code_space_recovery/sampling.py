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
import tempfile

import numpy as np

try:  # package import
    from ._version import ALGORITHM_VERSION, PACKAGE_VERSION
except ImportError:  # pragma: no cover - flat-module compatibility
    from _version import ALGORITHM_VERSION, PACKAGE_VERSION  # type: ignore

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
# Conventional module version follows the distribution; MODULE_VERSION remains
# the legacy output-schema identifier.
__version__ = PACKAGE_VERSION

POSTPROCESSING_BRANCHES = (
    "m3_off_reset_off",
    "m3_on_reset_off",
)
_BRANCH_SAVE_INCOMPLETE_MARKER = ".branch_save_incomplete"
_CHECKPOINT_INCOMPLETE_MARKER = ".sampling_checkpoint_incomplete"


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


def _preflight_fresh_run_dir(run_dir: str | Path) -> Path:
    """Require a new or empty checkpoint directory before the first write."""
    path = Path(run_dir)
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"run_dir exists but is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"run_dir {path} is not empty; use a fresh directory for a "
                "sampling checkpoint."
            )
    return path


def _preflight_sampling_submission_run_dir(
    run_dir: str | Path,
    m3_file: str | Path | None,
) -> Path:
    """Allow only an existing in-place M3 file in a submission directory."""
    path = Path(run_dir)
    if path.is_symlink():
        raise ValueError(f"run_dir must not be a symbolic link: {path}")
    if not path.exists():
        return path
    if not path.is_dir():
        raise NotADirectoryError(f"run_dir exists but is not a directory: {path}")

    entries = list(path.iterdir())
    if not entries:
        return path
    if m3_file is None:
        raise FileExistsError(
            f"run_dir {path} is not empty; use a fresh directory for a "
            "sampling checkpoint."
        )

    src = Path(str(m3_file))
    if ".." in src.parts:
        raise ValueError(
            "An existing run_dir may contain only its exact M3 calibration "
            "file; the M3 path must not escape run_dir."
        )
    if src.is_symlink():
        raise ValueError(
            "An existing run_dir M3 calibration file must not be a symbolic "
            f"link: {src}"
        )

    run_dir_resolved = path.resolve(strict=True)
    src_resolved = src.resolve(strict=True)
    if src.parent.resolve(strict=True) != run_dir_resolved:
        raise ValueError(
            "An existing run_dir may contain only its exact M3 calibration "
            f"file; M3 path is outside run_dir: {src}"
        )
    if len(entries) != 1:
        raise FileExistsError(
            f"run_dir {path} contains entries other than its M3 calibration "
            "file; use a fresh directory for a sampling checkpoint."
        )

    existing = entries[0]
    if existing.is_symlink():
        raise ValueError(
            "An existing run_dir M3 calibration file must not be a symbolic "
            f"link: {existing}"
        )
    if not existing.is_file():
        raise FileExistsError(
            f"run_dir {path} contains a non-file entry; use a fresh directory "
            "for a sampling checkpoint."
        )
    if existing.name != src.name or existing.resolve(strict=True) != src_resolved:
        raise FileExistsError(
            f"run_dir {path} contains an entry other than its exact M3 "
            "calibration file; use a fresh directory for a sampling checkpoint."
        )
    return path


def _begin_checkpoint_write(run_dir: Path) -> Path:
    """Create the checkpoint completion marker before any artifact write."""
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / _CHECKPOINT_INCOMPLETE_MARKER
    with marker.open("x", encoding="utf-8") as marker_file:
        marker_file.write(
            "Sampling checkpoint is incomplete until saved_paths is committed.\n"
        )
    return marker


def _require_complete_checkpoint(run_dir: str | Path) -> Path:
    """Reject a checkpoint whose writer did not reach its final commit."""
    path = Path(run_dir)
    marker = path / _CHECKPOINT_INCOMPLETE_MARKER
    if marker.exists():
        raise ValueError(
            f"Sampling checkpoint is incomplete and cannot be loaded: {marker}"
        )
    return path


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


def _validate_backend_matches_run_info(
    run_info: Mapping[str, Any],
    backend: Any | None,
) -> None:
    """Reject a supplied backend that conflicts with saved run provenance."""
    if backend is None:
        return
    saved_name = run_info.get("backend_name")
    supplied_name = _backend_name(backend)
    if (
        saved_name is not None
        and supplied_name is not None
        and str(saved_name) != supplied_name
    ):
        raise ValueError(
            "backend does not match the saved sampling run: "
            f"saved={saved_name!r}, supplied={supplied_name!r}."
        )


def _validate_live_mitigator_matches_run_info(
    run_info: Mapping[str, Any],
    mitigator: Any | None,
) -> None:
    """Reject a live mitigator whose available backend identity conflicts."""
    if mitigator is None or run_info.get("backend_name") is None:
        return
    system_info = getattr(mitigator, "system_info", None)
    if not isinstance(system_info, Mapping):
        return
    mitigator_name = system_info.get("name")
    if (
        mitigator_name is not None
        and str(mitigator_name) != str(run_info["backend_name"])
    ):
        raise ValueError(
            "live M3 mitigator backend does not match the saved sampling run: "
            f"mitigator={mitigator_name!r}, saved={run_info['backend_name']!r}."
        )


def _require_positive_int(value: Any, name: str) -> int:
    """Return a strictly positive integer without accepting bool or coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer; got {value!r}.")
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be greater than zero; got {out}.")
    return out


def _mapping_entries(
    mapping: Any,
    *,
    name: str,
) -> tuple[list[Any], dict[int, Any] | None]:
    """Return raw mapping values without silently coercing physical qubits."""
    if isinstance(mapping, Mapping):
        by_cbit: dict[int, Any] = {}
        for raw_key, raw_value in mapping.items():
            if isinstance(raw_key, (bool, np.bool_)):
                raise TypeError(f"{name} classical-bit keys must not be boolean.")
            if isinstance(raw_key, (int, np.integer)):
                cbit = int(raw_key)
            elif isinstance(raw_key, str) and raw_key.isdecimal():
                cbit = int(raw_key)
            else:
                raise TypeError(
                    f"{name} classical-bit keys must be nonnegative integers or "
                    "decimal integer strings."
                )
            if cbit < 0:
                raise ValueError(f"{name} contains a negative classical-bit index.")
            if cbit in by_cbit:
                raise ValueError(
                    f"{name} contains duplicate representations of classical bit {cbit}."
                )
            by_cbit[cbit] = raw_value
        return [by_cbit[key] for key in sorted(by_cbit)], by_cbit

    if isinstance(mapping, (str, bytes)):
        raise TypeError(f"{name} must be a mapping or a sequence, not text.")
    try:
        return list(mapping), None
    except TypeError as exc:
        raise TypeError(f"{name} must be a mapping or an iterable sequence.") from exc


def _validate_measurement_mapping(
    mapping: Any,
    *,
    name: str,
    cbit_indices: Sequence[int] | None = None,
    expected_width: int | None = None,
    num_qubits: int | None = None,
    allow_selected_duplicates: bool = False,
) -> list[int]:
    """Validate a measurement mapping and return final-measurement values.

    All physical-qubit entries are checked before selecting a named ``meas``
    register.  Uniqueness is required for the selected final measurements, but
    legacy full mappings may repeat a qubit across different registers.
    """
    raw_values, by_cbit = _mapping_entries(mapping, name=name)
    if len(raw_values) == 0:
        raise ValueError(f"{name} is empty.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_values
    ):
        raise TypeError(f"{name} must contain only integers (not bool).")

    all_values = [int(value) for value in raw_values]
    if any(value < 0 for value in all_values):
        raise ValueError(f"{name} contains a negative physical qubit.")
    if num_qubits is not None and any(value >= num_qubits for value in all_values):
        raise ValueError(
            f"{name} contains a physical qubit outside the ISA circuit width "
            f"{num_qubits}."
        )

    if cbit_indices is None:
        selected = all_values
    else:
        indices = [int(index) for index in cbit_indices]
        try:
            # Keep selection semantics in one place.  This helper understands
            # both list-style and dict-style M3 mappings.
            selected = _mapping_values_for_cbits(mapping, indices)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} does not cover every classical bit in the named meas register."
            ) from exc

    if expected_width is not None and len(selected) != expected_width:
        raise ValueError(
            f"{name} has final-measurement width {len(selected)}, but expected "
            f"{expected_width}."
        )
    if not allow_selected_duplicates and len(set(selected)) != len(selected):
        raise ValueError(
            f"{name} contains duplicate physical qubits in the final measurement."
        )
    return selected


def _validate_used_qubits_for_m3(
    raw_used: Any,
    *,
    expected: Sequence[int],
    name: str = "compiled_run_info['used_qubits_for_m3']",
) -> list[int]:
    if isinstance(raw_used, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not text.")
    try:
        raw_values = list(raw_used)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable sequence.") from exc
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_values
    ):
        raise TypeError(f"{name} must contain only integers (not bool).")
    values = [int(value) for value in raw_values]
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must contain only nonnegative qubit indices.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate qubit indices.")
    listed = sorted(values)
    expected_sorted = sorted(int(value) for value in expected)
    if listed != expected_sorted:
        raise ValueError(
            f"{name} conflicts with the compiled measurement mappings: "
            f"listed={listed}, expected={expected_sorted}."
        )
    return listed


def _instruction_parts(instruction: Any) -> tuple[Any, Sequence[Any], Sequence[Any]]:
    """Return operation, qubits, and clbits across supported Qiskit layouts."""
    operation = getattr(instruction, "operation", None)
    qubits = getattr(instruction, "qubits", None)
    clbits = getattr(instruction, "clbits", None)
    if operation is None:
        try:
            operation, qubits, clbits = instruction
        except (TypeError, ValueError):
            return None, (), ()
    return operation, tuple(qubits or ()), tuple(clbits or ())


def _measurement_mapping_from_circuit(
    circ: Any,
    cbit_indices: Sequence[int],
) -> list[int] | None:
    """Infer measured physical qubits when circuit instructions are inspectable."""
    data = getattr(circ, "data", None)
    if data is None:
        return None
    try:
        instructions = list(data)
    except TypeError:
        return None

    measured: dict[int, int] = {}
    saw_measurement = False
    for instruction in instructions:
        operation, qubits, clbits = _instruction_parts(instruction)
        if getattr(operation, "name", None) != "measure":
            continue
        saw_measurement = True
        if len(qubits) != len(clbits):
            raise ValueError("ISA circuit contains a malformed measurement instruction.")
        for qubit, clbit in zip(qubits, clbits, strict=True):
            try:
                cbit = int(circ.find_bit(clbit).index)
                physical_qubit = int(circ.find_bit(qubit).index)
            except Exception as exc:
                raise ValueError(
                    "ISA circuit measurement bits could not be resolved."
                ) from exc
            previous = measured.get(cbit)
            if previous is not None and previous != physical_qubit:
                raise ValueError(
                    f"ISA circuit classical bit {cbit} is measured from multiple qubits."
                )
            measured[cbit] = physical_qubit

    if not saw_measurement:
        return None
    missing = [int(cbit) for cbit in cbit_indices if int(cbit) not in measured]
    if missing:
        raise ValueError(
            "ISA circuit is missing final measurement instruction(s) for classical "
            f"bit(s) {missing}."
        )
    return [measured[int(cbit)] for cbit in cbit_indices]


def _validate_record_circuit_metadata(
    record: Mapping[str, Any],
    circ: Any,
    index: int,
) -> None:
    """Reject a record/QPY circuit pairing whose available identities conflict."""
    metadata = getattr(circ, "metadata", None)
    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise TypeError(f"compiled ISA circuit {index} metadata must be a mapping.")

    if "circuit_index" in metadata and metadata["circuit_index"] is not None:
        raw_index = metadata["circuit_index"]
        if isinstance(raw_index, (bool, np.bool_)) or not isinstance(
            raw_index, (int, np.integer)
        ):
            raise TypeError(
                f"compiled ISA circuit {index} metadata circuit_index must be an integer."
            )
        if int(raw_index) != index:
            raise ValueError(
                f"compiled ISA circuit {index} metadata identifies circuit "
                f"{int(raw_index)}."
            )

    for field in ("record_id", "k", "r"):
        if (
            field in record
            and record[field] is not None
            and field in metadata
            and metadata[field] is not None
            and record[field] != metadata[field]
        ):
            raise ValueError(
                f"compiled record/circuit metadata mismatch at circuit {index} for "
                f"{field}: record={record[field]!r}, circuit={metadata[field]!r}."
            )


def _validate_compiled_sampling_run_info(
    compiled_run_info: Mapping[str, Any],
) -> tuple[list[Any], list[Any], list[Any], list[int]]:
    """Validate a compile checkpoint before calibration or job submission."""
    if not isinstance(compiled_run_info, Mapping):
        raise TypeError("compiled_run_info must be a mapping.")

    required = ("records", "isa_circuits", "mappings")
    missing = [key for key in required if key not in compiled_run_info]
    if missing:
        raise ValueError(
            "compiled_run_info is missing required field(s): " + ", ".join(missing)
        )

    sequences: dict[str, list[Any]] = {}
    for key in required:
        value = compiled_run_info[key]
        if isinstance(value, (str, bytes)):
            raise TypeError(f"compiled_run_info[{key!r}] must be a sequence, not text.")
        try:
            sequences[key] = list(value)
        except TypeError as exc:
            raise TypeError(
                f"compiled_run_info[{key!r}] must be an iterable sequence."
            ) from exc

    records = sequences["records"]
    isa_circuits = sequences["isa_circuits"]
    mappings = sequences["mappings"]
    n_circuits = len(isa_circuits)
    if n_circuits == 0:
        raise ValueError("compiled_run_info contains no ISA circuits to sample.")
    if len(records) != n_circuits or len(mappings) != n_circuits:
        raise ValueError(
            "compiled_run_info structure is inconsistent: "
            f"records={len(records)}, isa_circuits={n_circuits}, "
            f"mappings={len(mappings)}."
        )
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError("compiled_run_info records must all be mappings.")

    validated_mappings: list[list[int]] = []
    for index, (record, circ, mapping) in enumerate(
        zip(records, isa_circuits, mappings, strict=True)
    ):
        if circ is None:
            raise ValueError(f"compiled_run_info ISA circuit {index} is None.")
        _validate_record_circuit_metadata(record, circ, index)
        meas_indices = _classical_register_indices(circ, "meas")
        if meas_indices:
            expected_width = len(meas_indices)
        else:
            try:
                raw_expected_width = circ.num_clbits
            except AttributeError as exc:
                raise ValueError(
                    f"compiled_run_info ISA circuit {index} has no valid classical width."
                ) from exc
            if isinstance(raw_expected_width, (bool, np.bool_)) or not isinstance(
                raw_expected_width, (int, np.integer)
            ):
                raise TypeError(
                    f"compiled_run_info ISA circuit {index} classical width must be "
                    "an integer."
                )
            expected_width = int(raw_expected_width)
        if expected_width <= 0:
            raise ValueError(
                f"compiled_run_info ISA circuit {index} has no measured classical bits."
            )

        try:
            raw_num_qubits = circ.num_qubits
        except AttributeError as exc:
            raise ValueError(
                f"compiled_run_info ISA circuit {index} has no valid qubit width."
            ) from exc
        if isinstance(raw_num_qubits, (bool, np.bool_)) or not isinstance(
            raw_num_qubits, (int, np.integer)
        ):
            raise TypeError(
                f"compiled_run_info ISA circuit {index} qubit width must be an integer."
            )
        num_qubits = int(raw_num_qubits)
        if num_qubits <= 0:
            raise ValueError(
                f"compiled_run_info ISA circuit {index} has invalid qubit width "
                f"{num_qubits}."
            )
        raw_mapping_values, mapping_by_cbit = _mapping_entries(
            mapping,
            name=f"compiled_run_info mapping {index}",
        )
        mapping_for_validation = (
            mapping if mapping_by_cbit is not None else raw_mapping_values
        )
        # Current/upgraded checkpoints already store only the final register,
        # even when a legacy ISA circuit still has globally indexed probe
        # clbits. Slice by global `meas` indices only for an actual full mapping.
        mapping_cbit_indices = (
            meas_indices
            if meas_indices and len(raw_mapping_values) != expected_width
            else None
        )
        values = _validate_measurement_mapping(
            mapping_for_validation,
            name=f"compiled_run_info mapping {index}",
            cbit_indices=mapping_cbit_indices,
            expected_width=expected_width,
            num_qubits=num_qubits,
        )
        target_cbits = meas_indices if meas_indices else list(range(expected_width))
        circuit_mapping = _measurement_mapping_from_circuit(circ, target_cbits)
        if circuit_mapping is not None and values != circuit_mapping:
            raise ValueError(
                f"compiled_run_info mapping {index} conflicts with the ISA circuit's "
                "final measurement order: "
                f"saved={values}, circuit={circuit_mapping}."
            )
        validated_mappings.append(values)

    used_qubits = sorted({q for mapping in validated_mappings for q in mapping})
    if "used_qubits_for_m3" in compiled_run_info:
        _validate_used_qubits_for_m3(
            compiled_run_info["used_qubits_for_m3"],
            expected=used_qubits,
        )

    return records, isa_circuits, validated_mappings, used_qubits


def _normalized_compiled_sampling_run_info(
    compiled_run_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a validated snapshot whose required iterables are reusable lists."""
    records, isa_circuits, mappings, used_qubits = (
        _validate_compiled_sampling_run_info(compiled_run_info)
    )
    normalized = dict(compiled_run_info)
    normalized.update(
        {
            "records": records,
            "isa_circuits": isa_circuits,
            "mappings": mappings,
            "used_qubits_for_m3": used_qubits,
        }
    )
    return normalized


# =============================================================================
# Circuit and submission helpers
# =============================================================================


def make_logical_ghz_initial_state(n_logical: int):
    """Create an ``n_logical``-qubit GHZ state-preparation circuit."""
    n_logical = _require_positive_int(n_logical, "n_logical")
    _require_qiskit()
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


def _m3_qubits_for_counts(
    mapping: Any,
    n_bits: int,
    *,
    allow_legacy_full_mapping: bool = False,
) -> list[int]:
    """Return a qubit list whose length matches an n-bit counts dictionary.

    Current mappings contain only final measurements, so their width must match
    the result width exactly. Legacy saved mappings may contain additional
    leading classical registers; callers must opt in explicitly before the
    final `meas` tail is selected.
    """
    n_bits = int(n_bits)
    values = _mapping_values(mapping)

    if len(values) == n_bits:
        selected = values

    elif len(values) > n_bits and allow_legacy_full_mapping:
        # Legacy artifacts stored the full mapping while counts contained only
        # the final `meas` register.
        selected = values[-n_bits:]

    elif len(values) > n_bits:
        raise ValueError(
            "M3 mapping length is larger than the current counts bitstring length. "
            "Automatic tail selection is allowed only for an explicitly marked "
            "legacy reset-probe artifact. "
            f"mapping length={len(values)}, counts bits={n_bits}."
        )

    else:
        raise ValueError(
            "M3 mapping length is shorter than the counts bitstring length. "
            f"mapping length={len(values)}, counts bits={n_bits}."
        )

    if any(q < 0 for q in selected) or len(set(selected)) != len(selected):
        raise ValueError(
            "Final-measurement M3 mapping must contain unique nonnegative qubits."
        )
    return selected


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
        return {
            "package_version": PACKAGE_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "summary": summary,
            "per_circuit": per_circuit,
        }

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
    return {
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "summary": summary,
        "per_circuit": per_circuit,
    }


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
    if isinstance(optimization_level, (bool, np.bool_)) or not isinstance(
        optimization_level,
        (int, np.integer),
    ):
        raise TypeError("optimization_level must be an integer in [0, 3].")
    optimization_level = int(optimization_level)
    if optimization_level < 0 or optimization_level > 3:
        raise ValueError("optimization_level must be in [0, 3].")
    if not isinstance(print_summary, bool):
        raise TypeError(f"print_summary must be bool, got {type(print_summary).__name__}.")
    _require_runtime()
    _require_mthree()

    records = list(sqdrift_circuits["circuit_records"])

    circuits = []
    for record in records:
        qc = add_final_measurement(record["circuit"])
        qc.metadata = dict(record["circuit"].metadata or {})
        qc.metadata.update(
            {
                "record_id": record["record_id"],
                "k": int(record["k"]),
                "r": int(record["r"]),
                "evolution_time": float(record["evolution_time"]),
                "package_version": PACKAGE_VERSION,
                "algorithm_version": ALGORITHM_VERSION,
            }
        )
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
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
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
    shots = _require_positive_int(shots, "shots")
    chunk_size = _require_positive_int(chunk_size, "chunk_size")
    max_circuits_per_batch = _require_positive_int(
        max_circuits_per_batch, "max_circuits_per_batch"
    )
    if not isinstance(dd_enabled, bool):
        raise TypeError(f"dd_enabled must be bool, got {type(dd_enabled).__name__}.")
    compiled_run_info = _normalized_compiled_sampling_run_info(compiled_run_info)
    isa_circuits = compiled_run_info["isa_circuits"]
    mappings = compiled_run_info["mappings"]
    used_qubits = compiled_run_info["used_qubits_for_m3"]
    actual_backend_name = _backend_name(backend)
    compiled_backend_name = compiled_run_info.get("backend_name")
    if (
        compiled_backend_name is not None
        and actual_backend_name is not None
        and str(compiled_backend_name) != actual_backend_name
    ):
        raise ValueError(
            "backend does not match the compiled checkpoint: "
            f"compiled={compiled_backend_name!r}, supplied={actual_backend_name!r}."
        )

    # Optional dependencies and all external calibration/submission work are
    # intentionally reached only after the local configuration is validated.
    _require_runtime()
    _require_mthree()

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
                job = sampler.run(chunk, shots=shots)
                jobs.append(job)
                job_slices.append((int(start), int(stop)))
                print("submitted sampling job:", job.job_id(), "| circuits:", f"[{start}, {stop})")

    run_info = dict(compiled_run_info)
    run_info.update(
        {
            # Persist the validated final-measurement-only mappings.  This also
            # upgrades legacy full mappings before later M3 post-processing.
            "mappings": mappings,
            "used_qubits_for_m3": used_qubits,
            "mitigator": mit,
            "m3_file": m3_file,
            "m3_jobs": m3_jobs,
            "m3_method": str(m3_method),
            "shots": shots,
            "sampling_jobs": jobs,
            "sampling_job_ids": _job_ids(jobs),
            "job_slices": job_slices,
            "chunk_size": chunk_size,
            "max_circuits_per_batch": max_circuits_per_batch,
            "batch_max_time": batch_max_time,
            "backend_name": actual_backend_name,
            "dd_enabled": bool(dd_enabled),
            "dd_sequence_type": str(dd_sequence_type) if dd_enabled else None,
            "created_at_utc": _utc_now_iso(),
            "sampling_submitted": True,
            "package_version": PACKAGE_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
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
    shots = _require_positive_int(shots, "shots")
    chunk_size = _require_positive_int(chunk_size, "chunk_size")
    max_circuits_per_batch = _require_positive_int(
        max_circuits_per_batch, "max_circuits_per_batch"
    )
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
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
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
    if len(jobs_eff) != len(job_slices):
        raise ValueError(
            "jobs/job_slices length mismatch while building the record table: "
            f"jobs={len(jobs_eff)}, job_slices={len(job_slices)}."
        )

    job_lookup: dict[int, dict[str, Any]] = {}
    for job_pos, (job, slc) in enumerate(zip(jobs_eff, job_slices, strict=True)):
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
    normalized_run_info = dict(run_info)
    if "isa_circuits" in normalized_run_info:
        normalized_run_info = _normalized_compiled_sampling_run_info(
            normalized_run_info
        )
    (
        validated_jobs,
        validated_records,
        validated_mappings,
        validated_slices,
        _,
    ) = _validate_postprocessing_inputs(
        jobs,
        normalized_run_info,
    )
    normalized_run_info.update(
        {
            "records": validated_records,
            "mappings": validated_mappings,
            "job_slices": validated_slices,
        }
    )
    run_info = normalized_run_info
    jobs = validated_jobs
    _validate_backend_matches_run_info(run_info, backend)
    _validate_live_mitigator_matches_run_info(run_info, run_info.get("mitigator"))
    for flag_name, flag_value in (
        ("save_qpy", save_qpy),
        ("save_hardware_snapshot", save_hardware_snapshot),
        ("fail_on_qpy_error", fail_on_qpy_error),
    ):
        if not isinstance(flag_value, bool):
            raise TypeError(f"{flag_name} must be bool.")
    if extra_metadata is not None and not isinstance(extra_metadata, Mapping):
        raise TypeError("extra_metadata must be a mapping or None.")
    if sqdrift_circuits is not None and not isinstance(sqdrift_circuits, Mapping):
        raise TypeError("sqdrift_circuits must be a mapping or None.")

    m3_file = run_info.get("m3_file")
    reserved_run_names = {
        _CHECKPOINT_INCOMPLETE_MARKER,
        "manifest.json",
        "record_table.json",
        "compiled_info.json",
        "run_info_serializable.json",
        "hardware_snapshot.json",
        "qpy_save_errors.json",
        "saved_paths.json",
        "circuits",
    }
    reserved_run_names_casefold = {name.casefold() for name in reserved_run_names}
    if m3_file is not None:
        src = Path(str(m3_file))
        if src.name.casefold() in reserved_run_names_casefold:
            raise ValueError(
                "M3 calibration filename collides with a reserved sampling "
                f"artifact name: {src.name!r}."
            )
        if not src.exists():
            raise FileNotFoundError(f"M3 calibration file not found: {src}")
        if not src.is_file():
            raise ValueError(f"M3 calibration path is not a regular file: {src}")
        legacy_full_mapping = (
            run_info.get("reset_probe") is True and "isa_circuits" not in run_info
        )
        _validate_m3_file_provenance(
            src,
            backend_name=(
                str(run_info["backend_name"])
                if run_info.get("backend_name") is not None
                else _backend_name(backend)
            ),
            expected_qubits=(
                None
                if legacy_full_mapping
                else sorted(
                    {
                        qubit
                        for mapping in validated_mappings
                        for qubit in mapping
                    }
                )
            ),
        )

    run_dir = _preflight_sampling_submission_run_dir(run_dir, m3_file)
    record_table = build_record_table(run_info, jobs)
    checkpoint_marker = _begin_checkpoint_write(run_dir)
    circuits_dir = run_dir / "circuits"
    saved: dict[str, str] = {}

    sampling_job_ids = _job_ids(jobs)
    m3_job_ids = _job_ids(run_info.get("m3_jobs"))

    manifest = {
        "module": "sampling",
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
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

    # Copy the preflighted M3 calibration file into the checkpoint bundle.
    if m3_file is not None:
        src = Path(str(m3_file))
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
                    qc.metadata = dict(record["circuit"].metadata or {})
                    qc.metadata.update(
                        {
                            "record_id": record.get("record_id"),
                            "k": record.get("k"),
                            "r": record.get("r"),
                            "evolution_time": record.get("evolution_time"),
                            "package_version": PACKAGE_VERSION,
                            "algorithm_version": ALGORITHM_VERSION,
                        }
                    )
                    sampling_circuits.append(qc)
        if sampling_circuits is not None:
            _attempt_qpy("sampling_circuits_qpy", sampling_circuits, "sampling_circuits_final_meas_only.qpy")

        if "isa_circuits" in run_info:
            _attempt_qpy("isa_circuits_qpy", run_info["isa_circuits"], "isa_circuits_transpiled.qpy")

        if qpy_errors:
            saved["qpy_save_errors"] = str(_write_json(run_dir / "qpy_save_errors.json", qpy_errors))

    saved["saved_paths"] = str(_write_json(run_dir / "saved_paths.json", saved))
    checkpoint_marker.unlink()
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
    compiled_run_info = _normalized_compiled_sampling_run_info(compiled_run_info)
    _validate_backend_matches_run_info(compiled_run_info, backend)
    for flag_name, flag_value in (
        ("save_qpy", save_qpy),
        ("save_hardware_snapshot", save_hardware_snapshot),
        ("fail_on_qpy_error", fail_on_qpy_error),
    ):
        if not isinstance(flag_value, bool):
            raise TypeError(f"{flag_name} must be bool.")
    if extra_metadata is not None and not isinstance(extra_metadata, Mapping):
        raise TypeError("extra_metadata must be a mapping or None.")
    if sqdrift_circuits is not None and not isinstance(sqdrift_circuits, Mapping):
        raise TypeError("sqdrift_circuits must be a mapping or None.")

    run_dir = _preflight_fresh_run_dir(run_dir)
    record_table = build_record_table(compiled_run_info, jobs=[])
    checkpoint_marker = _begin_checkpoint_write(run_dir)
    circuits_dir = run_dir / "circuits"
    saved: dict[str, str] = {}

    manifest = {
        "module": "sampling",
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
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
    checkpoint_marker.unlink()
    return saved


def load_compiled_run_info(run_dir: str | Path) -> dict[str, Any]:
    """Load a compile checkpoint that contains a saved ISA-circuit QPY file."""
    run_dir = _require_complete_checkpoint(run_dir)
    info_path = run_dir / "run_info_compiled_serializable.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"No compile-only run info found at {info_path}. "
            "Call save_compiled_sampling_run(...) after compile_sampling_circuits(...)."
        )
    info = _read_json(info_path)
    info["records"] = _read_json(run_dir / "record_table.json")

    # Checkpoints are relocatable bundles. Never follow a path recorded by the
    # machine that created the checkpoint: it may point outside a copied run or
    # silently reconnect the copy to stale source artifacts.
    isa_path = run_dir / "circuits" / "isa_circuits_transpiled.qpy"
    if isa_path.exists():
        info["isa_circuits"] = load_qpy_circuits(isa_path)
    else:
        raise FileNotFoundError(f"Compiled ISA QPY file not found: {isa_path}")

    sampling_path = run_dir / "circuits" / "sampling_circuits_final_meas_only.qpy"
    if sampling_path.exists():
        info["sampling_circuits"] = load_qpy_circuits(sampling_path)

    return info


def load_sampling_jobs_from_manifest(service: Any, run_dir_or_manifest: str | Path) -> tuple[list[Any], dict[str, Any]]:
    """Load sampling jobs from `manifest.json` using `QiskitRuntimeService.job`."""
    p = Path(run_dir_or_manifest)
    run_dir = p if p.is_dir() else p.parent
    _require_complete_checkpoint(run_dir)
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
    run_dir = _require_complete_checkpoint(run_dir)
    info = _read_json(run_dir / "run_info_serializable.json")
    record_table = _read_json(run_dir / "record_table.json")
    info["records"] = record_table
    saved_paths_file = run_dir / "saved_paths.json"
    saved_paths = _read_json(saved_paths_file) if saved_paths_file.exists() else {}
    stored_m3 = saved_paths.get("m3_cals")
    candidates: list[Path] = []
    if stored_m3 is not None:
        # Only the basename is provenance; the original absolute parent is not.
        candidates.append(run_dir / Path(str(stored_m3)).name)
    if info.get("m3_file") is not None:
        candidates.append(run_dir / Path(str(info["m3_file"])).name)
    for candidate in candidates:
        if candidate.is_file():
            info["m3_file"] = str(candidate)
            break
    else:
        # Never leave an original-machine absolute path active after the bundle
        # has been moved or copied.
        info["m3_file"] = None

    isa_path = run_dir / "circuits" / "isa_circuits_transpiled.qpy"
    if isa_path.is_file():
        info["isa_circuits"] = load_qpy_circuits(isa_path)
        _validate_compiled_sampling_run_info(info)
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
    n_bits = _require_positive_int(n_bits, "n_bits")
    upper_bound = 1 << n_bits

    if isinstance(key, (bool, np.bool_)):
        raise TypeError("bitstring keys must be integers or binary/hex strings, not bool.")
    if isinstance(key, (int, np.integer)):
        value = int(key)
        if value < 0 or value >= upper_bound:
            raise ValueError(
                f"Integer bitstring key {key!r} is outside [0, {upper_bound})."
            )
        return format(value, f"0{n_bits}b")

    if not isinstance(key, str):
        raise TypeError(
            "bitstring keys must be integers or binary/hex strings; "
            f"got {type(key).__name__}."
        )

    s = key.replace(" ", "")
    if s.lower().startswith("0x"):
        digits = s[2:]
        max_digits = (n_bits + 3) // 4
        if not digits or len(digits) > max_digits or any(
            ch not in "0123456789abcdefABCDEF" for ch in digits
        ):
            raise ValueError(
                f"Invalid {n_bits}-bit hexadecimal key {key!r}: expected between "
                f"1 and {max_digits} hexadecimal digit(s) after 0x."
            )
        value = int(digits, 16)
        if value >= upper_bound:
            raise ValueError(
                f"Hexadecimal bitstring key {key!r} is outside [0, {upper_bound})."
            )
        return format(value, f"0{n_bits}b")

    if s.lower().startswith("0b"):
        digits = s[2:]
        if not digits or len(digits) > n_bits or any(ch not in "01" for ch in digits):
            raise ValueError(
                f"Invalid {n_bits}-bit binary key {key!r}: expected between 1 and "
                f"{n_bits} binary digit(s) after 0b."
            )
        return digits.zfill(n_bits)

    if not s or len(s) > n_bits or any(ch not in "01" for ch in s):
        raise ValueError(
            f"Invalid {n_bits}-bit binary key {key!r}: expected between 1 and "
            f"{n_bits} bits."
        )
    return s.zfill(n_bits)


def _bitstrings_to_uint8(bitstrings: Sequence[str], n_bits: int) -> np.ndarray:
    n_bits = _require_positive_int(n_bits, "n_bits")
    if len(bitstrings) == 0:
        return np.empty((0, n_bits), dtype=np.uint8)
    return np.asarray([[1 if ch == "1" else 0 for ch in s] for s in bitstrings], dtype=np.uint8)


def _finite_real_value(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real numeric value; got {value!r}.")
    try:
        out = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be represented as a finite float.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return out


def _counts_to_arrays(counts: Mapping[Any, Any], n_bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_bits = _require_positive_int(n_bits, "n_bits")
    if not isinstance(counts, Mapping):
        raise TypeError("physical counts must be a mapping.")

    merged: dict[str, int] = {}
    for raw_key, raw_count in counts.items():
        key = _normalise_bit_key(raw_key, n_bits)
        value_name = f"physical count for key {raw_key!r}"
        if isinstance(raw_count, (bool, np.bool_)):
            raise TypeError(f"{value_name} must be an integer count, not bool.")
        if isinstance(raw_count, (int, np.integer)):
            # Preserve arbitrary-size Python integers and NumPy integral values
            # exactly through validation and duplicate-key merging.  Converting
            # first would silently round integers above 2**53.
            count = int(raw_count)
            if count < 0:
                raise ValueError(
                    f"{value_name} must be nonnegative; got {raw_count!r}."
                )
        else:
            value = _finite_real_value(raw_count, value_name)
            if value < 0.0:
                raise ValueError(
                    f"{value_name} must be nonnegative; got {value}."
                )
            if not value.is_integer():
                raise ValueError(
                    f"{value_name} must have an integer value; got {value}."
                )
            count = int(value)
        if count != 0:
            merged[key] = merged.get(key, 0) + count

    if not merged:
        raise ValueError("physical counts must have a positive total count.")

    keys = list(merged)
    try:
        raw_values = np.asarray([merged[key] for key in keys], dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "physical counts cannot be represented as finite float64 values."
        ) from exc
    if not np.all(np.isfinite(raw_values)):
        raise ValueError(
            "physical counts cannot be represented as finite float64 values."
        )
    for key, stored in zip(keys, raw_values, strict=True):
        if int(stored) != merged[key]:
            raise ValueError(
                "physical count cannot be represented exactly by the float64 NPZ "
                f"schema for normalized key {key!r}; got {merged[key]}."
            )
    total = float(np.sum(raw_values))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("physical counts must have a finite positive total count.")
    weights = raw_values / total
    return _bitstrings_to_uint8(keys, n_bits), weights, raw_values


def _distribution_to_dict(
    dist: Any,
    n_bits: int,
    *,
    allow_negative: bool = True,
    value_name: str = "distribution weight",
) -> dict[str, float]:
    n_bits = _require_positive_int(n_bits, "n_bits")
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
        val = _finite_real_value(v, f"{value_name} for key {k!r}")
        if not allow_negative and val < 0.0:
            raise ValueError(
                f"{value_name} for key {k!r} must be nonnegative; got {val}."
            )
        key = _normalise_bit_key(k, n_bits)
        combined = out.get(key, 0.0) + val
        if not math.isfinite(combined):
            raise ValueError(
                f"combined {value_name} for normalized key {key!r} is non-finite."
            )
        if combined == 0.0:
            out.pop(key, None)
        else:
            out[key] = combined
    return out


def _project_quasi_to_probability(raw_quasi: Any, n_bits: int) -> tuple[dict[str, float], dict[str, Any]]:
    raw = _distribution_to_dict(
        raw_quasi,
        n_bits,
        allow_negative=True,
        value_name="M3 quasi-probability",
    )
    negative_mass = float(sum(-v for v in raw.values() if v < 0.0))
    raw_sum = float(sum(raw.values()))
    if not math.isfinite(negative_mass) or not math.isfinite(raw_sum):
        raise ValueError("aggregate M3 quasi-probability statistics must be finite.")

    if hasattr(raw_quasi, "nearest_probability_distribution") and callable(raw_quasi.nearest_probability_distribution):
        projected_obj = raw_quasi.nearest_probability_distribution()
        projected = _distribution_to_dict(
            projected_obj,
            n_bits,
            allow_negative=False,
            value_name="projected M3 probability",
        )
        method = "nearest_probability_distribution"
    else:
        projected = {k: max(0.0, v) for k, v in raw.items()}
        method = "clip_and_renormalize"

    total = float(sum(projected.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("projected M3 probabilities must have a finite positive total.")
    projected = {k: v / total for k, v in projected.items() if v > 0.0}

    all_keys = set(raw) | set(projected)
    l1_change = float(sum(abs(projected.get(k, 0.0) - raw.get(k, 0.0)) for k in all_keys))
    projected_sum = float(sum(projected.values()))
    if not math.isfinite(l1_change) or not math.isfinite(projected_sum):
        raise ValueError("aggregate projected M3 probability statistics must be finite.")
    meta = {
        "projection_method": method,
        "negative_mass": negative_mass,
        "clip_l1_change": l1_change,
        "raw_quasi_sum": raw_sum,
        "projected_sum": projected_sum,
    }
    return projected, meta


def _prob_dict_to_arrays(
    prob: Mapping[Any, Any],
    n_bits: int,
    *,
    allow_negative: bool = False,
    value_name: str = "probability",
) -> tuple[np.ndarray, np.ndarray]:
    validated = _distribution_to_dict(
        prob,
        n_bits,
        allow_negative=allow_negative,
        value_name=value_name,
    )
    items = list(validated.items())
    if len(items) == 0:
        return np.empty((0, n_bits), dtype=np.uint8), np.empty(0, dtype=np.float64)
    keys = [k for k, _ in items]
    vals = np.asarray([v for _, v in items], dtype=np.float64)
    return _bitstrings_to_uint8(keys, n_bits), vals


def _validate_m3_file_provenance(
    m3_file: str | Path,
    *,
    backend_name: str | None,
    expected_qubits: Sequence[int] | None,
) -> None:
    """Validate optional provenance fields exposed by an M3 JSON artifact."""
    path = Path(m3_file)
    if not path.is_file():
        raise FileNotFoundError(f"M3 calibration file not found: {path}")
    try:
        payload = _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"M3 calibration file is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        return

    sources = [payload]
    for key in ("metadata", "provenance", "system_info"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)

    file_backend: str | None = None
    for source in sources:
        for key in ("backend_name", "backend", "system_name"):
            value = source.get(key)
            if isinstance(value, str) and value:
                file_backend = value
                break
        if file_backend is not None:
            break
    if (
        backend_name is not None
        and file_backend is not None
        and file_backend != backend_name
    ):
        raise ValueError(
            "M3 calibration backend does not match the sampling run: "
            f"calibration={file_backend!r}, expected={backend_name!r}."
        )

    file_qubits: list[int] | None = None
    for source in sources:
        for key in ("qubits", "calibration_qubits", "qubit_list"):
            value = source.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                raw = list(value)
                if all(
                    not isinstance(item, (bool, np.bool_))
                    and isinstance(item, (int, np.integer))
                    and int(item) >= 0
                    for item in raw
                ):
                    file_qubits = [int(item) for item in raw]
                    break
        if file_qubits is not None:
            break
    if expected_qubits is not None and file_qubits is not None:
        missing = sorted(set(int(q) for q in expected_qubits) - set(file_qubits))
        if missing:
            raise ValueError(
                "M3 calibration file does not cover required measured qubits: "
                f"missing={missing}."
            )

    # Current mthree files store a backend-width ``cals`` list and use null
    # entries for qubits that were not calibrated. This is stronger evidence
    # than a separate optional qubit list when it is present.
    raw_cals = payload.get("cals")
    if (
        expected_qubits is not None
        and isinstance(raw_cals, Sequence)
        and not isinstance(raw_cals, (str, bytes))
    ):
        cals = list(raw_cals)
        uncovered = [
            int(qubit)
            for qubit in expected_qubits
            if int(qubit) >= len(cals) or cals[int(qubit)] in (None, [])
        ]
        if uncovered:
            raise ValueError(
                "M3 calibration file has no calibration matrix for required "
                f"measured qubits: missing={sorted(set(uncovered))}."
            )


def load_m3_mitigator(
    backend: Any,
    m3_file: str | Path,
    *,
    expected_backend_name: str | None = None,
    expected_qubits: Sequence[int] | None = None,
):
    """Reconstruct an M3 mitigator from a provenance-compatible calibration file."""
    _require_mthree()
    actual_backend_name = _backend_name(backend)
    if (
        expected_backend_name is not None
        and actual_backend_name is not None
        and str(expected_backend_name) != actual_backend_name
    ):
        raise ValueError(
            "backend does not match the saved sampling run: "
            f"saved={expected_backend_name!r}, supplied={actual_backend_name!r}."
        )
    _validate_m3_file_provenance(
        m3_file,
        backend_name=(
            str(expected_backend_name)
            if expected_backend_name is not None
            else actual_backend_name
        ),
        expected_qubits=expected_qubits,
    )
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
        raw_quasi_dict = _distribution_to_dict(
            raw_quasi,
            n_bits,
            allow_negative=True,
            value_name="M3 quasi-probability",
        )
        raw_quasi_bits, raw_quasi_weights = _prob_dict_to_arrays(
            raw_quasi_dict,
            n_bits,
            allow_negative=True,
            value_name="M3 quasi-probability",
        )
        projected, m3_meta = _project_quasi_to_probability(raw_quasi, n_bits)
        bitstrings, weights = _prob_dict_to_arrays(
            projected,
            n_bits,
            allow_negative=False,
            value_name="projected M3 probability",
        )
    else:
        bitstrings = count_bits
        weights = count_weights

    metadata = {
        "variant": variant,
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
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


def _validate_postprocessing_inputs(
    jobs: Sequence[Any],
    run_info: Mapping[str, Any],
) -> tuple[list[Any], list[Mapping[str, Any]], list[list[int]], list[tuple[int, int]], int | None]:
    """Preflight all post-processing structure before any job or M3 call."""
    if not isinstance(run_info, Mapping):
        raise TypeError("run_info must be a mapping.")
    required = ("records", "mappings", "job_slices")
    missing = [key for key in required if key not in run_info]
    if missing:
        raise ValueError("run_info is missing required field(s): " + ", ".join(missing))

    if isinstance(jobs, (str, bytes)):
        raise TypeError("jobs must be a sequence of job objects, not text.")
    try:
        validated_jobs = list(jobs)
    except TypeError as exc:
        raise TypeError("jobs must be an iterable sequence of job objects.") from exc

    actual_job_ids = [_job_id(job) for job in validated_jobs]
    if any(not job_id for job_id in actual_job_ids):
        raise ValueError("sampling job IDs must be nonempty strings.")
    if len(set(actual_job_ids)) != len(actual_job_ids):
        raise ValueError("jobs must not contain duplicate sampling job IDs.")
    if "sampling_jobs" in run_info:
        stored_jobs = run_info["sampling_jobs"]
        if isinstance(stored_jobs, (str, bytes)):
            raise TypeError("run_info['sampling_jobs'] must be a sequence, not text.")
        try:
            stored_job_ids = [_job_id(job) for job in stored_jobs]
        except TypeError as exc:
            raise TypeError(
                "run_info['sampling_jobs'] must be an iterable sequence."
            ) from exc
        if stored_job_ids != actual_job_ids:
            raise ValueError(
                "supplied sampling jobs do not match run_info['sampling_jobs'] "
                "IDs/order: "
                f"saved={stored_job_ids}, supplied={actual_job_ids}."
            )
    if "sampling_job_ids" in run_info:
        raw_saved_ids = run_info["sampling_job_ids"]
        if isinstance(raw_saved_ids, (str, bytes)):
            raise TypeError("run_info['sampling_job_ids'] must be a sequence, not text.")
        try:
            raw_saved_ids = list(raw_saved_ids)
        except TypeError as exc:
            raise TypeError(
                "run_info['sampling_job_ids'] must be an iterable sequence."
            ) from exc
        if any(not isinstance(value, str) for value in raw_saved_ids):
            raise TypeError("run_info['sampling_job_ids'] must contain only strings.")
        saved_job_ids = list(raw_saved_ids)
        if any(not value for value in saved_job_ids):
            raise ValueError("run_info['sampling_job_ids'] must not contain empty IDs.")
        if saved_job_ids != actual_job_ids:
            raise ValueError(
                "supplied sampling jobs do not match run_info['sampling_job_ids']: "
                f"saved={saved_job_ids}, supplied={actual_job_ids}."
            )

    if "isa_circuits" in run_info:
        records_raw, _, validated_mappings, used_qubits = (
            _validate_compiled_sampling_run_info(run_info)
        )
        records = list(records_raw)
        if len(validated_mappings) > 1:
            widths = {len(mapping) for mapping in validated_mappings}
            if len(widths) != 1:
                raise ValueError(
                    "run_info mappings must have one common final-measurement width."
                )
    else:
        try:
            records = list(run_info["records"])
            raw_mappings = list(run_info["mappings"])
        except TypeError as exc:
            raise TypeError("run_info records and mappings must be iterable sequences.") from exc
        if len(records) == 0:
            raise ValueError("run_info contains no records to post-process.")
        if any(not isinstance(record, Mapping) for record in records):
            raise TypeError("run_info records must all be mappings.")
        if len(raw_mappings) != len(records):
            raise ValueError(
                "run_info records/mappings length mismatch: "
                f"records={len(records)}, mappings={len(raw_mappings)}."
            )

        validated_mappings = []
        expected_width: int | None = None
        legacy_full_mapping = run_info.get("reset_probe") is True
        for index, mapping in enumerate(raw_mappings):
            values = _validate_measurement_mapping(
                mapping,
                name=f"run_info mapping {index}",
                expected_width=expected_width,
                # Saved legacy artifacts can contain probe+meas mappings with
                # repeated qubits. The final n_bits tail is validated when the
                # corresponding result is read.
                allow_selected_duplicates=legacy_full_mapping,
            )
            if expected_width is None:
                expected_width = len(values)
            validated_mappings.append(values)
        used_qubits = sorted(
            {value for mapping in validated_mappings for value in mapping}
        )
        if "used_qubits_for_m3" in run_info:
            _validate_used_qubits_for_m3(
                run_info["used_qubits_for_m3"],
                expected=used_qubits,
                name="run_info['used_qubits_for_m3']",
            )

    try:
        raw_slices = list(run_info["job_slices"])
    except TypeError as exc:
        raise TypeError("run_info['job_slices'] must be an iterable sequence.") from exc
    if len(validated_jobs) != len(raw_slices):
        raise ValueError(
            "jobs/job_slices length mismatch: "
            f"jobs={len(validated_jobs)}, job_slices={len(raw_slices)}."
        )
    if len(validated_jobs) == 0:
        raise ValueError("jobs and run_info['job_slices'] must not be empty.")

    validated_slices: list[tuple[int, int]] = []
    expected_start = 0
    for index, raw_slice in enumerate(raw_slices):
        if isinstance(raw_slice, (str, bytes)):
            raise TypeError(f"job_slices[{index}] must be a two-integer sequence.")
        try:
            endpoints = list(raw_slice)
        except TypeError as exc:
            raise TypeError(
                f"job_slices[{index}] must be a two-integer sequence."
            ) from exc
        if len(endpoints) != 2:
            raise ValueError(f"job_slices[{index}] must contain exactly two endpoints.")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in endpoints
        ):
            raise TypeError(
                f"job_slices[{index}] endpoints must be integers (not bool)."
            )
        start, stop = (int(endpoints[0]), int(endpoints[1]))
        if start != expected_start:
            raise ValueError(
                "job_slices must be contiguous, ordered, and start at zero: "
                f"slice {index} starts at {start}, expected {expected_start}."
            )
        if stop <= start:
            raise ValueError(
                f"job_slices[{index}] must satisfy stop > start; got ({start}, {stop})."
            )
        if stop > len(records):
            raise ValueError(
                f"job_slices[{index}] stop {stop} exceeds record count {len(records)}."
            )
        validated_slices.append((start, stop))
        expected_start = stop
    if expected_start != len(records):
        raise ValueError(
            "job_slices do not cover all records: "
            f"covered [0, {expected_start}), records={len(records)}."
        )

    shots = run_info.get("shots")
    if shots is not None:
        shots = _require_positive_int(shots, "run_info['shots']")
    return validated_jobs, records, validated_mappings, validated_slices, shots


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
    branches: dict[str, list[dict[str, Any]]] = {name: [] for name in POSTPROCESSING_BRANCHES}
    for variant, record in iter_postprocessing_branch_records_from_batch_jobs(
        jobs,
        run_info,
        backend=backend,
        m3_file=m3_file,
    ):
        branches[variant].append(record)

    return branches


def _validate_bitstrings_and_weights(
    bitstrings: Any,
    weights: Any,
    *,
    source: str,
    weight_name: str = "weights",
    allow_negative: bool = False,
    require_integer_weights: bool = False,
    expected_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate encoded artifact arrays before performing any dtype conversion."""
    bits = np.asarray(bitstrings)
    if bits.dtype != np.dtype(bool) and not (
        np.issubdtype(bits.dtype, np.integer)
        or np.issubdtype(bits.dtype, np.floating)
    ):
        raise TypeError(
            f"{source}: bitstrings must have a real numeric or bool dtype; got "
            f"{bits.dtype}."
        )
    if bits.ndim != 2:
        raise ValueError(f"{source}: bitstrings must be a 2D array; got shape {bits.shape}.")

    width = int(bits.shape[1])
    if width <= 0 or width % 2 != 0:
        raise ValueError(
            f"{source}: encoded bitstring width must be a positive even integer; "
            f"got {width}."
        )
    if expected_width is not None and width != expected_width:
        raise ValueError(
            f"{source}: encoded bitstring width {width} conflicts with the expected "
            f"width {expected_width}."
        )
    if np.issubdtype(bits.dtype, np.floating) and bits.size and not np.all(
        np.isfinite(bits)
    ):
        raise ValueError(f"{source}: bitstrings must contain only finite values.")
    if bits.size and np.any((bits != 0) & (bits != 1)):
        raise ValueError(f"{source}: bitstrings must contain only exact 0/1 values.")

    raw_weights = np.asarray(weights)
    if raw_weights.ndim != 1:
        raise ValueError(
            f"{source}: {weight_name} must be a 1D array; got shape {raw_weights.shape}."
        )
    if len(raw_weights) != len(bits):
        raise ValueError(
            f"{source}: {weight_name} length {len(raw_weights)} does not match "
            f"bitstring row count {len(bits)}."
        )
    if raw_weights.dtype == np.dtype(bool) or not np.issubdtype(
        raw_weights.dtype, np.number
    ):
        raise TypeError(f"{source}: {weight_name} must have a real numeric dtype.")
    if np.issubdtype(raw_weights.dtype, np.complexfloating):
        raise TypeError(f"{source}: {weight_name} must be real, not complex.")

    try:
        checked_weights = raw_weights.astype(np.float64, copy=False)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: {weight_name} cannot be represented as float64."
        ) from exc
    if not np.all(np.isfinite(checked_weights)):
        raise ValueError(f"{source}: {weight_name} must contain only finite values.")
    if not allow_negative and np.any(checked_weights < 0.0):
        raise ValueError(f"{source}: {weight_name} must be nonnegative.")
    if require_integer_weights and np.any(checked_weights != np.floor(checked_weights)):
        raise ValueError(f"{source}: {weight_name} must contain integer-valued counts.")

    return (
        np.ascontiguousarray(bits, dtype=np.uint8),
        np.ascontiguousarray(checked_weights, dtype=np.float64),
    )


def _validate_branch_record_for_npz(
    rec: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    if not isinstance(rec, Mapping):
        raise TypeError(f"{source}: branch record must be a mapping.")
    required = (
        "bitstrings",
        "weights",
        "counts_bitstrings",
        "counts_values",
        "raw_quasi_bitstrings",
        "raw_quasi_weights",
        "metadata",
    )
    missing = [key for key in required if key not in rec]
    if missing:
        raise ValueError(f"{source}: missing required field(s): {', '.join(missing)}.")

    bits, weights = _validate_bitstrings_and_weights(
        rec["bitstrings"],
        rec["weights"],
        source=source,
    )
    total_weight = math.fsum(float(value) for value in weights)
    if total_weight <= 0.0 or not math.isclose(
        total_weight,
        1.0,
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"{source}: realization weights must have positive sum approximately "
            f"equal to 1; got {total_weight!r}."
        )
    width = int(bits.shape[1])
    count_bits, count_values = _validate_bitstrings_and_weights(
        rec["counts_bitstrings"],
        rec["counts_values"],
        source=source,
        weight_name="counts_values",
        require_integer_weights=True,
        expected_width=width,
    )
    quasi_bits, quasi_weights = _validate_bitstrings_and_weights(
        rec["raw_quasi_bitstrings"],
        rec["raw_quasi_weights"],
        source=source,
        weight_name="raw_quasi_weights",
        allow_negative=True,
        expected_width=width,
    )
    metadata = rec["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{source}: metadata must be a mapping.")
    if "circuit_index" not in metadata:
        raise ValueError(f"{source}: metadata is missing required field circuit_index.")
    circuit_index = metadata["circuit_index"]
    if isinstance(circuit_index, (bool, np.bool_)) or not isinstance(
        circuit_index, (int, np.integer)
    ):
        raise TypeError(f"{source}: metadata circuit_index must be an integer.")
    if int(circuit_index) < 0:
        raise ValueError(f"{source}: metadata circuit_index must be nonnegative.")

    return {
        "bitstrings": bits,
        "weights": weights,
        "counts_bitstrings": count_bits,
        "counts_values": count_values,
        "raw_quasi_bitstrings": quasi_bits,
        "raw_quasi_weights": quasi_weights,
        "metadata": dict(metadata),
    }


def _safe_filename_piece(text: Any) -> str:
    s = str(text if text is not None else "record")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


def _validate_safe_path_component(value: Any, name: str) -> str:
    """Require a single portable path component for generated artifacts."""
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{name} must be a nonempty safe filename component.")
    if Path(value).name != value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in value):
        raise ValueError(
            f"{name} must contain only letters, digits, '.', '_', or '-' and no path separators."
        )
    return value


def save_branch_npz(
    branches: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_root: str | Path = "processed",
    prefix: str = "sqdrift",
) -> dict[str, list[str]]:
    """Save realization-level branch records without overwriting artifacts.

    Every record and final output path is validated before any directory or
    file is created.  Duplicate planned paths, pre-existing branch files, and
    a pre-existing manifest are rejected rather than overwritten.
    """
    output_root = Path(output_root)
    prefix = _validate_safe_path_component(prefix, "prefix")

    # Validate every supplied record and plan every final path before creating
    # any output directory.  In particular, distinct record IDs can sanitize
    # to the same filename and must be treated as a collision.
    validated: dict[str, list[dict[str, Any]]] = {}
    for raw_variant, records in branches.items():
        variant = _validate_safe_path_component(raw_variant, "branch variant")
        if variant in validated:
            raise ValueError(f"duplicate branch variant {variant!r}.")
        validated[variant] = [
            _validate_branch_record_for_npz(
                rec,
                source=f"branch {variant!r} record {index}",
            )
            for index, rec in enumerate(records)
        ]

    saved: dict[str, list[str]] = {variant: [] for variant in validated}
    plan: list[tuple[dict[str, Any], Path, str]] = []
    planned_paths: set[Path] = set()
    for variant, records in validated.items():
        for rec in records:
            path = _branch_npz_path(output_root, variant, rec, prefix=prefix)
            if path in planned_paths:
                raise ValueError(
                    f"duplicate planned branch output path {path}; ensure circuit_index "
                    "and sanitized record_id combinations are unique."
                )
            planned_paths.add(path)
            meta = dict(rec["metadata"])
            metadata_json = json.dumps(_jsonify(meta), ensure_ascii=False)
            plan.append((rec, path, metadata_json))
            saved[variant].append(str(path))

    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(
            f"output_root exists but is not a directory: {output_root}"
        )
    manifest_path = output_root / "saved_branch_files.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"branch manifest already exists and will not be overwritten: {manifest_path}"
        )
    incomplete_marker = output_root / _BRANCH_SAVE_INCOMPLETE_MARKER
    if incomplete_marker.exists():
        raise FileExistsError(
            "output_root is marked as an incomplete prior branch save: "
            f"{incomplete_marker}"
        )
    for parent in {path.parent for path in planned_paths}:
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError(
                f"planned branch output parent exists but is not a directory: {parent}"
            )
    for path in planned_paths:
        if path.exists():
            raise FileExistsError(
                f"branch output already exists and will not be overwritten: {path}"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    with incomplete_marker.open("x", encoding="utf-8") as marker_file:
        marker_file.write("Branch artifacts are not complete until the manifest is committed.\n")

    for rec, path, metadata_json in plan:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Opening the destination in exclusive mode closes the race between
        # the existence preflight above and the actual archive write.
        with path.open("xb") as output_file:
            np.savez_compressed(
                output_file,
                bitstrings=rec["bitstrings"],
                weights=rec["weights"],
                counts_bitstrings=rec["counts_bitstrings"],
                counts_values=rec["counts_values"],
                raw_quasi_bitstrings=rec["raw_quasi_bitstrings"],
                raw_quasi_weights=rec["raw_quasi_weights"],
                metadata_json=np.array(metadata_json),
            )

    # Use exclusive creation here as well so a concurrently created manifest
    # can never be silently replaced after the preflight.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        json.dump(
            _jsonify(saved),
            manifest_file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    incomplete_marker.unlink()
    return saved


def _branch_npz_path(
    output_root: str | Path,
    variant: str,
    rec: Mapping[str, Any],
    *,
    prefix: str,
) -> Path:
    """Return the deterministic file path for one realization-level branch record."""
    variant = _validate_safe_path_component(variant, "branch variant")
    prefix = _validate_safe_path_component(prefix, "prefix")
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
    rec = _validate_branch_record_for_npz(
        rec,
        source=f"branch {variant!r} record",
    )
    path = _branch_npz_path(output_root, variant, rec, prefix=prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(rec["metadata"])
    np.savez_compressed(
        path,
        bitstrings=rec["bitstrings"],
        weights=rec["weights"],
        counts_bitstrings=rec["counts_bitstrings"],
        counts_values=rec["counts_values"],
        raw_quasi_bitstrings=rec["raw_quasi_bitstrings"],
        raw_quasi_weights=rec["raw_quasi_weights"],
        metadata_json=np.array(json.dumps(_jsonify(meta), ensure_ascii=False)),
    )
    return path


def _iter_validated_postprocessing_branch_records(
    jobs: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    mappings: Sequence[Sequence[int]],
    job_slices: Sequence[tuple[int, int]],
    shots: int | None,
    mitigator: Any | None,
    *,
    allow_legacy_full_mapping: bool,
):
    for job, slc in zip(jobs, job_slices, strict=True):
        start, stop = slc
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
            n_bits = _require_positive_int(
                meas.num_bits,
                f"job result circuit {circuit_index} meas.num_bits",
            )
            final_mapping = _m3_qubits_for_counts(
                mapping,
                n_bits,
                allow_legacy_full_mapping=allow_legacy_full_mapping,
            )

            raw_counts = meas.get_counts()
            _, _, count_values = _counts_to_arrays(raw_counts, n_bits)
            total_shots = int(np.sum(count_values))

            # Compute and validate both variants before yielding either one.
            # The streaming saver therefore cannot leave an M3-off file for a
            # circuit whose corresponding M3-on calculation failed.
            branch_pair: list[tuple[str, dict[str, Any]]] = []
            for variant, counts, accepted in (
                ("m3_off_reset_off", raw_counts, total_shots),
                ("m3_on_reset_off", raw_counts, total_shots),
            ):
                branch_record = _make_branch_record(
                    variant=variant,
                    circuit_index=circuit_index,
                    record=record,
                    job_id=_job_id(job),
                    job_local_index=local_idx,
                    counts=counts,
                    n_bits=n_bits,
                    shots=shots,
                    accepted_shots=accepted,
                    mapping=final_mapping,
                    mitigator=mitigator,
                )
                branch_record = _validate_branch_record_for_npz(
                    branch_record,
                    source=f"branch {variant!r} record for circuit {circuit_index}",
                )
                branch_pair.append((variant, branch_record))

            yield from branch_pair


def iter_postprocessing_branch_records_from_batch_jobs(
    jobs: Sequence[Any],
    run_info: Mapping[str, Any],
    *,
    backend: Any | None = None,
    m3_file: str | Path | None = None,
):
    """Return an iterator of validated ``(variant, record)`` pairs.

    Structural validation runs immediately when this function is called, before
    any job result, M3 loader, or output operation can be reached.
    """
    validated = _validate_postprocessing_inputs(jobs, run_info)
    validated_jobs, records, mappings, job_slices, shots = validated
    _validate_backend_matches_run_info(run_info, backend)
    allow_legacy_full_mapping = (
        run_info.get("reset_probe") is True and "isa_circuits" not in run_info
    )
    used_qubits = sorted({qubit for mapping in mappings for qubit in mapping})

    mitigator = run_info.get("mitigator")
    _validate_live_mitigator_matches_run_info(run_info, mitigator)
    if mitigator is None and backend is not None:
        m3_path = m3_file or run_info.get("m3_file")
        if m3_path is not None:
            mitigator = load_m3_mitigator(
                backend,
                m3_path,
                expected_backend_name=(
                    None
                    if run_info.get("backend_name") is None
                    else str(run_info["backend_name"])
                ),
                # A legacy full mapping can include probe qubits that are not
                # part of final-result M3 correction. Its final width is known
                # only after reading the result, so do not invent a calibration
                # coverage claim here.
                expected_qubits=(
                    None if allow_legacy_full_mapping else used_qubits
                ),
            )

    return _iter_validated_postprocessing_branch_records(
        validated_jobs,
        records,
        mappings,
        job_slices,
        shots,
        mitigator,
        allow_legacy_full_mapping=allow_legacy_full_mapping,
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

    Records are streamed into a private staging directory. The complete run is
    promoted to ``output_root`` only after every job and both M3 branches
    succeed, so a failed run cannot masquerade as a complete dataset.
    """
    output_root = Path(output_root)
    prefix = _validate_safe_path_component(prefix, "prefix")
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(
                f"output_root {output_root} is not an empty directory; use a fresh "
                "directory to avoid mixing stale and new sampling artifacts."
            )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{_safe_filename_piece(output_root.name)}.staging-",
            dir=output_root.parent,
        )
    )
    staged: dict[str, list[Path]] = {name: [] for name in POSTPROCESSING_BRANCHES}
    try:
        for variant, rec in iter_postprocessing_branch_records_from_batch_jobs(
            jobs,
            run_info,
            backend=backend,
            m3_file=m3_file,
        ):
            path = _save_single_branch_record_npz(
                rec,
                output_root=staging_root,
                variant=variant,
                prefix=prefix,
            )
            staged.setdefault(variant, []).append(path)

        saved = {
            variant: [
                str(output_root / path.relative_to(staging_root)) for path in paths
            ]
            for variant, paths in staged.items()
        }
        _write_json(staging_root / "saved_branch_files.json", saved)
        if output_root.exists():
            output_root.rmdir()  # verified empty above
        staging_root.replace(output_root)
        return saved
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def read_branch_npz_metadata(path: str | Path) -> dict[str, Any]:
    """Read the JSON metadata stored in one realization-level `.npz` file."""
    with np.load(path, allow_pickle=False) as data:
        raw = data["metadata_json"]
        text = str(raw.item() if hasattr(raw, "item") else raw)
    return json.loads(text)


# =============================================================================
# Branch folders to clustering-input files
# =============================================================================


def _load_validated_npz_arrays(
    path: str | Path,
    *,
    weight_key: str,
    expected_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = ("bitstrings", weight_key)
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(
                f"{path}: missing required NPZ key(s): {', '.join(missing)}."
            )
        raw_bits = data["bitstrings"]
        raw_weights = data[weight_key]
        managed_branch_artifact = "metadata_json" in data.files
        bits, weights = _validate_bitstrings_and_weights(
            raw_bits,
            raw_weights,
            source=str(path),
            weight_name=weight_key,
            expected_width=expected_width,
        )
    if managed_branch_artifact and weight_key == "weights":
        total_weight = math.fsum(float(value) for value in weights)
        if total_weight <= 0.0 or not math.isclose(
            total_weight,
            1.0,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"{path}: managed realization weights must have positive sum "
                f"approximately equal to 1; got {total_weight!r}."
            )
    return bits, weights


def _validate_managed_branch_manifest(folder: Path, files: Sequence[Path]) -> None:
    """Require a complete manifest for NPZ files produced by branch savers."""
    incomplete_marker = folder.parent / _BRANCH_SAVE_INCOMPLETE_MARKER
    if incomplete_marker.exists():
        raise ValueError(
            f"Managed branch folder {folder} has no completion manifest because "
            f"{incomplete_marker} marks an incomplete sampling save."
        )
    managed = False
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            managed = managed or "metadata_json" in data.files

    manifest_path = folder.parent / "saved_branch_files.json"
    if not managed and not manifest_path.exists():
        return
    if not manifest_path.is_file():
        raise ValueError(
            f"Managed branch folder {folder} has no completion manifest at "
            f"{manifest_path}; the sampling output may be incomplete."
        )

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping) or folder.name not in manifest:
        raise ValueError(
            f"Completion manifest {manifest_path} does not describe branch "
            f"{folder.name!r}."
        )
    listed = manifest[folder.name]
    if isinstance(listed, (str, bytes)):
        raise TypeError(
            f"Completion manifest entry for {folder.name!r} must be a sequence."
        )
    try:
        listed_names = [Path(str(value)).name for value in listed]
    except TypeError as exc:
        raise TypeError(
            f"Completion manifest entry for {folder.name!r} must be iterable."
        ) from exc
    actual_names = [path.name for path in files]
    if len(set(listed_names)) != len(listed_names) or sorted(listed_names) != sorted(
        actual_names
    ):
        raise ValueError(
            f"Completion manifest {manifest_path} does not match the NPZ files in "
            f"{folder}; listed={sorted(listed_names)}, actual={sorted(actual_names)}."
        )


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
    if not np.all(np.isfinite(out_w)):
        raise ValueError("Merging duplicate bitstrings produced non-finite weights.")
    return bits_c[first].astype(np.uint8, copy=True), out_w


def _prepare_clustering_input_from_npz_folder(
    folder: str | Path,
    *,
    normalize: bool,
    min_weight: float,
) -> dict[str, Any]:
    """Read, validate, and merge a folder without creating output artifacts."""
    folder = Path(folder)
    threshold = _finite_real_value(min_weight, "min_weight")
    if threshold < 0.0:
        raise ValueError(f"min_weight must be nonnegative; got {threshold}.")

    files = sorted(
        p for p in folder.glob("*.npz") if not p.name.startswith("clustering_input")
    )
    if len(files) == 0:
        raise FileNotFoundError(f"No realization .npz files found in {folder}.")
    _validate_managed_branch_manifest(folder, files)

    # Validate all files, including empty files, before merging any of them.
    validated_files: list[tuple[Path, np.ndarray, np.ndarray]] = []
    expected_width: int | None = None
    for path in files:
        bits, weights = _load_validated_npz_arrays(
            path,
            weight_key="weights",
            expected_width=expected_width,
        )
        if expected_width is None:
            expected_width = int(bits.shape[1])
        validated_files.append((path, bits, weights))

    nonempty = [entry for entry in validated_files if len(entry[1]) > 0]
    if not nonempty:
        raise ValueError(f"All realization files in {folder} were empty.")

    bitstrings = np.vstack([bits for _, bits, _ in nonempty])
    weights = np.concatenate([weights for _, _, weights in nonempty])
    source_files = [str(path) for path, _, _ in nonempty]

    bitstrings, weights = _merge_duplicate_rows_sum_weights(bitstrings, weights)
    keep = weights > threshold
    bitstrings = bitstrings[keep]
    weights = weights[keep]

    total_before_normalization = float(np.sum(weights))
    if not math.isfinite(total_before_normalization) or total_before_normalization <= 0.0:
        raise ValueError("Merged branch must have a finite positive total weight.")

    probabilities = weights / total_before_normalization if normalize else weights
    probabilities = np.ascontiguousarray(probabilities, dtype=np.float64)
    bitstrings = np.ascontiguousarray(bitstrings, dtype=np.uint8)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("Merged clustering probabilities must be finite and nonnegative.")

    return {
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "bitstrings": bitstrings,
        "probabilities": probabilities,
        "n_unique": int(len(bitstrings)),
        "total_weight_before_normalization": total_before_normalization,
        "source_files": source_files,
        "normalized": bool(normalize),
    }


def _write_prepared_clustering_input(
    output_file: str | Path,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        bitstrings=prepared["bitstrings"],
        probabilities=prepared["probabilities"],
        source_files=np.asarray(prepared["source_files"], dtype=str),
        n_realization_files=np.array(len(prepared["source_files"]), dtype=np.int64),
        n_unique=np.array(prepared["n_unique"], dtype=np.int64),
        total_weight_before_normalization=np.array(
            prepared["total_weight_before_normalization"], dtype=np.float64
        ),
        normalized=np.array(prepared["normalized"]),
        package_version=np.array(PACKAGE_VERSION),
        algorithm_version=np.array(ALGORITHM_VERSION),
    )
    return {
        "output_file": str(output_file),
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "bitstrings": prepared["bitstrings"],
        "probabilities": prepared["probabilities"],
        "n_unique": int(prepared["n_unique"]),
        "total_weight_before_normalization": float(
            prepared["total_weight_before_normalization"]
        ),
        "source_files": list(prepared["source_files"]),
    }


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
    prepared = _prepare_clustering_input_from_npz_folder(
        folder,
        normalize=normalize,
        min_weight=min_weight,
    )
    if output_file is None:
        output_file = folder / "clustering_input.npz"
    return _write_prepared_clustering_input(output_file, prepared)


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

    # Preflight every variant before creating the shared output directory or
    # writing a partial set of merged files.
    safe_variants = [
        _validate_safe_path_component(variant, "variant") for variant in variants
    ]
    if len(set(safe_variants)) != len(safe_variants):
        raise ValueError("variants must not contain duplicates.")
    prepared = {
        variant: _prepare_clustering_input_from_npz_folder(
            processed_root / variant,
            normalize=True,
            min_weight=0.0,
        )
        for variant in safe_variants
    }

    out: dict[str, dict[str, Any]] = {}
    for variant, variant_prepared in prepared.items():
        out_file = output_dir / f"{variant}.npz"
        out[variant] = _write_prepared_clustering_input(out_file, variant_prepared)
    _write_json(output_dir / "clustering_inputs_summary.json", {k: {kk: vv for kk, vv in v.items() if kk not in {"bitstrings", "probabilities"}} for k, v in out.items()})
    return out


def load_clustering_input(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load arrays ready for `assign_clusters_bmm(bitstrings, probabilities, k=...)`."""
    return _load_validated_npz_arrays(path, weight_key="probabilities")


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
    "build_all_clustering_inputs",
    "make_clustering_inputs_from_processed",
    "load_clustering_input",
    "load_clustering_input_npz",
]
