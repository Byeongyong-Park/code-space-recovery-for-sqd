"""Fail-fast contracts for sampling settings and probability inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from code_space_recovery import sampling


class _FakeISACircuit:
    num_clbits = 2
    num_qubits = 2
    cregs: tuple[object, ...] = ()


class _FakeBitLocation:
    def __init__(self, index: int) -> None:
        self.index = index


class _FakeRegister(tuple[object, ...]):
    def __new__(cls, name: str, bits: tuple[object, ...]):
        instance = super().__new__(cls, bits)
        instance.name = name
        return instance


class _FakeLegacyISACircuit:
    num_clbits = 4
    num_qubits = 5

    def __init__(self) -> None:
        self._bits = tuple(object() for _ in range(self.num_clbits))
        self.cregs = (
            _FakeRegister("probe", self._bits[:2]),
            _FakeRegister("meas", self._bits[2:]),
        )

    def find_bit(self, bit: object) -> _FakeBitLocation:
        return _FakeBitLocation(self._bits.index(bit))


class _FakeMeasureOperation:
    name = "measure"


class _FakeCircuitInstruction:
    def __init__(self, qubit: object, clbit: object) -> None:
        self.operation = _FakeMeasureOperation()
        self.qubits = (qubit,)
        self.clbits = (clbit,)


class _InspectableISACircuit:
    num_clbits = 2
    num_qubits = 2

    def __init__(
        self,
        measured_qubits_by_cbit: tuple[int, int],
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._qubits = tuple(object() for _ in range(self.num_qubits))
        self._clbits = tuple(object() for _ in range(self.num_clbits))
        self.cregs = (_FakeRegister("meas", self._clbits),)
        self.metadata = metadata
        self.data = [
            _FakeCircuitInstruction(self._qubits[qubit], self._clbits[cbit])
            for cbit, qubit in enumerate(measured_qubits_by_cbit)
        ]

    def find_bit(self, bit: object) -> _FakeBitLocation:
        if bit in self._qubits:
            return _FakeBitLocation(self._qubits.index(bit))
        return _FakeBitLocation(self._clbits.index(bit))


class _NamedBackend:
    def __init__(self, name: str) -> None:
        self.name = name


def _valid_compiled_run_info() -> dict[str, object]:
    return {
        "records": [{"record_id": "r0"}],
        "isa_circuits": [_FakeISACircuit()],
        "mappings": [[0, 1]],
        "used_qubits_for_m3": [0, 1],
    }


def _valid_branch_record(
    circuit_index: int = 0,
    record_id: str = "r0",
) -> dict[str, object]:
    return {
        "bitstrings": np.array([[0, 1]], dtype=np.uint8),
        "weights": np.array([1.0]),
        "counts_bitstrings": np.array([[0, 1]], dtype=np.uint8),
        "counts_values": np.array([10.0]),
        "raw_quasi_bitstrings": np.array([[0, 1]], dtype=np.uint8),
        "raw_quasi_weights": np.array([1.0]),
        "metadata": {
            "circuit_index": circuit_index,
            "record_id": record_id,
        },
    }


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("2", TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_invalid_ghz_size_fails_before_qiskit_dependency_check(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    error_type: type[Exception],
) -> None:
    dependency_check_reached = False

    def _unexpected_dependency_check() -> None:
        nonlocal dependency_check_reached
        dependency_check_reached = True
        raise AssertionError("Qiskit dependency check must not be reached")

    monkeypatch.setattr(sampling, "_require_qiskit", _unexpected_dependency_check)

    with pytest.raises(error_type, match="n_logical"):
        sampling.make_logical_ghz_initial_state(value)  # type: ignore[arg-type]

    assert not dependency_check_reached


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("shots", True, TypeError),
        ("shots", 1.0, TypeError),
        ("shots", 0, ValueError),
        ("chunk_size", False, TypeError),
        ("chunk_size", -1, ValueError),
        ("max_circuits_per_batch", 2.0, TypeError),
        ("max_circuits_per_batch", 0, ValueError),
    ],
)
def test_invalid_submission_settings_fail_before_runtime_or_m3(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    dependency_check_reached = False

    def _unexpected_dependency_check() -> None:
        nonlocal dependency_check_reached
        dependency_check_reached = True
        raise AssertionError("external dependency check must not be reached")

    monkeypatch.setattr(sampling, "_require_runtime", _unexpected_dependency_check)
    kwargs: dict[str, object] = {
        "shots": 100,
        "chunk_size": 10,
        "max_circuits_per_batch": 20,
    }
    kwargs[field] = value

    with pytest.raises(error_type, match=field):
        sampling.run_sampling_jobs_from_compiled(
            _valid_compiled_run_info(),
            backend=object(),
            **kwargs,
        )

    assert not dependency_check_reached


def test_backend_mismatch_fails_before_runtime_or_m3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _valid_compiled_run_info()
    compiled["backend_name"] = "compiled_backend"

    class OtherBackend:
        name = "other_backend"

    dependency_check_reached = False

    def _unexpected_dependency_check() -> None:
        nonlocal dependency_check_reached
        dependency_check_reached = True

    monkeypatch.setattr(sampling, "_require_runtime", _unexpected_dependency_check)
    with pytest.raises(ValueError, match="does not match"):
        sampling.run_sampling_jobs_from_compiled(
            compiled,
            backend=OtherBackend(),
            shots=100,
        )

    assert not dependency_check_reached


def test_submitted_run_info_persists_actual_sampling_job_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMitigator:
        system_info = {"name": "backend"}

        @staticmethod
        def cals_from_system(**_: object) -> list[object]:
            return []

    class _FakeMThree:
        M3Mitigation = lambda self, backend: _FakeMitigator()

    class _FakeBatch:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "_FakeBatch":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    class _DDOptions:
        enable = False
        sequence_type: str | None = None

    class _Options:
        dynamical_decoupling = _DDOptions()

    class _FakeSampler:
        options = _Options()

        def __init__(self, **_: object) -> None:
            pass

        @staticmethod
        def run(*_: object, **__: object) -> "_IdentifiedResultMustNotRunJob":
            return _IdentifiedResultMustNotRunJob("submitted-job")

    monkeypatch.setattr(sampling, "_require_runtime", lambda: None)
    monkeypatch.setattr(sampling, "_require_mthree", lambda: None)
    monkeypatch.setattr(sampling, "mthree", _FakeMThree())
    monkeypatch.setattr(sampling, "Batch", _FakeBatch)
    monkeypatch.setattr(sampling, "Sampler", _FakeSampler)
    record = {"record_id": "r0"}
    isa_circuit = _FakeISACircuit()
    compiled = {
        "records": (item for item in [record]),
        "isa_circuits": (item for item in [isa_circuit]),
        "mappings": (
            (qubit for qubit in mapping)
            for mapping in [(0, 1)]
        ),
        "used_qubits_for_m3": (qubit for qubit in (0, 1)),
        "backend_name": "backend",
    }

    jobs, run_info = sampling.run_sampling_jobs_from_compiled(
        compiled,
        backend=_NamedBackend("backend"),
        shots=10,
    )

    assert [job.job_id() for job in jobs] == ["submitted-job"]
    assert run_info["sampling_job_ids"] == ["submitted-job"]
    assert run_info["records"] == [record]
    assert run_info["isa_circuits"] == [isa_circuit]
    assert run_info["mappings"] == [[0, 1]]
    assert run_info["used_qubits_for_m3"] == [0, 1]


def test_wrapper_validates_submission_settings_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compilation_reached = False

    def _unexpected_compile(**_: object) -> dict[str, object]:
        nonlocal compilation_reached
        compilation_reached = True
        raise AssertionError("compilation must not be reached")

    monkeypatch.setattr(sampling, "compile_sampling_circuits", _unexpected_compile)

    with pytest.raises(ValueError, match="shots"):
        sampling.run_sampling_jobs_batch({}, backend=object(), shots=0)

    assert not compilation_reached


@pytest.mark.parametrize(
    ("keyword", "value", "error_type"),
    [
        ("optimization_level", True, TypeError),
        ("optimization_level", 1.5, TypeError),
        ("optimization_level", -1, ValueError),
        ("optimization_level", 4, ValueError),
        ("print_summary", "False", TypeError),
    ],
)
def test_invalid_compile_settings_fail_before_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    keyword: str,
    value: object,
    error_type: type[Exception],
) -> None:
    dependency_check_reached = False

    def _unexpected_dependency_check() -> None:
        nonlocal dependency_check_reached
        dependency_check_reached = True

    monkeypatch.setattr(sampling, "_require_runtime", _unexpected_dependency_check)
    kwargs: dict[str, object] = {"optimization_level": 3, "print_summary": True}
    kwargs[keyword] = value
    with pytest.raises(error_type):
        sampling.compile_sampling_circuits({}, backend=object(), **kwargs)

    assert not dependency_check_reached


@pytest.mark.parametrize(
    "compiled",
    [
        {
            "records": [],
            "isa_circuits": [_FakeISACircuit()],
            "mappings": [[0, 1]],
        },
        {
            "records": [{}],
            "isa_circuits": [_FakeISACircuit()],
            "mappings": [[0]],
        },
        {
            "records": [{}],
            "isa_circuits": [_FakeISACircuit()],
            "mappings": [[0, 0]],
        },
        {
            "records": [{}],
            "isa_circuits": [_FakeISACircuit()],
            "mappings": [[0, 2]],
        },
        {
            "records": [{}],
            "isa_circuits": [_FakeISACircuit()],
            "mappings": [[0, 1]],
            "used_qubits_for_m3": [0, 0],
        },
    ],
)
def test_invalid_compiled_structure_fails_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    compiled: dict[str, object],
) -> None:
    dependency_check_reached = False

    def _unexpected_dependency_check() -> None:
        nonlocal dependency_check_reached
        dependency_check_reached = True
        raise AssertionError("external dependency check must not be reached")

    monkeypatch.setattr(sampling, "_require_runtime", _unexpected_dependency_check)

    with pytest.raises(ValueError):
        sampling.run_sampling_jobs_from_compiled(
            compiled,
            backend=object(),
            shots=100,
        )

    assert not dependency_check_reached


def test_legacy_full_mapping_is_restricted_to_named_meas_register() -> None:
    compiled = {
        "records": [{"record_id": "legacy"}],
        "isa_circuits": [_FakeLegacyISACircuit()],
        # The leading values belong to a legacy probe register.  Repetition
        # across registers is valid; the final meas mapping itself is unique.
        "mappings": [[0, 1, 0, 1]],
        # Calibration-qubit order is set-like, not an ordering contract.
        "used_qubits_for_m3": [1, 0],
    }

    _, _, mappings, used_qubits = sampling._validate_compiled_sampling_run_info(
        compiled
    )

    assert mappings == [[0, 1]]
    assert used_qubits == [0, 1]


def test_legacy_isa_accepts_already_upgraded_final_only_mapping() -> None:
    compiled = {
        "records": [{"record_id": "legacy"}],
        "isa_circuits": [_FakeLegacyISACircuit()],
        # This checkpoint has already upgraded away the leading probe mapping,
        # while its saved ISA QPY still has globally indexed probe clbits.
        "mappings": [[0, 1]],
        "used_qubits_for_m3": [0, 1],
    }

    _, _, mappings, used_qubits = sampling._validate_compiled_sampling_run_info(
        compiled
    )

    assert mappings == [[0, 1]]
    assert used_qubits == [0, 1]


def test_compiled_mapping_one_shot_iterable_is_materialized_once() -> None:
    compiled = {
        "records": [{"record_id": "r0"}],
        "isa_circuits": [_FakeISACircuit()],
        "mappings": [(value for value in (0, 1))],
    }

    _, _, mappings, _ = sampling._validate_compiled_sampling_run_info(compiled)

    assert mappings == [[0, 1]]


def test_saved_mapping_must_match_inspectable_isa_measurement_order() -> None:
    compiled = {
        "records": [{"record_id": "r0"}],
        "isa_circuits": [_InspectableISACircuit((1, 0))],
        "mappings": [[0, 1]],
    }

    with pytest.raises(ValueError, match="final measurement order"):
        sampling._validate_compiled_sampling_run_info(compiled)


def test_record_identity_must_match_isa_circuit_metadata() -> None:
    compiled = {
        "records": [{"record_id": "r0", "k": 1, "r": 2}],
        "isa_circuits": [
            _InspectableISACircuit(
                (0, 1),
                metadata={"record_id": "different", "k": 1, "r": 2},
            )
        ],
        "mappings": [[0, 1]],
    }

    with pytest.raises(ValueError, match="record/circuit metadata mismatch"):
        sampling._validate_compiled_sampling_run_info(compiled)


def test_bit_keys_require_range_without_rejecting_short_valid_strings() -> None:
    assert sampling._normalise_bit_key(1, 3) == "001"
    assert sampling._normalise_bit_key("1", 3) == "001"
    assert sampling._normalise_bit_key("001", 3) == "001"
    assert sampling._normalise_bit_key("0b1", 3) == "001"
    assert sampling._normalise_bit_key("0x1", 8) == "00000001"
    assert sampling._normalise_bit_key("0x01", 8) == "00000001"

    for key, n_bits in (
        (-1, 3),
        (8, 3),
        ("0001", 3),
        ("0x8", 3),
        ("", 3),
        ("0x", 3),
        ("0b", 3),
    ):
        with pytest.raises(ValueError):
            sampling._normalise_bit_key(key, n_bits)


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        (-1, ValueError),
        (0.5, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_physical_counts_reject_invalid_values(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="physical count"):
        sampling._counts_to_arrays({"00": value}, 2)


@pytest.mark.parametrize("counts", [{}, {"00": 0}])
def test_physical_counts_require_positive_total(counts: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive total"):
        sampling._counts_to_arrays(counts, 2)


def test_physical_counts_merge_equivalent_keys_before_normalizing() -> None:
    bits, weights, values = sampling._counts_to_arrays({1: 2, "1": 3, "01": 4}, 2)

    np.testing.assert_array_equal(bits, np.array([[0, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(values, np.array([9.0]))
    np.testing.assert_array_equal(weights, np.array([1.0]))


def test_integral_counts_are_merged_exactly_before_float_conversion() -> None:
    large = 2**53 + 1

    _, weights, values = sampling._counts_to_arrays(
        {1: large, "01": np.int64(1)},
        2,
    )

    # float(large) first would lose one count.  Exact integer merging produces
    # 2**53 + 2, which is exactly representable as float64.
    np.testing.assert_array_equal(values, np.array([float(2**53 + 2)]))
    np.testing.assert_array_equal(weights, np.array([1.0]))


def test_non_roundtrippable_raw_integer_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="represented exactly"):
        sampling._counts_to_arrays({"01": 2**53 + 1}, 2)


def test_arbitrarily_large_negative_integer_count_is_rejected_as_negative() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        sampling._counts_to_arrays({"00": -(10**400)}, 2)


def test_negative_m3_quasi_probabilities_remain_valid() -> None:
    raw = {"00": -0.25, "01": 1.25}

    converted = sampling._distribution_to_dict(
        raw,
        2,
        allow_negative=True,
        value_name="M3 quasi-probability",
    )
    projected, metadata = sampling._project_quasi_to_probability(raw, 2)

    assert converted == raw
    assert projected == {"01": 1.0}
    assert metadata["negative_mass"] == pytest.approx(0.25)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_m3_quasi_probabilities_raise(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        sampling._distribution_to_dict(
            {"00": value},
            2,
            allow_negative=True,
            value_name="M3 quasi-probability",
        )


def test_negative_projected_m3_probability_raises() -> None:
    class _BadProjection(dict[str, float]):
        def nearest_probability_distribution(self) -> dict[str, float]:
            return {"00": -0.1, "01": 1.1}

    with pytest.raises(ValueError, match="nonnegative"):
        sampling._project_quasi_to_probability(_BadProjection({"01": 1.0}), 2)


class _ResultMustNotRunJob:
    def __init__(self) -> None:
        self.result_called = False

    def result(self) -> object:
        self.result_called = True
        raise AssertionError("job.result() must not run before preflight succeeds")


class _IdentifiedResultMustNotRunJob(_ResultMustNotRunJob):
    def __init__(self, job_id: str) -> None:
        super().__init__()
        self._job_id = job_id

    def job_id(self) -> str:
        return self._job_id


def _valid_postprocessing_run_info() -> dict[str, object]:
    return {
        "records": [{"record_id": "r0"}, {"record_id": "r1"}],
        "mappings": [[0, 1], [1, 0]],
        "job_slices": [(0, 2)],
        "shots": 10,
        "mitigator": object(),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "jobs_length",
        "bool_slice",
        "noncontiguous_slices",
        "slice_out_of_range",
        "bool_mapping",
        "negative_mapping",
        "duplicate_mapping",
        "mapping_width",
    ],
)
def test_postprocessing_structure_fails_before_job_result(
    mutation: str,
) -> None:
    run_info = _valid_postprocessing_run_info()
    jobs = [_ResultMustNotRunJob()]

    if mutation == "jobs_length":
        jobs = []
    elif mutation == "bool_slice":
        run_info["job_slices"] = [(False, 2)]
    elif mutation == "noncontiguous_slices":
        jobs = [_ResultMustNotRunJob(), _ResultMustNotRunJob()]
        run_info["job_slices"] = [(0, 1), (0, 2)]
    elif mutation == "slice_out_of_range":
        run_info["job_slices"] = [(0, 3)]
    elif mutation == "bool_mapping":
        run_info["mappings"] = [[0, True], [1, 0]]
    elif mutation == "negative_mapping":
        run_info["mappings"] = [[0, -1], [1, 0]]
    elif mutation == "duplicate_mapping":
        run_info["mappings"] = [[0, 0], [1, 0]]
    elif mutation == "mapping_width":
        run_info["mappings"] = [[0, 1], [0]]

    with pytest.raises((TypeError, ValueError)):
        sampling.make_postprocessing_branches_from_batch_jobs(jobs, run_info)

    assert all(not job.result_called for job in jobs)


def test_saved_job_ids_must_match_supplied_jobs_before_result() -> None:
    job = _IdentifiedResultMustNotRunJob("supplied-job")
    run_info = _valid_postprocessing_run_info()
    run_info["sampling_job_ids"] = ["saved-job"]

    with pytest.raises(ValueError, match="sampling_job_ids"):
        sampling.make_postprocessing_branches_from_batch_jobs([job], run_info)

    assert not job.result_called


def test_live_sampling_jobs_must_match_supplied_jobs_before_result() -> None:
    supplied = _IdentifiedResultMustNotRunJob("supplied-job")
    run_info = _valid_postprocessing_run_info()
    run_info["sampling_jobs"] = [
        _IdentifiedResultMustNotRunJob("stored-job")
    ]

    with pytest.raises(ValueError, match="sampling_jobs.*IDs/order"):
        sampling.make_postprocessing_branches_from_batch_jobs(
            [supplied],
            run_info,
        )

    assert not supplied.result_called


def test_postprocessing_backend_mismatch_fails_before_result_or_m3() -> None:
    job = _IdentifiedResultMustNotRunJob("job-0")
    run_info = _valid_postprocessing_run_info()
    run_info["backend_name"] = "saved-backend"

    with pytest.raises(ValueError, match="saved sampling run"):
        sampling.make_postprocessing_branches_from_batch_jobs(
            [job],
            run_info,
            backend=_NamedBackend("other-backend"),
        )

    assert not job.result_called


def test_live_mitigator_backend_mismatch_fails_before_result() -> None:
    class _WrongBackendMitigator:
        system_info = {"name": "other-backend"}

    job = _IdentifiedResultMustNotRunJob("job-0")
    run_info = _valid_postprocessing_run_info()
    run_info["backend_name"] = "saved-backend"
    run_info["mitigator"] = _WrongBackendMitigator()

    with pytest.raises(ValueError, match="live M3 mitigator backend"):
        sampling.make_postprocessing_branches_from_batch_jobs([job], run_info)

    assert not job.result_called


def test_process_and_save_preflight_does_not_create_output(
    tmp_path: Path,
) -> None:
    job = _ResultMustNotRunJob()
    run_info = _valid_postprocessing_run_info()
    run_info["job_slices"] = [(0, 3)]
    output = tmp_path / "must_not_exist"

    with pytest.raises(ValueError):
        sampling.process_and_save_sampling_jobs(
            [job],
            run_info,
            output_root=output,
        )

    assert not job.result_called
    assert not output.exists()


class _FakeMeasResult:
    num_bits = 2

    @staticmethod
    def get_counts() -> dict[str, int]:
        return {"00": 4, "11": 6}


class _FakePubResult:
    class _Data:
        meas = _FakeMeasResult()

    data = _Data()


class _OneBitMeasResult:
    num_bits = 1

    @staticmethod
    def get_counts() -> dict[str, int]:
        return {"0": 4, "1": 6}


class _OneBitPubResult:
    class _Data:
        meas = _OneBitMeasResult()

    data = _Data()


class _OneBitJob:
    @staticmethod
    def job_id() -> str:
        return "job-one-bit"

    @staticmethod
    def result() -> list[_OneBitPubResult]:
        return [_OneBitPubResult()]


class _OneCircuitJob:
    @staticmethod
    def job_id() -> str:
        return "job-0"

    @staticmethod
    def result() -> list[_FakePubResult]:
        return [_FakePubResult()]


class _FailingMitigator:
    @staticmethod
    def apply_correction(*_: object) -> object:
        raise RuntimeError("synthetic M3 failure")


class _SuccessfulMitigator:
    @staticmethod
    def apply_correction(*_: object) -> dict[str, float]:
        return {"00": 0.4, "11": 0.6}


def test_current_result_width_must_not_shrink_mapping_as_legacy() -> None:
    run_info = {
        "records": [{"record_id": "r0"}],
        "mappings": [[0, 1]],
        "job_slices": [(0, 1)],
        "shots": 10,
        "mitigator": _SuccessfulMitigator(),
        "reset_probe": False,
    }

    with pytest.raises(ValueError, match="explicitly marked legacy"):
        sampling.make_postprocessing_branches_from_batch_jobs(
            [_OneBitJob()],
            run_info,
        )


def test_explicit_legacy_reset_probe_mapping_still_processes() -> None:
    run_info = {
        "records": [{"record_id": "legacy"}],
        "mappings": [[2, 3, 0, 1]],
        "job_slices": [(0, 1)],
        "shots": 10,
        "mitigator": _SuccessfulMitigator(),
        "reset_probe": True,
    }

    branches = sampling.make_postprocessing_branches_from_batch_jobs(
        [_OneCircuitJob()],
        run_info,
    )

    assert len(branches["m3_off_reset_off"]) == 1
    assert len(branches["m3_on_reset_off"]) == 1


def test_streaming_save_writes_neither_branch_when_m3_fails_for_circuit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"
    run_info = {
        "records": [{"record_id": "r0"}],
        "mappings": [[0, 1]],
        "job_slices": [(0, 1)],
        "shots": 10,
        "mitigator": _FailingMitigator(),
    }

    with pytest.raises(RuntimeError, match="synthetic M3 failure"):
        sampling.process_and_save_sampling_jobs(
            [_OneCircuitJob()],
            run_info,
            output_root=output,
        )

    assert not output.exists()


def test_successful_sampling_run_is_promoted_with_final_paths(tmp_path: Path) -> None:
    output = tmp_path / "processed"
    run_info = {
        "records": [{"record_id": "r0"}],
        "mappings": [[0, 1]],
        "job_slices": [(0, 1)],
        "shots": 10,
        "mitigator": _SuccessfulMitigator(),
    }

    saved = sampling.process_and_save_sampling_jobs(
        [_OneCircuitJob()],
        run_info,
        output_root=output,
    )

    assert output.is_dir()
    assert (output / "saved_branch_files.json").is_file()
    assert all(Path(path).is_file() for paths in saved.values() for path in paths)
    assert list(tmp_path.glob(".processed.staging-*")) == []


class _TwoCircuitJob(_OneCircuitJob):
    @staticmethod
    def result() -> list[_FakePubResult]:
        return [_FakePubResult(), _FakePubResult()]


class _SecondCircuitFailingMitigator:
    def __init__(self) -> None:
        self.calls = 0

    def apply_correction(self, *_: object) -> dict[str, float]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("second-circuit M3 failure")
        return {"00": 0.4, "11": 0.6}


def test_late_m3_failure_leaves_no_promoted_or_staging_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"
    run_info = {
        "records": [{"record_id": "r0"}, {"record_id": "r1"}],
        "mappings": [[0, 1], [0, 1]],
        "job_slices": [(0, 2)],
        "shots": 10,
        "mitigator": _SecondCircuitFailingMitigator(),
    }

    with pytest.raises(RuntimeError, match="second-circuit"):
        sampling.process_and_save_sampling_jobs(
            [_TwoCircuitJob()],
            run_info,
            output_root=output,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".processed.staging-*")) == []


@pytest.mark.parametrize("bad", ["../escape", "a/b", "a\\b", "", ".."])
def test_generated_artifact_path_components_reject_traversal(
    tmp_path: Path,
    bad: str,
) -> None:
    with pytest.raises(ValueError, match="filename component|path separators"):
        sampling.save_branch_npz({}, output_root=tmp_path / "out", prefix=bad)

    with pytest.raises(ValueError, match="filename component|path separators"):
        sampling.make_all_clustering_inputs(
            tmp_path / "processed",
            output_dir=tmp_path / "merged",
            variants=(bad,),
        )


def test_branch_save_rejects_duplicate_final_paths_before_creating_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"
    record = _valid_branch_record()

    with pytest.raises(ValueError, match="duplicate planned branch output path"):
        sampling.save_branch_npz(
            {"m3_off_reset_off": [record, record]},
            output_root=output,
        )

    assert not output.exists()


def test_branch_save_rejects_record_ids_that_collide_after_sanitizing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"

    with pytest.raises(ValueError, match="sanitized record_id"):
        sampling.save_branch_npz(
            {
                "m3_off_reset_off": [
                    _valid_branch_record(record_id="a/b"),
                    _valid_branch_record(record_id="a?b"),
                ]
            },
            output_root=output,
        )

    assert not output.exists()


def test_branch_save_rejects_existing_npz_without_modifying_it(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"
    existing = output / "m3_off_reset_off" / "sqdrift_circuit_0000_r0.npz"
    existing.parent.mkdir(parents=True)
    original = b"existing artifact"
    existing.write_bytes(original)

    with pytest.raises(FileExistsError, match="branch output already exists"):
        sampling.save_branch_npz(
            {"m3_off_reset_off": [_valid_branch_record()]},
            output_root=output,
        )

    assert existing.read_bytes() == original
    assert not (output / "saved_branch_files.json").exists()


def test_branch_save_rejects_existing_manifest_before_writing_npz(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"
    output.mkdir()
    manifest = output / "saved_branch_files.json"
    original = b"existing manifest"
    manifest.write_bytes(original)

    with pytest.raises(FileExistsError, match="branch manifest already exists"):
        sampling.save_branch_npz(
            {"m3_off_reset_off": [_valid_branch_record()]},
            output_root=output,
        )

    assert manifest.read_bytes() == original
    assert not (output / "m3_off_reset_off").exists()


def test_branch_save_preflights_all_variant_parents_before_writing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"
    output.mkdir()
    blocked_parent = output / "m3_on_reset_off"
    original = b"not a directory"
    blocked_parent.write_bytes(original)

    with pytest.raises(NotADirectoryError, match="output parent"):
        sampling.save_branch_npz(
            {
                "m3_off_reset_off": [_valid_branch_record(circuit_index=0)],
                "m3_on_reset_off": [_valid_branch_record(circuit_index=1)],
            },
            output_root=output,
        )

    assert blocked_parent.read_bytes() == original
    assert not (output / "m3_off_reset_off").exists()
    assert not (output / "saved_branch_files.json").exists()


def test_branch_save_serializes_all_metadata_before_creating_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"
    first = _valid_branch_record(circuit_index=0)
    invalid = _valid_branch_record(circuit_index=1)
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    invalid_metadata = invalid["metadata"]
    assert isinstance(invalid_metadata, dict)
    invalid_metadata["cyclic"] = cyclic

    with pytest.raises(RecursionError):
        sampling.save_branch_npz(
            {"m3_off_reset_off": [first, invalid]},
            output_root=output,
        )

    assert not output.exists()


def test_branch_save_exclusive_open_prevents_post_preflight_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "processed"
    target = output / "m3_off_reset_off" / "sqdrift_circuit_0000_r0.npz"
    original_open = Path.open
    injected = False

    def _racing_open(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ):
        nonlocal injected
        if path == target and mode == "xb" and not injected:
            injected = True
            with original_open(path, "wb") as competing_file:
                competing_file.write(b"competing artifact")
        return original_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", _racing_open)

    with pytest.raises(FileExistsError):
        sampling.save_branch_npz(
            {"m3_off_reset_off": [_valid_branch_record()]},
            output_root=output,
        )

    assert injected
    assert target.read_bytes() == b"competing artifact"
    assert not (output / "saved_branch_files.json").exists()


def test_branch_save_allows_same_record_id_for_distinct_circuit_indices(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"

    saved = sampling.save_branch_npz(
        {
            "m3_off_reset_off": [
                _valid_branch_record(circuit_index=0, record_id="same"),
                _valid_branch_record(circuit_index=1, record_id="same"),
            ]
        },
        output_root=output,
    )

    expected = [
        output / "m3_off_reset_off" / "sqdrift_circuit_0000_same.npz",
        output / "m3_off_reset_off" / "sqdrift_circuit_0001_same.npz",
    ]
    assert saved == {"m3_off_reset_off": [str(path) for path in expected]}
    assert all(path.is_file() for path in expected)
    with np.load(expected[0], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["bitstrings"], [[0, 1]])
        np.testing.assert_array_equal(archive["weights"], [1.0])
        assert "metadata_json" in archive.files
    assert (output / "saved_branch_files.json").is_file()


def test_legacy_saved_run_full_mapping_is_deferred_to_result_width() -> None:
    run_info = {
        "records": [{"record_id": "legacy"}],
        "mappings": [[0, 1, 0, 1]],
        "job_slices": [(0, 1)],
        "shots": 10,
        "reset_probe": True,
        "mitigator": _FailingMitigator(),
    }

    jobs, records, mappings, slices, shots = sampling._validate_postprocessing_inputs(
        [_OneCircuitJob()],
        run_info,
    )

    assert len(jobs) == 1
    assert records == [{"record_id": "legacy"}]
    assert mappings == [[0, 1, 0, 1]]
    assert slices == [(0, 1)]
    assert shots == 10
    assert sampling._m3_qubits_for_counts(
        mappings[0],
        2,
        allow_legacy_full_mapping=True,
    ) == [0, 1]


def test_submission_save_preflights_jobs_before_creating_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_info = _valid_postprocessing_run_info()

    with pytest.raises(ValueError, match="jobs/job_slices"):
        sampling.save_sampling_submission(
            run_dir,
            [],
            run_info,
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert not run_dir.exists()


def test_valid_submission_save_preserves_normal_metadata_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_info = _valid_postprocessing_run_info()
    job = _IdentifiedResultMustNotRunJob("job-0")

    saved = sampling.save_sampling_submission(
        run_dir,
        [job],
        run_info,
        save_qpy=False,
        save_hardware_snapshot=False,
    )

    assert not job.result_called
    assert Path(saved["manifest"]).is_file()
    manifest = json.loads(Path(saved["manifest"]).read_text(encoding="utf-8"))
    assert manifest["sampling_job_ids"] == ["job-0"]
    assert manifest["job_slices"] == [[0, 2]]
    assert not (run_dir / sampling._CHECKPOINT_INCOMPLETE_MARKER).exists()


def test_submission_save_requires_fresh_empty_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    existing = run_dir / "existing.txt"
    existing.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        sampling.save_sampling_submission(
            run_dir,
            [_IdentifiedResultMustNotRunJob("job-0")],
            _valid_postprocessing_run_info(),
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert existing.read_text(encoding="utf-8") == "preserve"
    assert not (run_dir / sampling._CHECKPOINT_INCOMPLETE_MARKER).exists()


def test_submission_save_rejects_reserved_m3_basename_before_writing(
    tmp_path: Path,
) -> None:
    m3_file = tmp_path / "calibration_source" / "manifest.json"
    m3_file.parent.mkdir()
    m3_file.write_text("{}", encoding="utf-8")
    run_info = _valid_postprocessing_run_info()
    run_info["m3_file"] = str(m3_file)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="reserved sampling artifact"):
        sampling.save_sampling_submission(
            run_dir,
            [_IdentifiedResultMustNotRunJob("job-0")],
            run_info,
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert not run_dir.exists()


def test_submission_save_rejects_reserved_m3_basename_even_when_missing(
    tmp_path: Path,
) -> None:
    run_info = _valid_postprocessing_run_info()
    run_info["m3_file"] = str(tmp_path / "missing" / "MANIFEST.JSON")
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="reserved sampling artifact"):
        sampling.save_sampling_submission(
            run_dir,
            [_IdentifiedResultMustNotRunJob("job-0")],
            run_info,
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert not run_dir.exists()


def test_submission_save_rejects_checkpoint_marker_m3_before_writing(
    tmp_path: Path,
) -> None:
    m3_file = tmp_path / "calibration_source" / sampling._CHECKPOINT_INCOMPLETE_MARKER
    m3_file.parent.mkdir()
    m3_file.write_text("{}", encoding="utf-8")
    run_info = _valid_postprocessing_run_info()
    run_info["m3_file"] = str(m3_file)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="reserved sampling artifact"):
        sampling.save_sampling_submission(
            run_dir,
            [_IdentifiedResultMustNotRunJob("job-0")],
            run_info,
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert not run_dir.exists()
    assert m3_file.read_text(encoding="utf-8") == "{}"


def test_submission_save_allows_sole_local_m3_file_and_preserves_it(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    m3_file = run_dir / "m3_cals.json"
    original_m3 = b'{"sentinel": "preserve-in-place"}'
    m3_file.write_bytes(original_m3)
    run_info = _valid_postprocessing_run_info()
    run_info["m3_file"] = str(m3_file)

    saved = sampling.save_sampling_submission(
        run_dir,
        [_IdentifiedResultMustNotRunJob("job-0")],
        run_info,
        save_qpy=False,
        save_hardware_snapshot=False,
    )

    assert m3_file.read_bytes() == original_m3
    assert Path(saved["m3_cals"]).resolve() == m3_file.resolve()
    assert Path(saved["manifest"]).is_file()
    assert not (run_dir / sampling._CHECKPOINT_INCOMPLETE_MARKER).exists()


def test_submission_save_rejects_local_m3_with_extra_entry_before_writing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    m3_file = run_dir / "m3_cals.json"
    m3_file.write_text("{}", encoding="utf-8")
    extra = run_dir / "existing.txt"
    extra.write_text("preserve", encoding="utf-8")
    run_info = _valid_postprocessing_run_info()
    run_info["m3_file"] = str(m3_file)

    with pytest.raises(FileExistsError, match="entries other than"):
        sampling.save_sampling_submission(
            run_dir,
            [_IdentifiedResultMustNotRunJob("job-0")],
            run_info,
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert m3_file.read_text(encoding="utf-8") == "{}"
    assert extra.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "existing.txt",
        "m3_cals.json",
    ]


def test_submission_save_requires_existing_regular_m3_file(tmp_path: Path) -> None:
    run_info = _valid_postprocessing_run_info()
    run_info["m3_file"] = str(tmp_path / "missing_m3.json")
    run_dir = tmp_path / "run"

    with pytest.raises(FileNotFoundError, match="M3 calibration file not found"):
        sampling.save_sampling_submission(
            run_dir,
            [_IdentifiedResultMustNotRunJob("job-0")],
            run_info,
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert not run_dir.exists()


def test_submission_save_rejects_m3_directory(tmp_path: Path) -> None:
    m3_directory = tmp_path / "m3_directory"
    m3_directory.mkdir()
    run_info = _valid_postprocessing_run_info()
    run_info["m3_file"] = str(m3_directory)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="not a regular file"):
        sampling.save_sampling_submission(
            run_dir,
            [_IdentifiedResultMustNotRunJob("job-0")],
            run_info,
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert not run_dir.exists()


def test_submission_save_rejects_m3_backend_mismatch_before_write(
    tmp_path: Path,
) -> None:
    m3_file = tmp_path / "m3_cals.json"
    m3_file.write_text(
        json.dumps(
            {
                "backend": "other-backend",
                "cals": [
                    [[0.9, 0.1], [0.1, 0.9]],
                    [[0.9, 0.1], [0.1, 0.9]],
                ],
            }
        ),
        encoding="utf-8",
    )
    run_info = _valid_postprocessing_run_info()
    run_info["backend_name"] = "saved-backend"
    run_info["m3_file"] = str(m3_file)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="calibration backend"):
        sampling.save_sampling_submission(
            run_dir,
            [_IdentifiedResultMustNotRunJob("job-0")],
            run_info,
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert not run_dir.exists()


def test_submission_strict_qpy_failure_leaves_incomplete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"

    def _fail_qpy(*_: object, **__: object) -> Path:
        raise RuntimeError("synthetic QPY failure")

    monkeypatch.setattr(sampling, "_save_qpy", _fail_qpy)
    with pytest.raises(RuntimeError, match="synthetic QPY failure"):
        sampling.save_sampling_submission(
            run_dir,
            [_IdentifiedResultMustNotRunJob("job-0")],
            _valid_postprocessing_run_info(),
            save_qpy=True,
            save_hardware_snapshot=False,
            fail_on_qpy_error=True,
        )

    assert (run_dir / sampling._CHECKPOINT_INCOMPLETE_MARKER).is_file()
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        sampling.load_saved_run_info(run_dir)


def test_submission_nonfatal_qpy_failure_commits_error_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"

    def _fail_qpy(*_: object, **__: object) -> Path:
        raise RuntimeError("synthetic QPY failure")

    monkeypatch.setattr(sampling, "_save_qpy", _fail_qpy)
    saved = sampling.save_sampling_submission(
        run_dir,
        [_IdentifiedResultMustNotRunJob("job-0")],
        _valid_postprocessing_run_info(),
        save_qpy=True,
        save_hardware_snapshot=False,
        fail_on_qpy_error=False,
    )

    assert Path(saved["qpy_save_errors"]).is_file()
    assert Path(saved["saved_paths"]).is_file()
    assert not (run_dir / sampling._CHECKPOINT_INCOMPLETE_MARKER).exists()


def test_compiled_checkpoint_loader_uses_only_local_canonical_qpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "copied_run"
    circuits_dir = run_dir / "circuits"
    circuits_dir.mkdir(parents=True)
    canonical = circuits_dir / "isa_circuits_transpiled.qpy"
    canonical.write_bytes(b"local canonical")
    external = tmp_path / "original_run" / "isa_circuits_transpiled.qpy"
    external.parent.mkdir()
    external.write_bytes(b"stale external")
    (run_dir / "run_info_compiled_serializable.json").write_text(
        json.dumps({"mappings": [[0, 1]]}),
        encoding="utf-8",
    )
    (run_dir / "record_table.json").write_text(
        json.dumps([{"record_id": "r0"}]),
        encoding="utf-8",
    )
    (run_dir / "saved_paths_compiled.json").write_text(
        json.dumps({"isa_circuits_qpy": str(external)}),
        encoding="utf-8",
    )
    loaded_paths: list[Path] = []

    def _fake_load(path: str | Path) -> list[str]:
        loaded_paths.append(Path(path))
        return ["local circuit"]

    monkeypatch.setattr(sampling, "load_qpy_circuits", _fake_load)

    info = sampling.load_compiled_run_info(run_dir)

    assert info["isa_circuits"] == ["local circuit"]
    assert loaded_paths == [canonical]


def test_compiled_checkpoint_save_commits_and_removes_marker(tmp_path: Path) -> None:
    run_dir = tmp_path / "compiled"

    saved = sampling.save_compiled_sampling_run(
        run_dir,
        _valid_compiled_run_info(),
        save_qpy=False,
        save_hardware_snapshot=False,
    )

    assert Path(saved["saved_paths_compiled"]).is_file()
    assert not (run_dir / sampling._CHECKPOINT_INCOMPLETE_MARKER).exists()


def test_compiled_checkpoint_one_shot_iterables_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "compiled"
    record = {"record_id": "r0"}
    isa_circuit = _FakeISACircuit()
    compiled = {
        "records": (item for item in [record]),
        "isa_circuits": (item for item in [isa_circuit]),
        "mappings": (
            (qubit for qubit in mapping)
            for mapping in [(0, 1)]
        ),
        "used_qubits_for_m3": (qubit for qubit in (0, 1)),
    }
    qpy_payloads: dict[str, list[object]] = {}

    def _fake_save_qpy(circuits: object, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        qpy_payloads[destination.name] = list(circuits)
        destination.write_bytes(b"fake qpy")
        return destination

    monkeypatch.setattr(sampling, "_save_qpy", _fake_save_qpy)
    monkeypatch.setattr(
        sampling,
        "load_qpy_circuits",
        lambda path: qpy_payloads[Path(path).name],
    )

    sampling.save_compiled_sampling_run(
        run_dir,
        compiled,
        save_qpy=True,
        save_hardware_snapshot=False,
    )
    loaded = sampling.load_compiled_run_info(run_dir)

    assert len(loaded["records"]) == 1
    assert loaded["records"][0]["record_id"] == record["record_id"]
    assert loaded["records"][0]["circuit_index"] == 0
    assert loaded["isa_circuits"] == [isa_circuit]
    assert loaded["mappings"] == [[0, 1]]
    assert loaded["used_qubits_for_m3"] == [0, 1]


def test_submission_checkpoint_one_shot_iterables_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "submission"
    record = {"record_id": "r0"}
    isa_circuit = _FakeISACircuit()
    run_info = {
        "records": (item for item in [record]),
        "isa_circuits": (item for item in [isa_circuit]),
        "mappings": (
            (qubit for qubit in mapping)
            for mapping in [(0, 1)]
        ),
        "used_qubits_for_m3": (qubit for qubit in (0, 1)),
        "job_slices": (item for item in [(0, 1)]),
        "shots": 10,
        "sampling_job_ids": (item for item in ["job-0"]),
    }
    qpy_payloads: dict[str, list[object]] = {}

    def _fake_save_qpy(circuits: object, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        qpy_payloads[destination.name] = list(circuits)
        destination.write_bytes(b"fake qpy")
        return destination

    monkeypatch.setattr(sampling, "_save_qpy", _fake_save_qpy)
    monkeypatch.setattr(
        sampling,
        "load_qpy_circuits",
        lambda path: qpy_payloads[Path(path).name],
    )

    sampling.save_sampling_submission(
        run_dir,
        [_IdentifiedResultMustNotRunJob("job-0")],
        run_info,
        save_qpy=True,
        save_hardware_snapshot=False,
    )
    loaded = sampling.load_saved_run_info(run_dir)

    assert len(loaded["records"]) == 1
    assert loaded["records"][0]["record_id"] == "r0"
    assert loaded["records"][0]["circuit_index"] == 0
    assert loaded["records"][0]["job_id"] == "job-0"
    assert loaded["records"][0]["job_position"] == 0
    assert loaded["records"][0]["job_local_index"] == 0
    assert loaded["isa_circuits"] == [isa_circuit]
    assert loaded["mappings"] == [[0, 1]]
    assert loaded["used_qubits_for_m3"] == [0, 1]
    assert loaded["job_slices"] == [[0, 1]]


def test_compiled_checkpoint_backend_mismatch_fails_before_write(
    tmp_path: Path,
) -> None:
    compiled = _valid_compiled_run_info()
    compiled["backend_name"] = "saved-backend"
    run_dir = tmp_path / "compiled"

    with pytest.raises(ValueError, match="saved sampling run"):
        sampling.save_compiled_sampling_run(
            run_dir,
            compiled,
            backend=_NamedBackend("other-backend"),
            save_qpy=False,
            save_hardware_snapshot=False,
        )

    assert not run_dir.exists()


def test_strict_qpy_failure_leaves_checkpoint_incomplete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "compiled"

    def _fail_qpy(*_: object, **__: object) -> Path:
        raise RuntimeError("synthetic QPY failure")

    monkeypatch.setattr(sampling, "_save_qpy", _fail_qpy)
    with pytest.raises(RuntimeError, match="synthetic QPY failure"):
        sampling.save_compiled_sampling_run(
            run_dir,
            _valid_compiled_run_info(),
            save_qpy=True,
            save_hardware_snapshot=False,
            fail_on_qpy_error=True,
        )

    assert (run_dir / sampling._CHECKPOINT_INCOMPLETE_MARKER).is_file()
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        sampling.load_compiled_run_info(run_dir)


def test_nonfatal_qpy_failure_still_commits_error_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "compiled"

    def _fail_qpy(*_: object, **__: object) -> Path:
        raise RuntimeError("synthetic QPY failure")

    monkeypatch.setattr(sampling, "_save_qpy", _fail_qpy)
    saved = sampling.save_compiled_sampling_run(
        run_dir,
        _valid_compiled_run_info(),
        save_qpy=True,
        save_hardware_snapshot=False,
        fail_on_qpy_error=False,
    )

    assert Path(saved["qpy_save_errors"]).is_file()
    assert Path(saved["saved_paths_compiled"]).is_file()
    assert not (run_dir / sampling._CHECKPOINT_INCOMPLETE_MARKER).exists()


def test_saved_run_loader_rebases_m3_file_inside_copied_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "copied_run"
    run_dir.mkdir()
    local_m3 = run_dir / "m3_cals.json"
    local_m3.write_text("{}", encoding="utf-8")
    external = tmp_path / "original_run" / "m3_cals.json"
    external.parent.mkdir()
    external.write_text("{}", encoding="utf-8")
    (run_dir / "run_info_serializable.json").write_text(
        json.dumps({"m3_file": str(external)}),
        encoding="utf-8",
    )
    (run_dir / "record_table.json").write_text("[]", encoding="utf-8")
    (run_dir / "saved_paths.json").write_text(
        json.dumps({"m3_cals": str(external)}),
        encoding="utf-8",
    )

    info = sampling.load_saved_run_info(run_dir)

    assert Path(info["m3_file"]) == local_m3


def test_saved_run_loader_clears_stale_external_m3_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "copied_run"
    run_dir.mkdir()
    stale = tmp_path / "original_run" / "m3_cals.json"
    (run_dir / "run_info_serializable.json").write_text(
        json.dumps({"m3_file": str(stale)}),
        encoding="utf-8",
    )
    (run_dir / "record_table.json").write_text("[]", encoding="utf-8")
    (run_dir / "saved_paths.json").write_text(
        json.dumps({"m3_cals": str(stale)}),
        encoding="utf-8",
    )

    info = sampling.load_saved_run_info(run_dir)

    assert info["m3_file"] is None


def test_saved_run_local_isa_qpy_enables_record_binding_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "copied_run"
    circuits_dir = run_dir / "circuits"
    circuits_dir.mkdir(parents=True)
    (circuits_dir / "isa_circuits_transpiled.qpy").write_bytes(b"local qpy")
    (run_dir / "run_info_serializable.json").write_text(
        json.dumps({"mappings": [[0, 1]], "used_qubits_for_m3": [0, 1]}),
        encoding="utf-8",
    )
    (run_dir / "record_table.json").write_text(
        json.dumps([{"record_id": "r0", "k": 1, "r": 0}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sampling,
        "load_qpy_circuits",
        lambda _: [
            _InspectableISACircuit(
                (0, 1),
                metadata={"record_id": "wrong", "k": 1, "r": 0},
            )
        ],
    )

    with pytest.raises(ValueError, match="record/circuit metadata mismatch"):
        sampling.load_saved_run_info(run_dir)


def test_all_checkpoint_loaders_reject_incomplete_marker(tmp_path: Path) -> None:
    run_dir = tmp_path / "incomplete"
    run_dir.mkdir()
    (run_dir / sampling._CHECKPOINT_INCOMPLETE_MARKER).write_text(
        "incomplete",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        sampling.load_saved_run_info(run_dir)
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        sampling.load_compiled_run_info(run_dir)
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        sampling.load_sampling_jobs_from_manifest(object(), run_dir)


def test_m3_file_backend_provenance_is_checked_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m3_file = tmp_path / "m3_cals.json"
    m3_file.write_text(
        json.dumps({"backend_name": "other-backend", "qubits": [0, 1]}),
        encoding="utf-8",
    )

    class _M3MustNotConstruct:
        def __init__(self, _: object) -> None:
            raise AssertionError("M3 mitigator must not be constructed")

    class _FakeMThree:
        M3Mitigation = _M3MustNotConstruct

    monkeypatch.setattr(sampling, "mthree", _FakeMThree())

    with pytest.raises(ValueError, match="calibration backend"):
        sampling.load_m3_mitigator(
            _NamedBackend("expected-backend"),
            m3_file,
            expected_backend_name="expected-backend",
            expected_qubits=[0, 1],
        )


def test_m3_file_must_cover_required_measured_qubits_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m3_file = tmp_path / "m3_cals.json"
    m3_file.write_text(
        json.dumps(
            {
                "backend": "expected-backend",
                "cals": [[[0.9, 0.1], [0.1, 0.9]], None],
            }
        ),
        encoding="utf-8",
    )

    class _M3MustNotConstruct:
        def __init__(self, _: object) -> None:
            raise AssertionError("M3 mitigator must not be constructed")

    class _FakeMThree:
        M3Mitigation = _M3MustNotConstruct

    monkeypatch.setattr(sampling, "mthree", _FakeMThree())

    with pytest.raises(ValueError, match="no calibration matrix"):
        sampling.load_m3_mitigator(
            _NamedBackend("expected-backend"),
            m3_file,
            expected_backend_name="expected-backend",
            expected_qubits=[1],
        )


def test_branch_save_rejects_non_normalized_realization_weights(
    tmp_path: Path,
) -> None:
    record = _valid_branch_record()
    record["weights"] = np.array([0.5])
    output = tmp_path / "processed"

    with pytest.raises(ValueError, match="sum approximately equal to 1"):
        sampling.save_branch_npz(
            {"m3_off_reset_off": [record]},
            output_root=output,
        )

    assert not output.exists()


def test_incomplete_managed_branch_is_not_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "processed"
    original_savez = np.savez_compressed
    calls = 0

    def _fail_second_save(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic disk failure")
        original_savez(*args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", _fail_second_save)
    with pytest.raises(OSError, match="synthetic disk failure"):
        sampling.save_branch_npz(
            {
                "m3_off_reset_off": [
                    _valid_branch_record(circuit_index=0),
                    _valid_branch_record(circuit_index=1),
                ]
            },
            output_root=output,
        )

    assert not (output / "saved_branch_files.json").exists()
    with pytest.raises(ValueError, match="no completion manifest"):
        sampling.make_clustering_input_from_npz_folder(
            output / "m3_off_reset_off"
        )
