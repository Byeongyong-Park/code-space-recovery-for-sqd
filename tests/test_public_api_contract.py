"""Version and import contracts for the installable public package."""

from __future__ import annotations

from importlib import import_module, metadata
from pathlib import Path
import re

import code_space_recovery


EXPECTED_PUBLIC_SYMBOLS = {
    "clustering": (
        "ClusteredSamples",
        "assign_clusters_bmm",
        "cluster_weight_sums",
    ),
    "diagonalization": (
        "DEFAULT_MAX_LOGICAL_QUBITS",
        "PauliTerm",
        "CompiledPauliHamiltonian",
        "ProjectedCSR",
        "ProjectedPauliBuildStats",
        "ProjectedPauliCSRPRIMMEDiagonalizer",
        "compile_pauli_hamiltonian",
        "make_projected_pauli_primme_diagonalize_fn",
        "csr_to_dense_for_debug",
    ),
    "encoding": (
        "PauliEncodingMode",
        "DEFAULT_PAIR_PAULI_MAP",
        "EncodingMetadata",
        "encode_pauli_label",
        "encode_sparse_pauliop",
        "encode_sparse_pauliop_with_metadata",
        "summarize_encoding",
    ),
    "energy_variance": (
        "ALGORITHM_VERSION",
        "MODULE_VERSION",
        "EnergyVarianceResult",
        "compute_full_hamiltonian_variance",
        "compute_full_hamiltonian_variance_from_sqd_result",
        "make_energy_variance_csv_row",
        "write_energy_variance_csv",
    ),
    "hamiltonians": (
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
    ),
    "recovery": (
        "DiagonalizationResult",
        "CodeSpaceRecoveryResult",
        "RecoveryResult",
        "SparseInvalidPairs",
        "encode_logical_to_valid_encoded",
        "is_valid_encoded_bitstrings",
        "decode_valid_encoded_to_logical",
        "pairwise_normalize_reference_vectors",
        "initialize_reference_vectors",
        "assign_to_nearest_reference",
        "build_sparse_invalid_pairs",
        "mrelu_recover_encoded_samples",
        "build_one_batch",
        "select_carryover_basis",
        "update_reference_vectors",
        "run_code_space_recovery",
        "modified_relu_score",
        "make_relu_pair_recovery_prob_fn",
        "make_mrelu_recovery_fn",
    ),
    "sampling": (
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
    ),
    "sqdrift": ("generate_sqdrift_logical_circuits",),
    "state_encoding": (
        "PairEncodedStateCircuitMetadata",
        "pair_code_rail_qubits",
        "encode_state_preparation_circuit_pair_code",
    ),
}


def test_package_and_distribution_versions_are_v2() -> None:
    assert code_space_recovery.PACKAGE_VERSION == "2.0.0"
    assert code_space_recovery.__version__ == "2.0.0"
    assert metadata.version("code-space-recovery-for-sqd") == "2.0.0"


def test_citation_version_matches_the_installed_package() -> None:
    citation = (Path(__file__).parents[1] / "CITATION.cff").read_text(
        encoding="utf-8"
    )
    match = re.search(r'^version:\s*"([^"]+)"\s*$', citation, re.MULTILINE)
    assert match is not None
    assert match.group(1) == code_space_recovery.__version__


def test_algorithm_version_does_not_change_with_the_packaging_release() -> None:
    assert code_space_recovery.ALGORITHM_VERSION == "code_space_recovery_v1.0"


def test_documented_public_submodules_and_symbols_remain_importable() -> None:
    for module_name, symbols in EXPECTED_PUBLIC_SYMBOLS.items():
        module = import_module(f"code_space_recovery.{module_name}")
        assert tuple(module.__all__) == symbols
        for symbol in symbols:
            assert hasattr(module, symbol), f"{module.__name__}.{symbol} is missing"


def test_package_root_stays_lightweight_and_version_only() -> None:
    assert tuple(code_space_recovery.__all__) == (
        "PACKAGE_VERSION",
        "ALGORITHM_VERSION",
        "__version__",
    )


def test_sampling_implementation_helpers_are_not_public_exports() -> None:
    sampling = import_module("code_space_recovery.sampling")
    assert {"_mapping_values", "_simple_compiled_summary"}.isdisjoint(
        sampling.__all__
    )
