"""Fail-fast validation tests for pair-code recovery helpers."""

from __future__ import annotations

import numpy as np
import pytest

import code_space_recovery.recovery as recovery
from code_space_recovery.recovery import (
    RecoveryResult,
    assign_to_nearest_reference,
    build_one_batch,
    build_sparse_invalid_pairs,
    decode_valid_encoded_to_logical,
    encode_logical_to_valid_encoded,
    initialize_reference_vectors,
    is_valid_encoded_bitstrings,
    make_relu_pair_recovery_prob_fn,
    mrelu_recover_encoded_samples,
    pairwise_normalize_reference_vectors,
    select_carryover_basis,
    update_reference_vectors,
)


@pytest.mark.parametrize(
    "bad_bits",
    [
        np.array([[0.0, 1.5]]),
        np.array([[0, 256]], dtype=np.int64),
        np.array([[0, 257]], dtype=np.int64),
        np.array([[0.0, np.nan]]),
        np.array([[0.0, np.inf]]),
    ],
)
def test_binary_helpers_reject_bad_values_before_uint8_cast(
    bad_bits: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        encode_logical_to_valid_encoded(bad_bits)


def test_binary_helpers_reject_non_numeric_dtype() -> None:
    with pytest.raises(TypeError, match="numeric binary dtype"):
        encode_logical_to_valid_encoded(np.array([["0", "1"]]))


def test_exact_float_binary_values_are_accepted_and_encoded_identically() -> None:
    logical = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    encoded = encode_logical_to_valid_encoded(logical)

    np.testing.assert_array_equal(
        encoded,
        np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.uint8),
    )
    assert encoded.dtype == np.uint8


def test_validity_check_does_not_wrap_large_integer_to_valid_bit() -> None:
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        is_valid_encoded_bitstrings(np.array([[0, 257]], dtype=np.int64))


def test_decode_validates_binary_values_even_when_pair_check_is_disabled() -> None:
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        decode_valid_encoded_to_logical(
            np.array([[257, 0]], dtype=np.int64),
            check_valid=False,
        )


def test_decode_requires_boolean_check_valid_flag() -> None:
    with pytest.raises(TypeError, match="check_valid must be a bool"):
        decode_valid_encoded_to_logical(
            np.array([[0, 1]], dtype=np.uint8),
            check_valid=1,  # type: ignore[arg-type]
        )


def test_reference_normalization_keeps_zero_pair_neutral() -> None:
    normalized = pairwise_normalize_reference_vectors(
        np.array([[0.0, 0.0, 2.0, 6.0]])
    )

    np.testing.assert_allclose(normalized, [[0.5, 0.5, 0.25, 0.75]])


@pytest.mark.parametrize(
    "bad_reference",
    [
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([[-1.0, 2.0]]),
    ],
)
def test_reference_normalization_rejects_nonfinite_or_negative_values(
    bad_reference: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        pairwise_normalize_reference_vectors(bad_reference)


def test_initialize_reference_vectors_validates_direct_helper_inputs() -> None:
    references = initialize_reference_vectors(
        [
            (
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([1.0, 3.0]),
            ),
            (
                np.empty((0, 2), dtype=np.uint8),
                np.empty(0, dtype=np.float64),
            ),
        ],
        n_logical=1,
    )

    np.testing.assert_allclose(references, [[0.75, 0.25], [0.5, 0.5]])


def test_initialize_reference_vectors_rejects_wrapping_bit_value() -> None:
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        initialize_reference_vectors(
            [(np.array([[256, 1]]), np.array([1.0]))],
            n_logical=1,
        )


@pytest.mark.parametrize("bad_chunk_size", [0, -1, 1.5, True])
def test_assign_requires_positive_integer_chunk_size(bad_chunk_size: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        assign_to_nearest_reference(
            np.array([[0, 1]], dtype=np.uint8),
            np.array([[0.5, 0.5]]),
            chunk_size=bad_chunk_size,  # type: ignore[arg-type]
        )


def test_assign_accepts_binary_float_input_without_changing_result() -> None:
    labels = assign_to_nearest_reference(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.25, 0.75], [0.75, 0.25]]),
        chunk_size=1,
    )

    np.testing.assert_array_equal(labels, [0, 1])


@pytest.mark.parametrize(
    "bad_reference",
    [
        np.array([[0.2, 0.2]]),
        np.array([[-0.1, 1.1]]),
        np.array([[np.nan, 1.0]]),
    ],
)
def test_assign_requires_finite_nonnegative_normalized_reference(
    bad_reference: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        assign_to_nearest_reference(
            np.array([[0, 1]], dtype=np.uint8),
            bad_reference,
        )


def _sparse_inputs() -> tuple[np.ndarray, np.ndarray]:
    bitstrings = np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    references = np.array(
        [
            [0.5, 0.5, 0.25, 0.75],
            [0.75, 0.25, 0.5, 0.5],
        ]
    )
    return bitstrings, references


@pytest.mark.parametrize(
    "bad_labels, expected_exception",
    [
        (np.array([-1, 0]), ValueError),
        (np.array([0, 2]), ValueError),
        (np.array([0.0, 1.0]), TypeError),
        (np.array([[0, 1]]), ValueError),
        (np.array([0]), ValueError),
        (np.array([False, True]), TypeError),
    ],
)
def test_sparse_invalid_pairs_rejects_bad_labels_before_indexing(
    bad_labels: np.ndarray,
    expected_exception: type[Exception],
) -> None:
    bitstrings, references = _sparse_inputs()

    with pytest.raises(expected_exception):
        build_sparse_invalid_pairs(bitstrings, bad_labels, references)


def test_sparse_invalid_pairs_preserves_valid_label_mapping() -> None:
    bitstrings, references = _sparse_inputs()

    sparse = build_sparse_invalid_pairs(
        bitstrings,
        np.array([0, 1], dtype=np.int64),
        references,
    )

    np.testing.assert_array_equal(sparse.sample_indices, [0, 1])
    np.testing.assert_array_equal(sparse.pair_indices, [0, 0])
    np.testing.assert_array_equal(sparse.cluster_indices, [0, 1])
    np.testing.assert_array_equal(sparse.invalid_pair_bits, [[0, 0], [1, 1]])
    np.testing.assert_allclose(
        sparse.invalid_pair_reference,
        [[0.5, 0.5], [0.75, 0.25]],
    )


def test_mrelu_rejects_bad_bitstrings_before_recovery_callback() -> None:
    callback_called = False

    def callback(bits: np.ndarray, references: np.ndarray) -> np.ndarray:
        nonlocal callback_called
        callback_called = True
        return np.full(len(bits), 0.5)

    with pytest.raises(ValueError, match="exactly 0 or 1"):
        mrelu_recover_encoded_samples(
            np.array([[256, 1]], dtype=np.int64),
            np.array([1.0]),
            np.array([0], dtype=np.int64),
            np.array([[0.5, 0.5]]),
            callback,
            np.random.default_rng(7),
        )

    assert callback_called is False


@pytest.mark.parametrize(
    "bad_weights",
    [
        np.array([-1.0]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array(["1.0"]),
    ],
)
def test_mrelu_rejects_invalid_weights(bad_weights: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        mrelu_recover_encoded_samples(
            np.array([[0, 0]], dtype=np.uint8),
            bad_weights,
            np.array([0], dtype=np.int64),
            np.array([[0.5, 0.5]]),
            lambda bits, refs: np.full(len(bits), 0.5),
            np.random.default_rng(7),
        )


def test_mrelu_rejects_duplicate_weight_accumulation_overflow() -> None:
    with pytest.raises(ValueError, match="merged weights must remain finite"):
        mrelu_recover_encoded_samples(
            np.array([[0, 1], [0, 1]], dtype=np.uint8),
            np.array([np.finfo(np.float64).max, np.finfo(np.float64).max]),
            np.array([0, 0], dtype=np.int64),
            np.array([[0.5, 0.5]]),
            lambda bits, refs: np.full(len(bits), 0.5),
            np.random.default_rng(7),
        )


def test_mrelu_rejects_non_numeric_probability_output() -> None:
    with pytest.raises(TypeError, match="real numeric dtype"):
        mrelu_recover_encoded_samples(
            np.array([[0, 0]], dtype=np.uint8),
            np.array([1.0]),
            np.array([0], dtype=np.int64),
            np.array([[0.5, 0.5]]),
            lambda bits, refs: np.array(["0.5"]),
            np.random.default_rng(7),
        )


def test_builtin_probability_callback_validates_its_contract() -> None:
    probability_fn = make_relu_pair_recovery_prob_fn()

    with pytest.raises(ValueError, match="exactly 0 or 1"):
        probability_fn(np.array([[257, 257]]), np.array([[0.5, 0.5]]))
    with pytest.raises(ValueError, match="must sum to 1"):
        probability_fn(np.array([[0, 0]]), np.array([[0.2, 0.2]]))
    with pytest.raises(ValueError, match="invalid pairs 00 or 11"):
        probability_fn(np.array([[0, 1]]), np.array([[0.5, 0.5]]))


def test_builtin_probability_callback_accepts_exact_float_binary_pairs() -> None:
    probability_fn = make_relu_pair_recovery_prob_fn()

    probabilities = probability_fn(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[0.25, 0.75], [0.75, 0.25]]),
    )

    assert probabilities.shape == (2,)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_custom_recovery_result_is_validated_before_binary_cast() -> None:
    result = RecoveryResult(
        bitstrings=np.array([[0, 257]], dtype=np.int64),
        weights=np.array([1.0]),
        metadata={},
    )

    with pytest.raises(ValueError, match="exactly 0 or 1"):
        recovery._validate_recovery_result(result, n_encoded=2)


def test_custom_recovery_result_requires_mapping_metadata() -> None:
    result = RecoveryResult(
        bitstrings=np.array([[0.0, 1.0]]),
        weights=np.array([1.0]),
        metadata=[],  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="metadata must be a mapping"):
        recovery._validate_recovery_result(result, n_encoded=2)


def test_custom_recovery_result_accepts_exact_float_binary_values() -> None:
    result = RecoveryResult(
        bitstrings=np.array([[0.0, 1.0]]),
        weights=np.array([1.0]),
        metadata={"source": "test"},
    )

    validated = recovery._validate_recovery_result(result, n_encoded=2)

    np.testing.assert_array_equal(validated.bitstrings, [[0, 1]])
    assert validated.bitstrings.dtype == np.uint8


def _build_one_batch(
    recovered_pool: np.ndarray,
    recovered_weights: np.ndarray,
    *,
    max_dim: object = 2,
) -> np.ndarray:
    return build_one_batch(
        recovered_pool,
        recovered_weights,
        deterministic_logical_basis=np.empty((0, 2), dtype=np.uint8),
        carryover_logical_basis=np.empty((0, 2), dtype=np.uint8),
        max_dim=max_dim,  # type: ignore[arg-type]
        rng=np.random.default_rng(11),
    )


def test_build_one_batch_rejects_wrapping_bits_before_sampling() -> None:
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        _build_one_batch(
            np.array([[256, 1], [1, 0]], dtype=np.int64),
            np.array([0.5, 0.5]),
        )


def test_build_one_batch_validates_aligned_finite_nonnegative_weights() -> None:
    pool = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    with pytest.raises(ValueError, match="one entry per recovered logical row"):
        _build_one_batch(pool, np.array([1.0]))
    with pytest.raises(ValueError, match="finite values"):
        _build_one_batch(pool, np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="nonnegative"):
        _build_one_batch(pool, np.array([1.0, -1.0]))


@pytest.mark.parametrize("bad_max_dim", [0, -1, 1.5, True])
def test_build_one_batch_requires_positive_integer_max_dim(
    bad_max_dim: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _build_one_batch(
            np.array([[0, 1]], dtype=np.uint8),
            np.array([1.0]),
            max_dim=bad_max_dim,
        )


def test_build_one_batch_keeps_normal_seeded_sampling_path() -> None:
    batch = _build_one_batch(
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        np.array([0.0, 1.0, 0.0]),
        max_dim=2,
    )

    assert batch.dtype == np.uint8
    assert batch.shape == (2, 2)
    assert any(np.array_equal(row, [0, 1]) for row in batch)


def test_select_carryover_rejects_wrapping_bits_before_selection() -> None:
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        select_carryover_basis(
            np.array([[257, 0]], dtype=np.int64),
            np.array([1.0]),
            carryover_threshold=0.1,
            max_keep=1,
        )


def test_select_carryover_validates_coefficients_and_settings() -> None:
    basis = np.array([[0, 1]], dtype=np.uint8)

    with pytest.raises(ValueError, match="one entry per logical basis row"):
        select_carryover_basis(
            basis,
            np.empty(0),
            carryover_threshold=0.1,
            max_keep=1,
        )
    with pytest.raises(ValueError, match="finite values"):
        select_carryover_basis(
            basis,
            np.array([np.nan]),
            carryover_threshold=0.1,
            max_keep=1,
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        select_carryover_basis(
            basis,
            np.array([1.0]),
            carryover_threshold=1.1,
            max_keep=1,
        )
    with pytest.raises(TypeError, match="non-negative integer"):
        select_carryover_basis(
            basis,
            np.array([1.0]),
            carryover_threshold=0.1,
            max_keep=True,
        )


def test_select_carryover_accepts_float_binary_basis_without_result_change() -> None:
    selected = select_carryover_basis(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([0.25, 0.75]),
        carryover_threshold=0.2,
        max_keep=1,
    )

    np.testing.assert_array_equal(selected, [[1, 0]])
    assert selected.dtype == np.uint8


def test_update_reference_vectors_validates_inputs_before_assignment() -> None:
    old_reference = np.array([[0.5, 0.5, 0.5, 0.5]])

    with pytest.raises(ValueError, match="exactly 0 or 1"):
        update_reference_vectors(
            np.array([[257, 0]]),
            np.array([1.0]),
            old_reference,
        )
    with pytest.raises(ValueError, match="must sum to 1"):
        update_reference_vectors(
            np.array([[0, 1]], dtype=np.uint8),
            np.array([1.0]),
            np.array([[0.2, 0.2, 0.5, 0.5]]),
        )
    with pytest.raises(ValueError, match="best_coefficients must contain only finite"):
        update_reference_vectors(
            np.array([[0, 1]], dtype=np.uint8),
            np.array([np.nan]),
            old_reference,
        )
    with pytest.raises(ValueError, match="finite positive squared norm"):
        update_reference_vectors(
            np.array([[0, 1]], dtype=np.uint8),
            np.array([0.0]),
            old_reference,
        )


def test_update_reference_vectors_checks_chunk_size_even_for_empty_basis() -> None:
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        update_reference_vectors(
            np.empty((0, 2), dtype=np.uint8),
            np.empty(0),
            np.array([[0.5, 0.5, 0.5, 0.5]]),
            chunk_size=0,
        )


def test_update_reference_vectors_keeps_normal_path_result() -> None:
    updated = update_reference_vectors(
        np.array([[0.0, 1.0]]),
        np.array([1.0 + 0.0j]),
        np.array([[0.5, 0.5, 0.5, 0.5]]),
        chunk_size=1,
    )

    np.testing.assert_allclose(updated, [[0.0, 1.0, 1.0, 0.0]])


@pytest.mark.parametrize(
    "bad_weights",
    [
        np.array(["1.0"]),
        np.array([1.0 + 1.0j]),
        np.array([np.nan]),
        np.array([-1.0]),
    ],
)
def test_main_recovery_input_rejects_invalid_weights_before_cast(
    bad_weights: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        recovery._validate_and_flatten_clustered_samples(
            ((np.array([[0, 1]], dtype=np.uint8), bad_weights),),
            n_logical=1,
        )


@pytest.mark.parametrize("num_qubits", [True, 1.5, "1"])
def test_main_recovery_rejects_coerced_hamiltonian_width(num_qubits: object) -> None:
    class InvalidHamiltonian:
        pass

    hamiltonian = InvalidHamiltonian()
    hamiltonian.num_qubits = num_qubits

    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        recovery.run_code_space_recovery(
            ((np.array([[0, 1]], dtype=np.uint8), np.array([1.0])),),
            hamiltonian,
            n_batches=1,
            max_dim=1,
            min_iterations=1,
            max_iterations=1,
            convergence_patience=1,
            carryover_threshold=1e-3,
            diagonalize_fn=lambda *_args, **_kwargs: None,
            recovery_prob_fn=lambda bits, refs: np.full(len(bits), 0.5),
        )


@pytest.mark.parametrize("unknown_key", ["iteration", "convergence_patence"])
def test_recovery_schedule_rejects_unknown_keys_before_stage_use(
    unknown_key: str,
) -> None:
    callback_called = False

    def recovery_fn(*_args: object) -> RecoveryResult:
        nonlocal callback_called
        callback_called = True
        raise AssertionError("recovery callback must not run")

    with pytest.raises(ValueError, match=r"unsupported key\(s\)") as caught:
        recovery._normalize_recovery_schedule_entry(
            {
                "recovery_fn": recovery_fn,
                unknown_key: 1,
            },
            stage_index=0,
            default_min_iterations=1,
            default_max_iterations=2,
            default_convergence_patience=1,
        )

    assert unknown_key in str(caught.value)
    assert callback_called is False


def test_weighted_sampling_handles_large_finite_weights_without_overflow() -> None:
    selected = recovery._weighted_sample_without_replacement(
        np.array(
            [np.finfo(np.float64).max, np.finfo(np.float64).max],
            dtype=np.float64,
        ),
        1,
        np.random.default_rng(7),
    )

    assert selected.shape == (1,)
    assert int(selected[0]) in {0, 1}


def test_weighted_sampling_handles_probability_underflow_without_replacement() -> None:
    selected = recovery._weighted_sample_without_replacement(
        np.array(
            [
                np.finfo(np.float64).max,
                np.nextafter(0.0, 1.0),
                np.nextafter(0.0, 1.0),
            ],
            dtype=np.float64,
        ),
        2,
        np.random.default_rng(7),
    )

    assert 0 in selected
    assert len(np.unique(selected)) == 2
