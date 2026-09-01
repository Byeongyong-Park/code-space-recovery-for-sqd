"""Fail-fast contracts for weighted BMM inputs."""

from __future__ import annotations

import numpy as np
import pytest

from code_space_recovery.clustering import _validate_bmm_inputs


def test_bmm_rejects_finite_weights_whose_total_overflows() -> None:
    bitstrings = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    probabilities = np.array(
        [np.finfo(np.float64).max, np.finfo(np.float64).max],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="finite and positive"):
        _validate_bmm_inputs(
            bitstrings,
            probabilities,
            k=1,
            random_state=7,
            n_init=1,
        )


def test_bmm_normal_probability_input_is_numerically_unchanged() -> None:
    bitstrings = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    probabilities = np.array([0.6, 0.4], dtype=np.float64)

    checked_bits, checked_probabilities, k, random_state, n_init = (
        _validate_bmm_inputs(
            bitstrings,
            probabilities,
            k=1,
            random_state=7,
            n_init=2,
        )
    )

    np.testing.assert_array_equal(checked_bits, bitstrings)
    np.testing.assert_array_equal(checked_probabilities, probabilities)
    assert (k, random_state, n_init) == (1, 7, 2)
