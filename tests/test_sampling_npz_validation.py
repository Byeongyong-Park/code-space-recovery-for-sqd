"""NPZ schema and all-files-preflight contracts for sampling artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from code_space_recovery import sampling


def _write_npz(
    path: Path,
    *,
    bitstrings: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    if bitstrings is not None:
        arrays["bitstrings"] = bitstrings
    if weights is not None:
        arrays["weights"] = weights
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize(
    "case",
    [
        "missing_bitstrings",
        "missing_weights",
        "nonnumeric_bitstrings",
        "fractional_bitstrings",
        "wrapping_bitstrings",
        "bitstrings_not_2d",
        "nonbinary_bitstrings",
        "odd_encoded_width",
        "weights_not_1d",
        "weight_row_mismatch",
        "nonfinite_weights",
        "negative_weights",
    ],
)
def test_invalid_realization_npz_fails_before_output_creation(
    tmp_path: Path,
    case: str,
) -> None:
    folder = tmp_path / "processed"
    path = folder / "realization.npz"
    bits: np.ndarray | None = np.array([[0, 1]], dtype=np.uint8)
    weights: np.ndarray | None = np.array([1.0], dtype=np.float64)

    if case == "missing_bitstrings":
        bits = None
    elif case == "missing_weights":
        weights = None
    elif case == "nonnumeric_bitstrings":
        bits = np.array([["0", "1"]])
    elif case == "fractional_bitstrings":
        bits = np.array([[0.0, 0.5]])
    elif case == "wrapping_bitstrings":
        bits = np.array([[256, 257]], dtype=np.int64)
    elif case == "bitstrings_not_2d":
        bits = np.array([0, 1], dtype=np.uint8)
    elif case == "nonbinary_bitstrings":
        bits = np.array([[0, 2]], dtype=np.uint8)
    elif case == "odd_encoded_width":
        bits = np.array([[0, 1, 0]], dtype=np.uint8)
    elif case == "weights_not_1d":
        weights = np.array([[1.0]], dtype=np.float64)
    elif case == "weight_row_mismatch":
        weights = np.array([0.5, 0.5], dtype=np.float64)
    elif case == "nonfinite_weights":
        weights = np.array([np.nan], dtype=np.float64)
    elif case == "negative_weights":
        weights = np.array([-0.1], dtype=np.float64)

    _write_npz(path, bitstrings=bits, weights=weights)
    output_file = tmp_path / "new_output" / "merged.npz"

    with pytest.raises((TypeError, ValueError)):
        sampling.make_clustering_input_from_npz_folder(
            folder,
            output_file=output_file,
        )

    assert not output_file.parent.exists()


def test_every_file_is_preflighted_before_output_creation(tmp_path: Path) -> None:
    folder = tmp_path / "processed"
    _write_npz(
        folder / "a_valid.npz",
        bitstrings=np.array([[0, 1]], dtype=np.uint8),
        weights=np.array([1.0]),
    )
    _write_npz(
        folder / "z_bad_width.npz",
        bitstrings=np.array([[0, 1, 1, 0]], dtype=np.uint8),
        weights=np.array([1.0]),
    )
    output_file = tmp_path / "new_output" / "merged.npz"

    with pytest.raises(ValueError, match="expected width"):
        sampling.make_clustering_input_from_npz_folder(
            folder,
            output_file=output_file,
        )

    assert not output_file.parent.exists()


def test_empty_npz_must_still_have_the_common_even_width(tmp_path: Path) -> None:
    folder = tmp_path / "processed"
    _write_npz(
        folder / "a_empty.npz",
        bitstrings=np.empty((0, 2), dtype=np.uint8),
        weights=np.empty(0, dtype=np.float64),
    )
    _write_npz(
        folder / "b_valid.npz",
        bitstrings=np.array([[0, 1]], dtype=np.uint8),
        weights=np.array([1.0]),
    )

    result = sampling.make_clustering_input_from_npz_folder(folder)

    assert result["n_unique"] == 1
    np.testing.assert_array_equal(result["bitstrings"], np.array([[0, 1]], dtype=np.uint8))


def test_malformed_empty_npz_is_not_silently_skipped(tmp_path: Path) -> None:
    folder = tmp_path / "processed"
    _write_npz(
        folder / "a_empty_bad.npz",
        bitstrings=np.empty((0, 3), dtype=np.uint8),
        weights=np.empty(0, dtype=np.float64),
    )
    _write_npz(
        folder / "b_valid.npz",
        bitstrings=np.array([[0, 1]], dtype=np.uint8),
        weights=np.array([1.0]),
    )
    output_file = tmp_path / "new_output" / "merged.npz"

    with pytest.raises(ValueError, match="positive even"):
        sampling.make_clustering_input_from_npz_folder(
            folder,
            output_file=output_file,
        )

    assert not output_file.parent.exists()


def test_valid_npz_files_merge_duplicate_rows_without_changing_math(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "processed"
    _write_npz(
        folder / "a.npz",
        bitstrings=np.array([[0, 1], [1, 0]], dtype=np.uint8),
        weights=np.array([0.2, 0.3]),
    )
    _write_npz(
        folder / "b.npz",
        bitstrings=np.array([[0, 1]], dtype=np.uint8),
        weights=np.array([0.5]),
    )

    result = sampling.make_clustering_input_from_npz_folder(folder)
    merged = {
        tuple(row.tolist()): float(weight)
        for row, weight in zip(
            result["bitstrings"], result["probabilities"], strict=True
        )
    }

    assert result["bitstrings"].dtype == np.uint8
    assert set(merged) == {(0, 1), (1, 0)}
    assert merged[(0, 1)] == pytest.approx(0.7)
    assert merged[(1, 0)] == pytest.approx(0.3)


def test_all_variants_are_preflighted_before_shared_output_directory(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed"
    _write_npz(
        processed_root / "valid" / "a.npz",
        bitstrings=np.array([[0, 1]], dtype=np.uint8),
        weights=np.array([1.0]),
    )
    _write_npz(
        processed_root / "invalid" / "b.npz",
        bitstrings=np.array([[0.0, 0.5]]),
        weights=np.array([1.0]),
    )
    output_dir = tmp_path / "all_outputs"

    with pytest.raises(ValueError, match="exact 0/1"):
        sampling.make_all_clustering_inputs(
            processed_root,
            output_dir=output_dir,
            variants=("valid", "invalid"),
        )

    assert not output_dir.exists()


def test_clustering_input_loader_validates_before_casting(tmp_path: Path) -> None:
    path = tmp_path / "clustering_input.npz"
    np.savez_compressed(
        path,
        bitstrings=np.array([[0, 1]], dtype=np.int64),
        probabilities=np.array([1.0]),
    )

    loaded_bits, loaded_probabilities = sampling.load_clustering_input(path)

    assert loaded_bits.dtype == np.uint8
    np.testing.assert_array_equal(loaded_bits, [[0, 1]])
    np.testing.assert_array_equal(loaded_probabilities, [1.0])


def test_clustering_input_loader_rejects_wrapping_values(tmp_path: Path) -> None:
    path = tmp_path / "bad_clustering_input.npz"
    np.savez_compressed(
        path,
        bitstrings=np.array([[256, 257]], dtype=np.int64),
        probabilities=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="exact 0/1"):
        sampling.load_clustering_input(path)


def test_branch_save_rejects_nonbinary_values_before_creating_output(tmp_path: Path) -> None:
    record = {
        "bitstrings": np.array([[0.0, 0.5]]),
        "weights": np.array([1.0]),
        "counts_bitstrings": np.array([[0, 1]], dtype=np.uint8),
        "counts_values": np.array([10.0]),
        "raw_quasi_bitstrings": np.empty((0, 2), dtype=np.uint8),
        "raw_quasi_weights": np.empty(0, dtype=np.float64),
        "metadata": {"circuit_index": 0, "record_id": "r0"},
    }
    output_root = tmp_path / "saved"

    with pytest.raises(ValueError, match="exact 0/1"):
        sampling.save_branch_npz(
            {"m3_off_reset_off": [record]},
            output_root=output_root,
        )

    assert not output_root.exists()


def test_managed_branch_npz_requires_normalized_realization_weights(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    folder = root / "m3_off_reset_off"
    path = folder / "sqdrift_circuit_0000_r0.npz"
    folder.mkdir(parents=True)
    np.savez_compressed(
        path,
        bitstrings=np.array([[0, 1]], dtype=np.uint8),
        weights=np.array([0.5]),
        metadata_json=np.array(json.dumps({"circuit_index": 0, "record_id": "r0"})),
    )
    (root / "saved_branch_files.json").write_text(
        json.dumps({"m3_off_reset_off": [str(path)]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum approximately equal to 1"):
        sampling.make_clustering_input_from_npz_folder(folder)
