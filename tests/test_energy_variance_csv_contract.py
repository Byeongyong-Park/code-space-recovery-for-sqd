"""CSV schema-integrity contracts for energy-variance outputs."""

from __future__ import annotations

import csv

import pytest

from code_space_recovery.energy_variance import write_energy_variance_csv


def test_append_reuses_existing_header_order(tmp_path) -> None:
    path = tmp_path / "variance.csv"
    path.write_text("energy,module_version\n-1.0,v1.0\n", encoding="utf-8")

    write_energy_variance_csv(
        path,
        [{"module_version": "v1.0", "energy": -1.1}],
        append=True,
    )

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"energy": "-1.0", "module_version": "v1.0"},
        {"energy": "-1.1", "module_version": "v1.0"},
    ]


def test_append_rejects_v1_v2_schema_mismatch_without_modifying_file(
    tmp_path,
) -> None:
    path = tmp_path / "variance.csv"
    original = "energy,module_version\n-1.0,v1.0\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="CSV schema mismatch"):
        write_energy_variance_csv(
            path,
            [
                {
                    "energy": -1.1,
                    "module_version": "v1.0",
                    "package_version": "2.0.0",
                }
            ],
            append=True,
        )

    assert path.read_text(encoding="utf-8") == original
