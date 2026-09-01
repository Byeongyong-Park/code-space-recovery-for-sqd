# Code-space recovery for SQD

This package recovers pair-encoded measurement samples and diagonalizes the
logical Hamiltonian in the recovered subspace for sample-based quantum
diagonalization (SQD). It provides reusable methods associated with the paper.

## Versions and reproducibility

The Zenodo deposit accompanying the article is the **authoritative, immutable
paper snapshot**. Use that archive for exact reproduction of the reported
results, including the archived source, notebooks, inputs, and result tables.

The pre-maintenance GitHub package was labeled `v1.0.0`. Its scientific and
numerical implementation matches the archived source; the observed file-level
differences are documentation/comments and non-computational diagnostic text.
It is therefore computationally aligned with the paper code, although it is
not a byte-for-byte archival copy. Use Zenodo when the exact deposited files,
not only the calculation path, are required.

The `v2.0.0` package and the current `main` branch are maintained,
post-study software releases. They clarify the supported encoding convention,
public API, provenance metadata, documentation, packaging, and tests. The
scientific recovery algorithm remains `code_space_recovery_v1.0`.

Some legacy result dictionaries retain `module_version="v1.0"` or an
equivalent `core_version` field so existing artifact readers continue to work.
Those are compatibility fields, not the installed distribution version. New
provenance should use `package_version` together with `algorithm_version`.

## Features

- 1D and 2D Ising benchmark Hamiltonians
- pair-code encoding of Hamiltonians and state-preparation circuits
- SqDRIFT circuit construction for Pauli-sum Hamiltonians and weighted sample preparation
- probability-weighted Bernoulli-mixture clustering
- reference-guided recovery of invalid encoded pairs
- iterative batching, projected diagonalization, reference updates, and carryover
- PRIMME diagonalization and full-Hamiltonian energy/variance evaluation

## Installation

Python 3.10 or later is required; CI tests Python 3.10 through 3.13. From the
repository root:

```bash
python -m pip install .
```

The projected diagonalizer depends on `primme`, which is distributed as source
on PyPI. A local C/C++ build toolchain may therefore be required, especially on
Windows. The repository CI performs the full installation and quick-start test
on Ubuntu.

IBM Quantum Runtime and M3 support are optional:

```bash
python -m pip install ".[hardware]"
```

For local tests:

```bash
python -m pip install ".[test]"
python -m pytest -p no:cacheprovider
```

## Quick start

This small API example recovers two encoded samples and performs one projected
iteration for a two-site transverse-field Ising model.

```python
import numpy as np

from code_space_recovery.diagonalization import make_projected_pauli_primme_diagonalize_fn
from code_space_recovery.hamiltonians import make_1d_tfim_sparse_pauliop
from code_space_recovery.recovery import make_mrelu_recovery_fn, run_code_space_recovery

encoded_samples = np.array(
    [[0, 0, 0, 1], [1, 1, 1, 0]],
    dtype=np.uint8,
)
sample_weights = np.array([0.6, 0.4], dtype=np.float64)
clustered_samples = ((encoded_samples, sample_weights),)

hamiltonian = make_1d_tfim_sparse_pauliop(
    2, zz_coeff=-1.0, x_coeff=-0.5, periodic=False
)
diagonalize = make_projected_pauli_primme_diagonalize_fn(
    hamiltonian,
    eig_tol=1e-10,
    ncv=None,
)

result = run_code_space_recovery(
    clustered_samples,
    hamiltonian,
    n_batches=1,
    max_dim=2,
    min_iterations=1,
    max_iterations=1,
    convergence_patience=1,
    carryover_threshold=1e-3,
    diagonalize_fn=diagonalize,
    recovery_fn=make_mrelu_recovery_fn(delta=0.01, corner=0.5),
    max_recovery_draws_per_iteration=1,
    seed=7,
)

print(result.best_energy)
print(len(result.best_logical_basis))
```

For small projected subspaces, prefer `ncv=None` as shown above so PRIMME can
choose a basis size compatible with the projected dimension. A fixed large
`ncv` can be unsuitable for small toy problems.

See [ALGORITHM.md](ALGORITHM.md) for the recovery procedure and stopping rules.

## Public API

Stable imports use the defining submodule rather than the package root:

| Module | Public responsibility |
| --- | --- |
| `code_space_recovery.clustering` | weighted BMM clustering |
| `code_space_recovery.encoding` | logical-to-pair-code Hamiltonian encoding |
| `code_space_recovery.state_encoding` | pair-code state preparation |
| `code_space_recovery.sqdrift` | logical SqDRIFT circuit generation |
| `code_space_recovery.sampling` | sampling and mitigation utilities |
| `code_space_recovery.recovery` | iterative code-space recovery |
| `code_space_recovery.diagonalization` | projected sparse diagonalization |
| `code_space_recovery.energy_variance` | full-Hamiltonian energy and variance |
| `code_space_recovery.hamiltonians` | benchmark Hamiltonian builders |

Each module's `__all__` defines its supported public names. Underscore-prefixed
helpers are internal implementation details. Existing submodule import paths
are retained in `v2.0.0`; see [MIGRATION.md](MIGRATION.md) for the one removed
encoding mode.

## Input conventions

`clustered_samples` is a non-empty sequence of
`(encoded_bitstrings, weights)` pairs:

- `encoded_bitstrings` has shape `(N, 2 * n_logical)`, dtype `np.uint8`, and
  contains only `0` and `1`;
- `weights` has shape `(N,)` and contains finite, nonnegative values;
- the total weight over all clusters is finite and strictly positive; and
- encoded rows are globally unique and form a hard partition across clusters.

Duplicate raw outcomes must be merged before clustering. Weights may be raw
counts or global weights; the recovery driver normalizes them once over the
complete sample pool.

Public data-loading and recovery helpers validate binary arrays before
converting them to `np.uint8`. Malformed bits, non-finite values, invalid
cluster labels, and incompatible precompiled-Hamiltonian settings raise an
exception before numerical or hardware work begins.

Hamiltonian iterators that can be consumed only once require an explicit
`num_qubits`. Pauli Hamiltonians must be Hermitian, and callers should merge
duplicate Pauli terms before applying a coefficient cutoff. Recovery schedules
reject unknown keys. Saved sampling runs also reject mismatched job IDs,
measurement mappings, backend provenance, incomplete managed branch outputs,
and realization probability arrays whose sum is not approximately one.
Checkpoint savers require a new or empty run directory. The submission saver
also accepts the common calibration-first layout in which that directory
contains exactly its own regular M3 calibration file and nothing else; all
other pre-existing entries, reserved filenames, symlinks, and incomplete
checkpoint markers are rejected before artifact writes begin.

The pair code is

```text
logical 0 -> encoded 01
logical 1 -> encoded 10
```

Pairs `00` and `11` are invalid. Arrays follow Qiskit's displayed-bit order,
where the rightmost logical bit corresponds to qubit 0, and the two rails for
each logical bit remain adjacent.

For logical qubit `q`, the first displayed rail is physical qubit `2q + 1` and
the second is physical qubit `2q`. Hamiltonian encoding supports the single
explicit mode `qiskit_label`, with displayed-pair map
`I -> II`, `X -> XX`, `Y -> YX`, and `Z -> ZI`. The former
`qiskit_qubit_index` mode is not supported in `v2.0.0` because it reverses the
rail placement of asymmetric Pauli pairs.

## Reproducibility

Batch execution supports `threading` and optional `loky` backends. Record the
package version, algorithm version, seed, backend, dimensions, and numerical
settings for reproducibility. Different dependency versions, numerical
libraries, and hardware can still introduce floating-point differences.

## Citation

If you use this software, please cite:

> Byeongyong Park, Sanha Kang, Doyeol Ahn, and Keunhong Jeong,
> "Code-space recovery for sample-based quantum diagonalization beyond native
> symmetry constraints," [arXiv:2607.10227](https://arxiv.org/abs/2607.10227),
> 2026.

Machine-readable metadata is provided in `CITATION.cff`.

## License

Released under the MIT License. See [LICENSE](LICENSE).
