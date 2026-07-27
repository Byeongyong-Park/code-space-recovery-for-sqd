# Code-space recovery for SQD

This package recovers pair-encoded measurement samples and diagonalizes the
logical Hamiltonian in the recovered subspace for sample-based quantum
diagonalization (SQD). It provides reusable methods associated with the paper.

## Features

- 1D and 2D Ising benchmark Hamiltonians
- pair-code encoding of Hamiltonians and state-preparation circuits
- SqDRIFT circuit construction for Pauli-sum Hamiltonians and weighted sample preparation
- probability-weighted Bernoulli-mixture clustering
- reference-guided recovery of invalid encoded pairs
- iterative batching, projected diagonalization, reference updates, and carryover
- PRIMME diagonalization and full-Hamiltonian energy/variance evaluation

## Installation

Python 3.10 or later is required. From the repository root:

```bash
python -m pip install .
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

See [ALGORITHM.md](ALGORITHM.md) for the recovery procedure and stopping rules.

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

The pair code is

```text
logical 0 -> encoded 01
logical 1 -> encoded 10
```

Pairs `00` and `11` are invalid. Arrays follow Qiskit's displayed-bit order,
where the rightmost logical bit corresponds to qubit 0, and the two rails for
each logical bit remain adjacent.

## Reproducibility

Batch execution supports `threading` and optional `loky` backends. Record the
seed, backend, dimensions, and numerical settings for reproducibility.

## Citation

If you use this software, please cite:

> Byeongyong Park, Sanha Kang, Doyeol Ahn, and Keunhong Jeong,
> "Code-space recovery for sample-based quantum diagonalization beyond native
> symmetry constraints," [arXiv:2607.10227](https://arxiv.org/abs/2607.10227),
> 2026.

Machine-readable metadata is provided in `CITATION.cff`.

## License

Released under the MIT License. See [LICENSE](LICENSE).
