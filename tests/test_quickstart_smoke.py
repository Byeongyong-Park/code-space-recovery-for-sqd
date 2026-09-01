"""Small, credentials-free smoke test matching the README quick start."""

from __future__ import annotations

import numpy as np

from code_space_recovery.diagonalization import (
    make_projected_pauli_primme_diagonalize_fn,
)
from code_space_recovery.hamiltonians import make_1d_tfim_sparse_pauliop
from code_space_recovery.recovery import (
    make_mrelu_recovery_fn,
    run_code_space_recovery,
)


def test_readme_quickstart_runs_without_hardware_or_credentials() -> None:
    encoded_samples = np.array(
        [[0, 0, 0, 1], [1, 1, 1, 0]],
        dtype=np.uint8,
    )
    sample_weights = np.array([0.6, 0.4], dtype=np.float64)
    clustered_samples = ((encoded_samples, sample_weights),)

    hamiltonian = make_1d_tfim_sparse_pauliop(
        2,
        zz_coeff=-1.0,
        x_coeff=-0.5,
        periodic=False,
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

    assert np.isfinite(result.best_energy)
    assert result.best_logical_basis.ndim == 2
    assert result.best_logical_basis.shape[1] == 2
    assert result.best_logical_basis.shape[0] == result.best_coefficients.shape[0]
    np.testing.assert_allclose(
        np.sum(np.abs(result.best_coefficients) ** 2),
        1.0,
        rtol=1e-10,
        atol=1e-12,
    )
