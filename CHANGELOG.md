# Changelog

All notable changes to the maintained package are documented here. Package
versions describe the software release; the separately recorded algorithm
version describes the scientific recovery procedure.

## [2.0.0] - Unreleased

### Changed

- Designated `qiskit_label` as the only supported Hamiltonian-encoding mode.
- Removed the ambiguous `qiskit_qubit_index` rail-placement mode.
- Centralized the package version and separated it from the algorithm version.
- Defined module-level public APIs and provenance metadata consistently.
- Clarified installation, conventions, archival provenance, and migration.
- Added convention, API-contract, smoke, packaging, and CI checks.
- Added fail-fast validation for binary recovery inputs, cluster references,
  Hamiltonian coefficients, sampling counts, and NPZ clustering artifacts.
- Made energy-variance metadata report the effective compiled-Hamiltonian
  settings and reject explicitly conflicting settings.
- Made the projected diagonalizer reject a call-time Hamiltonian that differs
  from its construction-time Hamiltonian, while accepting an independently
  constructed Hamiltonian with the same canonical compiled representation.
- Validated caller-supplied compiled Hamiltonians completely before variance
  kernels use their masks, offsets, coefficient caches, and metadata.
- Made direct branch-NPZ saving reject filename collisions and existing target
  artifacts instead of silently overwriting them.
- Documented `ncv=None` as the recommended PRIMME setting for small projected
  subspaces.
- Made the canonical pair-Pauli map immutable and reject non-finite encoding
  coefficients before constructing an encoded operator.
- Made recovery schedules reject unknown keys so configuration typos cannot
  silently select a different fixed or adaptive iteration policy.
- Rejected one-shot Hamiltonian iterators without an explicit `num_qubits`,
  non-Hermitian canonical Hamiltonians, and duplicate terms whose cutoff order
  would make diagonalization and SqDRIFT use different operators.
- Bound sampling results to their saved job IDs, circuit records, measurement
  mappings, backend, and available M3 calibration provenance before result or
  mitigation work begins.
- Made copied sampling checkpoints use their bundle-local canonical artifacts,
  marked incomplete direct branch saves, required matching completion manifests
  for managed branch folders, and validated per-realization probability sums.
- Made sampling-checkpoint writes fail fast on non-fresh destinations,
  incomplete-marker filename collisions, and unsafe calibration paths, while
  preserving the calibration-first layout when the sole existing run artifact
  is that directory's own regular M3 file.
- Materialized validated one-shot sampling iterables once and reused that
  normalized snapshot for submission, serialization, QPY, and checkpoint
  metadata so validation cannot exhaust later consumers.

### Scientific scope

The recovery procedure remains algorithm version
`code_space_recovery_v1.0`. Version 2.0.0 is a maintained, post-study software
release; it is not the source snapshot used to produce the archived paper
results.

## [1.0.0] - 2026

Initial public package associated with the article. This packaged source is
computationally aligned with the authoritative source snapshot in the
article's Zenodo deposit. The observed differences are documentation/comments
and non-computational diagnostic text, so the files are not byte-for-byte
identical. Use the Zenodo archive when the exact deposited files are required.
