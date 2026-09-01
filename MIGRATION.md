# Migrating from 1.x to 2.0

Version 2.0 keeps the recovery, clustering, projected-diagonalization,
Hamiltonian, sampling, and variance APIs at their existing defining module
paths. The intentional breaking change is removal of the ambiguous
`qiskit_qubit_index` Hamiltonian-encoding mode.

## Encoding convention

Use the displayed-label convention explicitly:

```python
encoded = encode_sparse_pauliop(operator, mode="qiskit_label")
```

The `mode` keyword remains available for explicitness and compatibility, but
`qiskit_label` is its only accepted value. Omitting `mode` selects the same
convention.

The `pair_pauli_map` keyword likewise remains for call-signature compatibility,
but version 2 accepts only the canonical map shown below. Arbitrary custom maps
are rejected because they can silently break the shared rail convention.

Do not mechanically replace `qiskit_qubit_index` without checking the original
operator placement. That legacy mode swapped the two physical rails for the
asymmetric `Y -> YX` and `Z -> ZI` images and therefore changed their code-space
action. Version 2 raises an explicit error instead of silently reinterpreting
such a call.

The supported invariant is:

```text
logical 0 -> 01
logical 1 -> 10
displayed pair = [physical 2q + 1, physical 2q]
I -> II, X -> XX, Y -> YX, Z -> ZI
```

## Versions and archived reproduction

- Package version `2.0.0` identifies the maintained software release.
- Algorithm version `code_space_recovery_v1.0` identifies the unchanged
  scientific recovery procedure.
- Legacy `module_version="v1.0"` and `core_version` artifact fields remain only
  for reader compatibility; they do not report the installed package version.
- The article's Zenodo deposit is the authoritative, immutable paper snapshot.
- The earlier GitHub package labeled `v1.0.0` has the same scientific and
  numerical implementation. Its differences from the archived source are
  documentation/comments and non-computational diagnostic text, so it is not
  a byte-for-byte archival copy.

Use the Zenodo archive for exact reproduction of published results. Use the
maintained package for new work that benefits from the clarified convention,
API contract, packaging, and tests.
