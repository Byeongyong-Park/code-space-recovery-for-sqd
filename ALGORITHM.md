# Code-Space Recovery for SQD

This document summarizes the algorithm implemented in
`src/code_space_recovery/`. It focuses on the conventions and update rules
needed to use the code correctly.

## 1. Scope and data flow

The package provides pair encoding, weighted clustering, invalid-pair
recovery, projected diagonalization, and full-Hamiltonian variance evaluation.

The broader workflow is:

```text
logical Hamiltonian and state
    -> pair encoding and sampling
    -> globally unique encoded bitstrings with weights
    -> weighted BMM hard clusters
    -> pair recovery and logical decoding
    -> logical basis batches
    -> projected diagonalization
    -> global-best reference and carry-over update
    -> repeat
```

The sampled and recovered data live in the encoded space, but projected
diagonalization always uses the **original logical Hamiltonian**.

## 2. Bit order, pair code, and input contracts

### 2.1 Qiskit displayed-bit order

Qiskit displays the highest-index qubit on the left and `q0` on the right. The
default logical-basis convention follows this displayed order: column 0 is the
leftmost logical bit, and the last column corresponds to `q0`. In a Pauli label
of length `n`, qubit `q` is at position `n - 1 - q`.

### 2.2 Pair code and rails

Each logical bit is represented by two adjacent displayed columns:

```text
logical 0 -> 01
logical 1 -> 10

encoded pair = [b, 1 - b]
logical bit  = first column of the pair
```

Pairs `01` and `10` are valid; `00` and `11` are invalid and must be
recovered before decoding.

For logical qubit `q`, the physical rails are:

```text
first rail  = physical qubit 2q + 1
second rail = physical qubit 2q
displayed pair order = [first rail, second rail]
```

These orders must remain consistent across all modules.

### 2.3 Array and weight contracts

- Encoded bitstrings have shape `(N, 2n)`, dtype `np.uint8`, and values in
  `{0, 1}`.
- Logical bases have shape `(D, n)`, dtype `np.uint8`, and values in
  `{0, 1}`.
- BMM input weights have shape `(N,)`, dtype `np.float64`, and are finite and
  nonnegative.
- Recovery accepts one `(bitstrings, weights)` pair per cluster; their total
  weight must be finite and strictly positive.
- Encoded rows must form a globally unique partition. Raw duplicates must be
  merged before clustering.

Weights may be counts or global probabilities. Recovery normalizes them once
over the complete collection of clusters; it never normalizes each cluster
independently.

The projected diagonalizer supports at most 63 logical qubits, with a
configurable default limit of 36.

## 3. Hamiltonian and state encoding

### 3.1 Pauli encoding

The default map, written in displayed pair order, is:

```text
I -> II
X -> XX
Y -> YX
Z -> ZI
```

`encode_sparse_pauliop` applies this map term by term without changing
coefficients. Its default `qiskit_label` mode expands a displayed label from
left to right. Optional simplification combines duplicate labels and removes
small terms.

### 3.2 State encoding

For an `n`-qubit state-preparation circuit with no classical bits,
`encode_state_preparation_circuit_pair_code` constructs a `2n`-qubit circuit:

1. Apply `X` to every second rail `2q`.
2. Compose logical qubit `q` onto first rail `2q + 1`.
3. Apply `CX(first rail -> second rail)` to each pair.

This maps `sum_x alpha_x |x>` to `sum_x alpha_x |enc(x)>`, with
`b -> [b, 1-b]`. The helper does not add measurements.

## 4. Weighted BMM hard clustering

`assign_clusters_bmm` fits a Bernoulli mixture model to globally unique
encoded rows using their global weights.

Weights are normalized only for fitting. Each initialization fits one StepMix
model, and the successful model with the largest finite likelihood lower bound
is selected. Final labels come from `predict(bitstrings)`: every row and its
entire weight belong to one cluster. Returned weights preserve their input
scale rather than being normalized per cluster.

## 5. Self-consistent code-space recovery

`run_code_space_recovery` is the central iterative driver.

### 5.1 Initial references and reassignment

For cluster `k`, the initial reference is its weighted encoded mean, normalized
so each pair sums to one.

The distance between encoded row `x` and reference `r_k` is

\[
d(x,r_k)=\sum_{j=1}^{2n}|x_j-r_{k,j}|.
\]

The first iteration uses the input labels. Later iterations optionally assign
each row to the nearest current reference by L1 distance; exact ties select the
lowest cluster index. Otherwise the initial labels are retained.

### 5.2 Modified-ReLU pair recovery

For distance `d`, low-distance parameter `delta`, and corner `h`, the score is

\[
f(d)=
\begin{cases}
\delta d/h, & d\le h,\\
\delta +(1-\delta)(d-h)/(1-h), & d>h.
\end{cases}
\]

For an invalid pair `b=(b_0,b_1)` and its cluster reference
`r=(r_0,r_1)`, the implementation computes

\[
d_i=|b_i-r_i|,
\qquad
p_{\mathrm{first}}=\frac{f(d_0)}{f(d_0)+f(d_1)}.
\]

The first rail is flipped with probability `p_first`; otherwise the second is
flipped. Every invalid `00` or `11` pair is repaired once, valid pairs are
unchanged, and duplicate recovered rows have their weights summed.

### 5.3 Multiple recovery draws

Recovery is performed at least once per iteration. If the available logical
dimension remains below `max_dim`, additional draws may be performed up to
`max_recovery_draws_per_iteration`.

Every draw starts from the same original samples and weights, not the previous
draw's output; labels and references remain fixed within the iteration. Draws
are merged by logical state, and accumulated weights are divided by the number
of completed draws. More draws add stochastic support without multiplying the
total weight scale.

The loop stops when the actual available dimension reaches `max_dim`, after
accounting for duplicates and all fixed priorities. The default mReLU method
also stops after one draw when no invalid pairs exist.

### 5.4 First-iteration valid-state priority

Samples that were already fully valid receive special treatment only in the
first iteration. After deterministic and carry-over states are accounted for:

- If the eligible initially valid states can fill the remaining capacity,
  batches sample only from those states and exclude recovered-invalid states.
- Otherwise, all eligible initially valid states are forced into every batch,
  and the remaining capacity is sampled from the recovered pool.

### 5.5 Batch construction

Every batch uses this priority:

```text
deterministic basis
-> carry-over basis
-> forced initially valid basis
-> weighted sample from the recovered pool
```

Fixed rows are deduplicated in first-occurrence order and truncated at
`max_dim` if necessary. Otherwise they are removed from the stochastic pool,
which is sampled without replacement. Positive rows use normalized weights;
if too few exist, all are kept and zero-weight rows fill remaining slots
uniformly. A batch may underfill when unique support is insufficient. Batches
within an iteration are independent.

### 5.6 Iteration best, global best, and warm start

Each batch is diagonalized in its logical basis. The iteration best is the
batch with the strictly lowest energy. Exact energy ties select the smallest
batch index.

The global best is replaced only when `iteration_best_energy <
global_best_energy`. There is no comparison tolerance, so a tie does not
replace the existing global best.

When enabled, every batch in an iteration receives the same warm-start
snapshot: the global-best state that existed at the **start** of that
iteration. A result from an earlier batch is not used by later batches in the
same iteration.

### 5.7 Reference and carry-over updates

Both updates use the current global-best wavefunction, not merely the current
iteration best.

For references, the global-best basis is re-encoded and weighted by normalized
`|c_i|^2`. States are assigned to the nearest pre-update references, and each
cluster takes its weighted mean; an empty cluster keeps its old reference.
Carry-over candidates satisfy `|c_i| >= carryover_threshold` and enter the next
iteration before sampling. If necessary, the code leaves room for
deterministic states and keeps the largest amplitudes. Without a global-best
update, the same state continues to determine both updates.

### 5.8 Fixed and adaptive schedules

Fixed stages run a prescribed number of iterations. Adaptive stages stop after
`min_iterations` when the global best has not improved for
`convergence_patience` consecutive iterations, or when `max_iterations` is
reached. The global best, references, and carry-over states persist across
stages, while the patience counter resets at each stage boundary.

## 6. Projected logical diagonalization

For a unique logical basis `B`, the diagonalizer constructs `H_B = P_B H P_B`,
where `H` is the original logical Pauli Hamiltonian. A matrix element is kept
only when the Pauli action connects two states that both belong to `B`, subject
to the configured coefficient and matrix-element cutoffs.

`ProjectedPauliCSRPRIMMEDiagonalizer` stores this projection as a sparse CSR
matrix and calls PRIMME `eigsh` with `k=1` and `which="SA"` for the lowest
Ritz pair. A supplied global-best warm start is restricted to its overlap with
the new basis; if there is no overlap, a seeded random initial vector is used.

The residual `||H_B c - E c|| / max(1, |E|)` measures convergence inside the
projected subspace; it is not the full-Hamiltonian variance.

The diagonalizer compiles the Hamiltonian given at construction. It should be
constructed from the same original logical Hamiltonian passed to the recovery
driver.

## 7. Full-Hamiltonian energy and variance

For

\[
|\psi\rangle=\sum_i c_i|x_i\rangle,
\]

`compute_full_hamiltonian_variance` applies the full logical Hamiltonian and
retains support both inside and outside the SQD basis. It computes

\[
E=\operatorname{Re}\langle\psi|H|\psi\rangle,
\qquad
\langle H^2\rangle=\|H|\psi\rangle\|^2,
\qquad
\operatorname{Var}(H)=\max(0,\langle H^2\rangle-E^2).
\]

The result separates the squared norm of `H|psi>` inside and outside the input
basis.
A nonzero Hamiltonian coefficient cutoff discards small Pauli terms, while an
amplitude cutoff discards small combined amplitudes in `H|psi>`. Either can
make the energy, `H^2`, and variance approximate.

## 8. Parallel execution and reproducibility

Batch solves can run sequentially, with threads, or with Loky processes. The
sequential and threaded paths use the run-level generator; Loky uses
deterministically seeded worker-local generators for individual batches.

Reproduction requires the same inputs, seed, backend, batch count, recovery
draw limit, and numerical settings. Within the same software and numerical
environment, the implementation keeps results in batch-index order and uses
deterministic tie rules. Different libraries, hardware, or thread behavior may
still introduce small floating-point differences.

## 9. Result and storage scope

`CodeSpaceRecoveryResult` retains the final global-best energy, logical basis,
normalized coefficients, and reference vectors. It also records compact
per-iteration histories for global and iteration bests, batch energies and
dimensions, recovery draws, timing, stage termination, and the effective run
configuration.

The result does **not** retain every batch basis or every aggregated recovered
pool. Only the final global-best basis and coefficients are kept as state
arrays. Experiments that need larger intermediate objects must save them
explicitly outside the core driver.
