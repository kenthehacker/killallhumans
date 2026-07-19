# Wave 1 relative-state estimation task manifest

- Task ID: `vq2-wave1-relative-estimation`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `integration_pending`
- Objective: implement the highest-value fully offline feature-space filter,
  prediction, innovation gating, dropout/coasting, and estimator-health path
  required before M4. The implementation consumes frozen
  `GateObservationV1` values and emits frozen `RelativeGateStateV1` values.
- Starting commit: `3de33c3a568bc86638d9d7ac4dac6124f1e15397`
- Branch: `wave1-relative-estimation`
- Worktree: `C:\Users\John\aigp-worktrees\wt-relative-estimation`
- Integration owner: `/root`
- Lease owner: `/root/contract_doc_sync`
- Heartbeat date: `2026-07-18`
- Simulator access: `none`
- Artifact root:
  `C:\Users\John\aigp-worktrees\artifacts\wave1-relative-estimation`
- Cache root:
  `C:\Users\John\aigp-worktrees\.artifact-cache\wave1-relative-estimation`
- Owned interfaces: estimator/filter modules, their directly related tests,
  and estimator documentation.
- Excluded interfaces: frozen `/1` contract definitions, promotion/trust
  manifests, detector internals, controller/planning, runner/safety code, and
  every simulator or transport path.
- Safety boundary: no network, simulator, preflight, reset, arm, flight target,
  command send, or powered action. Perception never declares passage and the
  estimator may only echo safety-issued gate authority.
- Required adversarial evidence: irregular timing, bounded prediction horizon,
  innovation rejection, covariance positive-semidefiniteness, clipping-aware
  uncertainty, distinct-frame single update, stale/replayed source rejection,
  dropout/coasting and loss, reset/gate authority transitions, and deterministic
  replay-equivalent results.
- Required direct tests: the new estimator tests plus frozen VQ2 contract tests.
- Required candidate gate: `scripts\dev.cmd test-vq2`.
- Acceptance: deterministic updates on distinct observations; honest feature
  covariance and health; bounded dropout prediction; no stale-frame reuse or
  authority leakage; no metric-pose invention; no `/1` wire change; committed
  green branch.
- Candidate evidence:
  - estimator adversarial suite: `14 passed`;
  - frozen VQ2 contract suite: `53 passed`;
  - dedicated non-live VQ2 gate: `370 passed`;
  - repository `test-fast`: `1462 passed, 20 skipped, 42 deselected`;
  - isolated legacy `test-unit`: `1462 passed, 20 skipped, 42 deselected`;
  - `git diff --check`: clean.
- Final commit: `4a9833abbbcdd1d88b1f2e4d378fcf14c39de4d2`.
- Result/failure provenance: all evidence was produced offline in the named
  worktree with `AIGP_PYTHON` bound to the repository development environment.
  No simulator, preflight, network, reset, arm, target, command-send, or
  powered action occurred. No metric pose or official-simulator claim is made.
