# Wave 2 offline system-identification task manifest

- Task ID: `vq2-wave2-system-id`
- Parent: `vq2-wave2-offline-integration`
- State: `candidate_complete`
- Objective: implement deterministic offline tooling that validates exact
  timestamped rate-command and gyro samples on one host-monotonic clock, fits
  bounded per-axis delay plus first-order rate dynamics with bias, reports
  uncertainty and fit diagnostics, and validates only on samples excluded from
  training.
- Starting commit: `02f5b6baea4794d3561161110b411b69a0bbab4a`
- Branch: `wave2-system-id`
- Worktree: `C:\Users\John\aigp-worktrees\wt-system-id`
- Integration owner: `/root`
- Lease owner: `/root/contract_doc_sync`
- Heartbeat date: `2026-07-18`
- Simulator access: `none`
- Owned interfaces: `estimation/vq2_system_id.py`, its directly related tests,
  and this task record.
- Excluded interfaces: frozen `/1` contracts, runner, supervisor, controller,
  scheduler, transport, sockets, policy/count, trusted manifest, shared
  handoffs, simulator tooling, captures, and every command-send path.
- Safety boundary: bounded experiment definitions are inert immutable data;
  they cannot arm, reset, connect, send, schedule, approve, supervise, or
  create a transport command. No preflight, FlightSim process, external
  network, powered action, or private capture ingestion is authorized.
- Model scope: simple independent body-rate axes only, with a discrete
  constant delay followed by first-order lag, gain, and gyro bias. Synthetic
  parameter recovery is unit evidence, never measured plant truth.
- Fail-closed conditions: mixed clocks, duplicate or regressing sequence/time,
  non-finite or missing samples, saturation in the fitting partition,
  inadequate excitation, inadequate time span, unidentifiable delay/lag,
  non-physical fits, empty held-out data, or any training/validation overlap.
- Required adversarial evidence: bounded delay/gain/lag/bias recovery,
  irregular sampling, saturation and missing-data rejection, clock/sequence
  errors, deterministic results, finite conservative uncertainty, held-out
  scoring, leakage rejection, and weak-excitation rejection.
- Required gates: direct system-ID tests, `scripts\dev.cmd test-vq2`, and
  `scripts\dev.cmd test-fast`.
- Acceptance: exact offline-only APIs; deterministic fits and diagnostics;
  explicit uncertainty; honest held-out validation; no frozen-wire or powered
  wiring change; committed green worktree with no source caches.
- Implementation: `estimation/vq2_system_id.py` provides local immutable
  offline-analysis contracts, strict single-clock/sequence/time validation,
  inert bounded roll/pitch experiment definitions, and a deterministic
  training-only grid fit for delay plus first-order lag, gain, and gyro bias.
  It scores one later chronological partition as a free run. The first
  validation gyro is only an initializer and is excluded from scoring; later
  validation labels cannot change parameters or training profile intervals.
  Held-out residuals may only conservatively inflate the conditional gain/bias
  covariance.
- Candidate evidence:
  - direct adversarial system-ID suite: `17 passed`;
  - dedicated non-live `test-vq2`: `435 passed`;
  - repository `test-fast`: `1527 passed, 20 skipped, 42 deselected`;
  - `git diff --check`: clean before commit.
- Evidence limits: all recovery evidence is deterministic synthetic math, not
  measured FlightSim or vehicle truth. The model is an independent-axis FOPDT
  approximation; it does not identify cross-axis coupling, thrust response,
  nonlinear or saturated behavior, changing operating points, or yaw. Default
  delay and lag estimates are finite-grid quantities, and reported uncertainty
  is conditional/profile evidence from one chronological split rather than a
  guarantee of plant coverage. A matching host-clock token is a required data
  assertion, not a clock-calibration mechanism.
- Result/failure provenance: all accepted evidence was produced offline in the
  named worktree with `AIGP_PYTHON` bound to the repository development
  environment. No simulator, preflight, external network, capture, reset, arm,
  target, transport, or powered action occurred. Executing even the inert
  experiment definitions would require a separately reviewed and explicitly
  authorized powered workflow outside this module.
- Candidate commit: reported to the integration owner after commit because a
  commit cannot embed its own final hash.
