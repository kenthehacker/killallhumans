# Wave 2 mapless-guidance task manifest

- Task ID: `vq2-wave2-mapless-guidance`.
- Parent: `vq2-wave2-offline-integration`.
- State: `integration_pending`.
- Objective: implement deterministic, mapless local guidance/state logic that
  consumes an exact frozen `RelativeGateStateV1` plus caller-supplied safety
  authority/race/phase-time state and emits a local objective without command
  or transport authority.
- Starting candidate commit:
  `02f5b6baea4794d3561161110b411b69a0bbab4a`.
- Branch: `wave2-mapless-guidance`.
- Worktree: `C:\Users\John\aigp-worktrees\wt-vq2-planning`.
- Integration owner: `/root`.
- Lease owner: `/root/wave1_gate_geometry`.
- Heartbeat date: `2026-07-18`.
- Simulator access: `none`.
- Owned interfaces: `planning/vq2_guidance.py`, directly related planning
  tests, a deterministic synthetic scenario evaluator, and this task record.
- Frozen inputs: `GateAuthorityEpochV1` and `RelativeGateStateV1` from
  `competition/vq2_contracts.py`; this task does not edit their fields, codecs,
  validation, or meanings.
- Excluded interfaces: `CommandProposalV1`, controller implementation, frozen
  contracts, runtime/runner/safety-supervisor code, transports, test policy,
  trusted manifests, shared handoffs, simulator processes, and private replay
  or capture inputs.
- Authority boundary: visual state may echo but never create or advance safety
  authority, expected gate index, reset epoch, or gate epoch. Only a forward
  same-session caller-supplied authority transition may change gate phase. A
  safety-issued phase start on the evaluation host clock must remain exact for
  a phase and restart exactly at an accepted phase entry. A regressing, stale,
  cross-session, mismatched, shadow-only, or otherwise ambiguous input fails
  closed.
- Required behavior: exactly one authoritative active gate; explicit
  active/shadow isolation; conservative acquire, align, approach, commit,
  confirmation, and post-credit reacquire semantics; uncertainty-aware corridor
  eligibility; no visual passage declaration.
- Required evidence: adversarial affected tests, a deterministic generated
  synthetic scenario demonstrating Gate 0 phase non-regression and top-clipped
  Gate 1 recenter objective withholding/eligibility, canonical `test-vq2`,
  `test-fast`, clean committed worktree, and exact no-simulator provenance.
- Evidence boundary: generated unit/synthetic results are not replay,
  official-simulator, powered, passage, or race-readiness evidence.

## Implemented candidate

- `planning/vq2_guidance.py` provides planning-owned immutable safety input,
  memory, source-correlation, decision, configuration, and transition values.
  None is a wire schema or command-authority type.
- `step_vq2_guidance` is a pure transition. Initialization requires safety-
  supplied `ACQUIRE` plus `NOT_UNDERWAY`. Same-gate phases advance only through
  the adjacent sequence `ACQUIRE -> ALIGN -> APPROACH -> COMMIT ->
  CONFIRMATION` on a strictly newer same-session authority snapshot. Exact
  next-gate credit is required for `CONFIRMATION -> POST_CREDIT_REACQUIRE`, and
  the next forward snapshot may then select `ACQUIRE`.
- Cross-session, reset/gate/index regression, phase/race-state regression or
  jump, phase-start rewind/renewal, evaluation-time regression, authority
  mismatch, stale/revisited source, active-track transfer, retired-track reuse,
  and invalid shadow association all fail closed without replacing the last
  accepted memory.
- The safety input carries an exact planning evaluation time on the authority
  camera host clock plus a caller-supplied phase start on that same clock.
  Initial, reset, gate-credit, and every accepted phase transition require the
  start to equal evaluation time; same-phase updates, including race-only
  transitions, must preserve it exactly. The decision echoes the stable start
  for downstream dwell checks. Active and shadow inputs reject future
  publication or decision time, decision age above 100 ms, measurement age
  plus uncertainty above 150 ms, measurement uncertainty above 50 ms, and
  prediction lead plus delay uncertainty above 100 ms. These local checks do
  not replace controller, supervisor, runtime-stream, or transport freshness
  checks.
- Eligibility configuration is tightening-only against immutable offline hard
  defaults: centered target, at least three-sigma uncertainty, no wider
  corridor/rate limits, no weaker commit scale/expansion minima, preserved
  phase ordering, and no looser timing bounds. A high finite sigma ceiling
  prevents diagnostic arithmetic overflow.
- Only `ALIGN`, `APPROACH`, and `COMMIT` can produce an eligible local objective.
  Acquire, confirmation, post-credit reacquire, countdown, finish, and abort
  remain motion-withheld. `COMMIT` is planning eligibility only: it neither
  declares passage nor reaches a controller or transport in this branch.
- Shadows are identity/timing validated but never promoted and never contribute
  bearing, rate, scale, or uncertainty to an active objective. A prior gate's
  active tracker is retired across authoritative credit and cannot seed the
  next active gate.

## Accepted offline evidence

- Direct guidance plus generated-scenario target:
  `80 passed in 0.20s`.
- Canonical branch `test-vq2`: exactly `418 passed in 6.20s`. The branch did
  not edit the serialized policy; integration owns adding
  `planning/tests/test_vq2_guidance.py`, updating the exact count, and reviewing
  the trusted manifest.
- `test-fast`: `1,590 passed, 20 skipped, 42 deselected in 67.99s`.
- The 13-step generated image-space unit scenario passed Gate 0 visual-phase
  non-regression, same-snapshot phase rejection, stable phase-start echo,
  forward-authority phase-start renewal rejection, and forward phase
  acceptance, plus Gate 1 shadow isolation, high-uncertainty withholding, and
  lower-uncertainty top-clipped recenter eligibility. Its deterministic
  SHA-256 is
  `510eeaa21a4b476e389379394786c781ca2369f1401aff43a1596dfdb634d5ec`.
- The scenario scope is explicitly
  `deterministic_generated_image_space_unit_scenario_nonpowered;not_replay_not_simulator_not_passage_evidence`.
- `git diff --check`: clean before commit.
- Simulator access remained `none`: no external network, preflight, process
  launch, reset, arm/disarm, target, command send, or powered action occurred.

## Residual blockers

- No runtime, runner, controller, safety-supervisor, or transport integration
  exists. Integration owns any narrow checked adapter to the pure controller;
  that adapter must source the phase start from trusted safety state and use
  the echoed stable start for controller dwell timing. This branch does not
  establish that the local safety-input producer is trusted.
- The acquisition corridors, timing caps, and commit diagnostics are bounded
  offline planning defaults, not calibrated replay, actuator, or live-flight
  limits. Recorded replay, tracker-isolation acceptance, IMU/delay evidence,
  and separately authorized simulator stages remain required.
- Bounded degraded-state Gate 1 recentering still needs the controller's own
  covariance/time checks plus supervisor review. `COMMIT` has no Wave 2
  controller mapping and must remain unreachable from transport.
- The generated scenario has no vehicle dynamics and cannot support replay,
  passage, collision-clearance, official-simulator, powered, or race-readiness
  claims.
- Implementation commit:
  `b758684ca89bcddf70ae4e630061e64c3f04b215`. The final evidence-record
  commit is this follow-up changeset and is reported directly to the
  integration owner because a commit cannot contain its own immutable ID.
