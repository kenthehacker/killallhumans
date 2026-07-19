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
- Lease owner: `/root/wave2_guidance_review`.
- Heartbeat date: `2026-07-19`.
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
  authority, expected gate index, reset epoch, gate epoch, race state, or
  phase. Safety transitions are validated and stored independently of visual
  state: a rejected safety transition preserves the entire prior memory,
  while a valid safety transition remains accepted if the accompanying visual
  batch is rejected. In the latter case the visual batch is atomic and cannot
  alter retained active/shadow ownership or chronology. A safety-issued phase
  start on the evaluation host clock must remain exact for a phase and restart
  exactly at an accepted phase entry.
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
- `step_vq2_guidance` is a pure transition. Fresh memory requires gate epoch
  and expected gate index zero plus safety-supplied `ACQUIRE` and
  `NOT_UNDERWAY`; a nonzero reset epoch remains legal. Countdown cannot advance
  the phase. `GO` is a distinct forward transition that retains `ACQUIRE`,
  after which same-gate phases advance only while both snapshots are
  `UNDERWAY` through `ACQUIRE -> ALIGN -> APPROACH -> COMMIT -> CONFIRMATION`.
  Exact next-gate credit is required for `CONFIRMATION ->
  POST_CREDIT_REACQUIRE`, and the next forward snapshot may then select
  `ACQUIRE`. Finish or abort retains and freezes the current phase.
- Cross-session, reset/gate/index regression, phase/race-state regression or
  jump, phase-start rewind/renewal, evaluation-time regression, authority
  mismatch, stale/revisited source, track owner/role transfer, retired active-
  track reuse, and invalid shadow association all fail closed. Safety and
  visual memory have separate acceptance: rejected safety replaces nothing;
  rejected visual input preserves any independently accepted safety progress
  but replaces no visual history.
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
  remain motion-withheld. `ALIGN` accepts a degraded state only when its edge-
  clipping marker explains that degradation; degraded but unclipped state is
  recorded for chronology and withheld from motion. `APPROACH` and `COMMIT`
  require healthy state. `COMMIT` is planning eligibility only: it neither
  declares passage nor reaches a controller or transport in this branch.
- Active and shadow inputs share gate-scoped source ownership and per-tracker
  chronology. A source cannot move between tracker IDs or roles, a tracker
  cannot switch roles, duplicate tracker/source ownership inside one call is
  rejected before staging, and a late invalid member rolls back the complete
  visual batch. Shadows never contribute bearing, rate, scale, or uncertainty
  to an active objective. Gate credit clears current gate histories and retires
  the prior active tracker ID; reset clears/rekeys all guidance visual history.

## Accepted offline evidence

- Direct guidance plus generated-scenario target:
  `106 passed in 0.24s`.
- Frozen VQ2 contract plus direct guidance target:
  `159 passed in 0.44s`.
- Canonical branch `test-vq2`: exactly `418 passed in 5.77s`. The branch did
  not edit the serialized policy; integration owns adding
  `planning/tests/test_vq2_guidance.py`, updating the exact count, and reviewing
  the trusted manifest.
- `test-fast`: `1,616 passed, 20 skipped, 42 deselected in 67.18s`.
- The 16-step generated image-space unit scenario passed gate-zero-only fresh
  initialization, countdown phase rejection, Gate 0 visual-phase
  non-regression, same-snapshot phase rejection, stable phase-start echo,
  forward-authority phase-start renewal rejection, and forward phase
  acceptance. It also passed Gate 1 shadow isolation, shadow-to-active transfer
  rejection, high-uncertainty withholding, and lower-uncertainty top-clipped
  recenter eligibility. Its deterministic outcome-summary SHA-256 is
  `13b5a7b3120826780e529b0d183a4298ba562937d47edf15773092471d4516df`.
  The digest binds only that encoded outcome summary; it is not a hash of full
  inputs, configuration provenance, implementation identity, or environment.
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
- The Wave 1 relative estimator retains its constructor tracker ID while gate
  reset restarts its sequence. Integration must therefore mint or instantiate
  a distinct gate-scoped active tracker ID after exact gate credit; simply
  resetting a single estimator instance and reusing the prior active ID is
  intentionally rejected. Gate 1 helpers in this branch demonstrate the legal
  credit/reacquire path with a distinct tracker identity only.
- The acquisition corridors, timing caps, and commit diagnostics are bounded
  offline planning defaults, not calibrated replay, actuator, or live-flight
  limits. Recorded replay, tracker-isolation acceptance, IMU/delay evidence,
  and separately authorized simulator stages remain required.
- Bounded edge-clipped degraded-state Gate 1 recentering still needs the
  controller's own covariance/time checks plus supervisor review. Degraded
  unclipped state is motion-withheld. `COMMIT` has no Wave 2 controller mapping
  and must remain unreachable from transport.
- The generated scenario has no vehicle dynamics and cannot support replay,
  passage, collision-clearance, official-simulator, powered, or race-readiness
  claims.
- Initial implementation commit:
  `b758684ca89bcddf70ae4e630061e64c3f04b215`.
- Ownership/lifecycle hardening commit:
  `72d89c33429f31d831a134b3183108df8d523445`. The final evidence-record
  commit is this follow-up changeset and is reported directly to the
  integration owner because a commit cannot contain its own immutable ID.
