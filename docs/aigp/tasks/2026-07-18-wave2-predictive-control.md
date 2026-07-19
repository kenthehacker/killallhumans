# Wave 2 predictive-control task manifest

- Task ID: `vq2-wave2-predictive-control`
- Parent: `vq2-wave2-offline-integration`
- State: `committed_integration_pending`
- Objective: implement a pure, deterministic controller from exact
  `RelativeGateStateV1` plus explicit local configuration and phase inputs to
  exact `CommandProposalV1`, reproducing the representable legacy Gate 0
  pixel-rate/thrust behavior and providing only a bounded Gate 1 recenter
  proposal mode.
- Non-goals: no FlightSim process launch or connection; no external network
  access; no preflight, reset, arm/disarm, flight target, approval, transport
  send, gate-passage declaration, or powered action; no runner, frozen-contract,
  policy, trusted-manifest, or shared-handoff changes; no metric-pose
  dependency or official-simulator claim.
- Starting commit: `02f5b6baea4794d3561161110b411b69a0bbab4a`.
- Branch: `wave2-predictive-control`.
- Worktree: `C:\Users\John\aigp-worktrees\wt-vq2-control`.
- Integration owner: `/root`.
- Task owner: `/root/wave1_runtime_timing`.
- Heartbeat date: `2026-07-19`.
- Simulator access: `none`.
- Owned interfaces: the pure predictive-controller module, its direct and
  adversarial tests, and controller-specific documentation/task evidence.
- Excluded interfaces: `scripts/aigp_vq2_run.py`, frozen `/1` contracts,
  promotion policy/manifests, shared handoffs, safety-supervisor approval,
  transport projection, and every simulator or powered path.
- Required adversarial evidence: exact source/authority binding, deterministic
  equality, stale/unhealthy/uncertain/mismatched-state rejection, reviewed
  saturation and thrust diagnostics, exact-zero yaw, no metric-pose dependence,
  Gate 0 regression fixtures, and bounded Gate 1 recentering with no passage
  semantics.
- Required branch gates: directly affected tests, `scripts\dev.cmd test-vq2`,
  `scripts\dev.cmd test-fast`, clean diff/worktree, and no source caches.
- Promotion claim: offline proposal evidence only. `CommandProposalV1` carries
  intent but cannot approve, arm, reset, send, advance a gate, or prove powered
  behavior.

## Accepted offline evidence

- Implementation: `competition/vq2_controller.py`; direct adversarial tests:
  `competition/tests/test_vq2_controller.py`; interface/evidence reference:
  `docs/aigp/vq2_predictive_controller.md`.
- Direct controller plus frozen-contract group: `162 passed` (`109` controller
  nodes plus the existing `53` contract nodes).
- Canonical broad `test-vq2`: `527 passed`.
- Repository `test-fast`: `1,619 passed, 20 skipped, 42 deselected`.
- Exact Gate 0 elapsed, normalized-pixel thrust, pitch-blend, and quaternion
  attitude-loop regression fixtures passed. The claim is limited to the
  representable feature/control law; fitted-aperture state is not relabeled as
  legacy bbox/square-center inference or live-runner equivalence.
- Source/authority binding, deterministic equality, exact-zero yaw, source-less
  withholding, decision-time/state-sequence watermarks, age/health/covariance
  gates, unconditional prediction/delay uncertainty arithmetic,
  rejected-innovation withholding, centered-objective enforcement, saturation
  diagnostics, metric-pose independence, and bounded Gate 1 corridor/timeout
  behavior passed adversarial tests.
- Caller `elapsed_s` was removed. Phase timing now uses an exact safety-expected
  host-monotonic phase start plus an objective evaluation timestamp and minimum
  evaluation watermark. Exact-clock/start mismatch, attempted same-phase start
  renewal, future/regressing evaluation, and objective age above the
  tighten-only 100 ms bound all return source-less exact zero. Transition
  `start == evaluation == proposal` and the exact 0.60-second timeout boundary
  passed.
- Every Gate 0 legacy law, schedule, shared vertical, attitude-gain, and output-
  envelope constant rejects deviations in either direction. Gate 1 corridor
  thresholds likewise reject both smaller and larger values; only Gate 1 gains
  and envelopes plus evidence thresholds remain tightenable.
- Gate 1 permits `DEGRADED` state only with nonzero clipping and all ordinary
  guidance, authority, freshness, and uncertainty gates. Such proposals, and
  any accepted clipped proposal, are explicitly uncertainty-limited. A
  degraded or clipped state inside the corridor returns distinct
  `corridor_unconfirmed_limited` exact zero with `uncertainty.limited=true` and
  cannot return `corridor_reached`.
- `git diff --check`: clean before commit.
- The test-created `.pytest_cache` was resolved to this exact worktree and
  removed; no source `__pycache__`, `.pyc`, or `.pyo` remained.
- Simulator access remained `none`: no FlightSim process or connection,
  external network, preflight, reset, arm/disarm, flight target, approval,
  command send, gate-passage action, or powered operation occurred.

## Residual limitations

- `ControllerAttitudeInput` has no timestamp, clock identity, or source
  correlation, and `CommandProposalV1` cannot bind attitude provenance. No
  shadow, runtime, or powered wiring is eligible until a reviewed IMU timing
  and derotation seam exists.
- Body-rate/thrust clamps bound proposal intent only. The external safety
  supervisor and runtime watchdogs retain actual attitude/rate/stream aborts,
  approval, pacing, single-use enforcement, transport, and cleanup.
- The guidance-to-controller adapter must validate the guidance value's echoed
  authority, complete source correlation, and evaluation time before mapping
  its local objective into `ControllerPhaseInput`. The adapter/supervisor must
  own the stable phase start, set it equal to transition evaluation, preserve
  it within the phase, and enforce its exact tick echo. The pure controller can
  compare but cannot authenticate that ownership, and frozen
  `CommandProposalV1` cannot bind these local phase fields.
- No approved recorded replay was available. Fitted-aperture Gate 0
  non-regression, top-clipped Gate 1 replay, tracker isolation, measured delay,
  actuator response, and any powered recenter behavior remain unproved.
- Initial implementation commit:
  `54940db048c0c688c56c748423e4e1305154f064`.
- Adversarial hardening commit:
  `eb380cb7abeffcec3894487f75c4b4dcf9144c0e`.
- Independent-audit phase/config hardening commit: reported to the integration
  owner after commit.
- The evidence-record commit is reported to the integration owner after commit
  because a Git object cannot contain its own object ID.
