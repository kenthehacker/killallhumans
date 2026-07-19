# Wave 1 clipped gate geometry task manifest

- Task ID: `vq2-wave1-gate-geometry`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `tested`
- Objective: add a deterministic, fully offline VQ2 inner-aperture fitter that
  treats frame-clipped edges as censored, infers a conservative square-aperture
  model only when the visible support is sufficient, and publishes honest
  confidence, residual, and covariance data through `GateObservationV1`.
- Non-goals: no FlightSim connection; no preflight; no network access; no
  reset, arm/disarm, flight target, or powered stage; no metric pose or distance
  estimate; no tracker ownership or active/shadow association; no runner,
  controller, safety-supervisor, estimator, promotion, or frozen `/1` schema
  change; no new/private full capture.
- Starting main commit: `3de33c3a568bc86638d9d7ac4dac6124f1e15397`
- Branch: `wave1-gate-geometry`
- Worktree: `C:\Users\John\aigp-worktrees\wt-gate-geometry`
- Integration owner: `/root`
- Lease owner: `/root/wave1_gate_geometry`
- Heartbeat date: `2026-07-18`
- Simulator access: `none`
- Artifact root: `C:\Users\John\aigp-worktrees\artifacts\wave1-gate-geometry`
- Cache root: `C:\Users\John\aigp-worktrees\.artifact-cache`
- Owned files: VQ2 gate-geometry implementation under `gate_detection/src`,
  its explicit observation compatibility adapter, directly related tests, and
  gate-geometry documentation.
- Excluded integration-hot files: `competition/vq2_contracts.py`, runner and
  safety code, estimator modules, promotion/evidence schemas, test policy, and
  trusted-file manifests.
- Dependencies: frozen `GateObservationV1` generation `/1`; build-3385 VQ2 HSV
  preset; existing OpenCV and NumPy runtime dependencies; approved synthetic
  fixtures already permitted by the repository policy.
- Required direct tests: VQ2 geometry and detector/adapter tests, including
  top/bottom/left/right clipping, corner ordering and convexity, visible versus
  inferred semantics, uncertainty growth, degenerate support, determinism, and
  exact legacy-adapter compatibility.
- Required candidate gate: `scripts\dev.cmd test-vq2`.
- Acceptance: a fully visible synthetic gate yields an ordered convex inner
  aperture with measured visible edges/corners; a singly clipped aperture is
  fitted from visible support with the clipped edge/corners marked inferred and
  strictly larger uncertainty; insufficient or degenerate support cannot invent
  precise geometry; repeated identical input is exactly deterministic; existing
  Gate 0 bbox detection and the legacy bbox-to-observation path do not regress.
- Implementation: `gate_detection/src/vq2_geometry.py` adds deterministic HSV
  mask extraction, connected-support and competing-gap rejection, visible inner
  line fitting, one-side censored image-square inference, clipped visible
  segments, and conservative diagonal feature covariance. The explicit
  `gate_detection_with_aperture_to_observation_v1` adapter publishes successful
  fits through the frozen `/1` fields or an honest bbox-only fallback. The
  original bbox adapter remains unchanged and the runtime does not opt in
  automatically.
- Accepted offline evidence:
  - direct frozen-contract plus geometry group: `79 passed`;
  - canonical `test-vq2`: `377 passed` (+21 reviewed test nodes from the
    356-test starting policy; serialized policy update is reserved to the
    integration owner);
  - `test-fast`: `1,469 passed, 20 skipped, 42 deselected`;
  - `git diff --check`: clean before commit.
- Evidence limits and residual blockers: tests use generated synthetic
  VQ2-colour frames and make no official-simulator, powered, metric-pose, or
  replay claim. Stable center/uncertainty on the recorded top-clipped Gate 1 and
  Gate 0 replay non-regression remain unproved until an approved replay input
  and final processor are available. Crossing-residue isolation and
  active/shadow ownership remain tracker work outside this task.
- Result/failure provenance: all accepted evidence was produced offline in the
  named worktree with the repository development environment. No capture,
  FlightSim process, preflight, network connection, reset, arm/disarm, flight
  target, or powered command was used. An initial patch-placement error made the
  legacy adapter return `None`; it was corrected before acceptance, after which
  the original 53 contract tests passed and remained green.
- Final commit hash: pending.
