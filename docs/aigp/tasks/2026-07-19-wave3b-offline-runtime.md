# Wave 3B generated offline scheduler composition

- Task ID: `vq2-wave3b-generated-offline-runtime`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `post_merge_verified`
- Objective: compose the already-decoded generated image path, exact local IMU
  provenance/correlation, unchanged raw-camera relative estimator, Wave 3
  adapter, 50 Hz scheduler, and immutable latency evidence while terminating
  at a quarantined `CommandProposalV1`.

## Ownership

- `competition/vq2_wave3_offline_runtime.py`
- `competition/tests/test_vq2_wave3_offline_runtime.py`
- `docs/aigp/vq2_wave3_offline_runtime.md`
- this task record

No `/1` contract, runner, supervisor, transport, simulator, reset, arm,
cleanup, promotion scheduler, trusted manifest, or shared handoff is owned by
the implementation tranche. Promotion metadata and handoffs remain integration
owner work after the behavior is committed and independently reviewed.

## Required invariants

- Inputs are exact generated `VQ2VisionSnapshot` values, immutable timed IMU
  samples, caller-supplied host occurrence times, safety guidance, and explicit
  calibration/model/uncertainty configuration.
- The mutable legacy `CameraFrame` nested inside a snapshot is revalidated for
  exact type and metadata, simulator timestamp binding, immutable contiguous
  `uint8` BGR storage, and a finite legacy freshness stamp. That independent
  freshness clock is not relabeled as the `/1` packet-occurrence clock.
- One runtime owns the latest-frame cursor, fixed-rate scheduler, IMU
  provenance/history, raw relative estimator, Wave 3 adapter memory, proposal
  sequence, and cumulative trace.
- Session/reset, host clock, camera stream/generation/frame, IMU
  stream/generation/sequence, safety authority, and controller tick lineage
  match exactly or fail before state advances.
- Perception runs at most once per distinct publication. A repeated-frame tick
  supplies no correlated update, performs no coast relabeling, and returns a
  source-less exact-zero proposal at the current reviewed boundary.
- Ticks are never due faster than 20 ms. Expired and planned-overrun ticks are
  skipped before cursor/estimator/adapter advancement, and stalls never create
  catch-up bursts.
- Capture and target attitudes are chosen deterministically only from accepted
  samples available by the decision time and within the explicit derotation
  bounds.
- Rotation-corrected bearing remains standalone evidence. The relative state,
  guidance, and proposal retain the unchanged raw-camera measurement basis.
- Trace assembly orders exact host-occurrence facts rather than append order.
  A `GYRO_SAMPLE` records an IMU occurrence only; it does not claim command
  causality, actuator response, or detection-through-gyro response latency.
- An exported result structurally binds its current scheduler/controller
  lifecycle, lease, proposal, diagnostic reason, distinct camera/perception
  stages, and selected IMU facts. Preview validation occurs before commit;
  detached, truncated, contradictory, send-bearing, or actuator-bearing traces
  are rejected.
- No supervisor approval, command projection/send, actuator response,
  transport, reset, arm/disarm, cleanup, network I/O, simulator, or powered
  value is constructed or invoked.

## Evidence boundary

This tranche is deterministic generated/offline mechanism evidence. It is not
receiver/reassembly evidence, recorded replay, shadow/runtime evidence,
production per-sample arrival capture, calibrated clock/delay evidence, a
supervisor-verifiable `/2` provenance envelope, or powered FlightSim evidence.
Operational T0-T4 remains blocked by the approved corpus, final processor, and
administrator-owned isolation wrapper. Applying the corrected ray to filter
state remains blocked on a separately reviewed stable-frame or explicitly
time-aligned estimator design.

## Closeout record

- Behavioral implementation:
  `8eab146e3a9a7a1a1b28070d3e0234adff900595`.
- Main reconciliation merge:
  `7904fbadbc4b220b81afb846a69b15a7b30ef4bb`.
- Promotion-policy and trusted-manifest closeout:
  `28b7d782404d6b825cebae3b65a8443d756be234`.
- Focused runtime tests: 38 passed. Coupled runtime, Wave 3 adapter,
  provenance, derotation, and relative-estimator matrix: 181 passed.
- Independent compatibility/adversarial matrix: 199 passed. Review explicitly
  cleared the production snapshot seam, scheduler/IMU chronology,
  transactionality, result/trace binding, raw-versus-corrected bearing split,
  and authority isolation with no remaining tranche-local blocker.
- Canonical and isolated-manifest VQ2 policy: exactly 910 passed, including
  the post-merge main run.
- `test-fast` and `test-unit`: 2,005 passed, 20 skipped, and 42 deselected
  each.
- Promotion-boundary `test-full-non-live`: 2,046 passed and 21 skipped.
  Skipped optional coverage is not positive evidence.
- Strict trusted manifest: 127 files, semantic identity
  `cdd0db402b6f1c8bb0c90c1b8d445ca64741d3bfc3aa03a78c3fe4d73c8dcce2`,
  file SHA-256
  `e270a194031d463accfb50b28bd3296eb672004d1c41241fab3cb368bab1640a`,
  and exact VQ2 policy SHA-256
  `64cfefc083a52fc925ad98c2e3a99e8f6eefcaebb0f4243d214c1e87729a864c`.
  The reviewed trust delta added only the new runtime test and changed only
  the policy hash; no file was removed and the trust root did not expand.
- Main fast-forwarded to the promotion commit, post-merge `test-vq2` passed all
  910 tests, and tracked Git status was empty before this documentation
  closeout.
- Simulator access remained `none`. No FlightSim launch or connection,
  preflight, external network access, reset, arm/disarm, target, transport,
  shadow selection, or powered command occurred.

The next generated/offline tranche is a separately reviewed, default-off,
single-tick IMU-correlated coast lease. Stable-frame corrected-ray application,
recorded replay, production timing/calibration, supervisor provenance, and all
powered work remain outside this completed task.
