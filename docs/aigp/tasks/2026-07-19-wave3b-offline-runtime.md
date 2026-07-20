# Wave 3B generated offline scheduler composition

- Task ID: `vq2-wave3b-generated-offline-runtime`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `active`
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

Final commits, exact affected/canonical/full counts, trusted-manifest identity,
independent findings, and residuals are pending integration review.
