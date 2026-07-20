# AIGP VQ2 execution-plan handoff

**Date:** 2026-07-18
**Current offline update:** 2026-07-20
**Continuation handoff:** docs/aigp/2026-07-20-vq2-development-continuation-handoff.md
**Target:** DCL FlightSim build 3385, VQ2 Training and Qualification
**Repository:** C:\Users\John\killallhumans
**Integrated foundation baseline:** b9382da162c1c1e2984288ad7f3cfa7e5a1b11f8
**Integrated Wave 1A implementation:** a6782cd9dcc34aee94e0f064021399985e0f6839
**Integrated Wave 1 offline record:** 1cf17ea5f4e0a330bee89b0128d30b13657899a2
**Integrated Wave 2 offline implementation:** 8176cbac20ff16bfa4b8c24764596d9366fe98cf
**Integrated Wave 3 control-plane increment:** ab62cde9464442e4b448f293ba8efd31ad601c27
**Integrated Wave 3 offline IMU provenance:** ecaa794aeaed87a169b7b87b284d1440f1768a28
**Integrated Wave 3B generated offline runtime:** 28b7d782404d6b825cebae3b65a8443d756be234
**Integrated Wave 3C correlated coast:** 168220ba7060d07743335d0e9c56bcd2d05d669d
**Integrated Wave 3D stable-reference prototype:** 46df0adee76070e10509fa5e807b986a9469c68e
**Integrated Wave 3E local-differential reducer:** 16dd5c84995cafb5158e277d730e670557ba69f2
**Historical pre-foundation baseline:** c7c37c047039bcac055d77c57a234effe36f73e1
**Plan state:** M0 and Wave 1A are complete. The three Wave 1 offline
foundations, Wave 2 controller/system-ID/guidance tranche, Wave 3 control-plane
dogfood, Wave 3 local IMU provenance/rotation-only tranche, Wave 3B generated
scheduler composition, Wave 3C proof-bound one-tick coast, and Wave 3D
standalone stable-reference local-feature transform, and Wave 3E standalone
rectified-homography local-differential reducer are integrated on main and
post-merge verified. M1/M2/M4 runtime acceptance remains incomplete; the next
high-value state-estimation work is blocked on real calibration, a reviewed
dense homography-covariance producer, and approved replay evidence. No new
FlightSim evidence was collected.

## Purpose

This document turns the VQ2 research into a resumable implementation program.
It is intentionally detailed about boundaries, evidence, safety, and integration,
while leaving room to choose algorithms from measured results.

Read these first:

1. 2026-07-18-vq2-handoff.md for the authoritative build-3385 flight state.
2. ../2026-07-18-development-cycle-handoff.md for the testing/tooling program.
3. durable_improvement_loop.md for replay, promotion, ledger, and campaign rules.
4. decisions.md for the current live safety decisions.

## Program outcome

The immediate objective is a conservative valid full VQ2 lap using the available
camera, HIGHRES_IMU, race status, actuator output, and collision telemetry.
Speed optimization follows only after completion is repeatable.

The recommended progression is now:

1. Freeze the shared timing, observation, estimator, command, and evidence
   contracts, while dogfooding the committed evaluation foundation on a clean
   candidate.
2. Establish simulator timing and closed-loop plant truth.
3. Recover robust gate geometry, including clipped inner apertures.
4. Predict relative gate state to command-effect time.
5. Recenter and pass Gate 1 through separately bounded stages.
6. Generalize the gate state machine to a conservative full lap.
7. Establish repeatability.
8. Optimize speed, crossing points, lookahead, and course-specific behavior.
9. Evaluate advanced state estimation and control only where evidence supports it.

## Non-negotiable operating contract

### Main stays clean

From the integrated foundation baseline onward:

- No agent edits files directly in the main worktree.
- Main is used only as a clean integration and release reference.
- Every implementation task uses its own branch and worktree.
- A task is not eligible for integration until all intended changes are committed.
- A dirty candidate is rejected rather than silently snapshotted or merged.
- Generated captures, caches, ledgers, profiles, and training artifacts stay outside
  Git or in explicitly ignored roots.
- Main must report an empty git status after every merge.

### Safety authority stays separate

The VQ2 safety supervisor remains the only component allowed to own:

- reset-epoch proof;
- countdown and GO proof;
- arm/disarm freshness;
- stream watchdogs;
- command envelopes;
- collision handling;
- expected gate-index transitions;
- exact-zero crossing confirmation;
- cleanup proof;
- stage timeout and latched aborts.

Perception never declares passage. Planning never sends commands. Experimental
control never arms, resets, or performs cleanup. Race status remains the sole
VQ2 gate-pass authority, and crossing confirmation continues to send exact zero
rate and zero thrust unless a later safety decision explicitly replaces it.

### FlightSim is an exclusive resource

- Offline work may run concurrently.
- Only one process or agent may own the live simulator lease.
- Only the integration worktree may run powered FlightSim stages.
- Passive probes must still coordinate through the simulator lease so they do not
  compete for UDP ports or invalidate a powered trial.
- Every new powered stage requires explicit authorization and bounded cleanup.

## Current repository snapshot

The development-cycle work is much farther along than its original handoff:

- strict pytest markers and timeouts are implemented;
- the two production-length VQ1 tests are bounded;
- a VQ2-first Windows task surface exists;
- runtime, development, legacy, and training dependency profiles exist;
- content-addressed prepared artifacts and atomic writes exist;
- benchmark phase timing and provenance are substantially implemented;
- replay bundle capture, verification, scoring, and splitting exist;
- promotion, successive halving, SQLite leases, resumability, detached worktrees,
  and a merger exist;
- a warm live-campaign safety abstraction exists;
- dependency inventory and disclosure tooling exist.

The foundation batch was committed to main as
`b9382da162c1c1e2984288ad7f3cfa7e5a1b11f8` (`aigp: implement durable offline
development loop`). Main was clean after the commit and the checks below. This
closes the M0 implementation/bootstrap milestone and makes that commit, or any
later clean main commit, a safe worktree base.

Post-commit validation on 2026-07-18 used the canonical `scripts\dev.cmd`
entry point and did not connect to or command FlightSim:

- `test-vq2`: 303 passed in 5.54 seconds;
- `test-fast`: 1,273 passed, 20 skipped, 42 deselected in 65.51 seconds;
- `test-unit`: 1,273 passed, 20 skipped, 42 deselected in 62.67 seconds;
- `test-slow`: 2 passed, 1,333 deselected in 2.74 seconds;
- `test-benchmark`: 39 passed, 1 skipped, 1,295 deselected in 358.05 seconds;
- all 113 entries in `config/promotion_trusted_files.json` were present and
  matched their recorded SHA-256 hashes.

The committed foundation now provides the following:

- the default fast, unit, scoped VQ2, slow, and benchmark tiers are independently
  selectable and fail closed on missing plugins, unknown markers, or timeouts;
- the VQ2 policy pins the exact reviewed discovery inputs and 303-test count;
- the trajectory gate-metadata round trip, cache corruption/recovery,
  cold/warm equivalence, concurrent publication, and invalidation paths have
  regression coverage;
- `scripts\dev.cmd` is the canonical Windows entry point and works even when the
  host policy blocks direct `.ps1` execution;
- T0-T4 command shapes, tier-specific evidence, successive halving, immutable
  checkpoints, exact detached worktrees, and a single merger are implemented;
- T2-T4 are operational, nonpowered synthetic evaluators. T1 remains
  intentionally non-runnable until an operator replaces the example's private
  corpus, processor, and administrator-owned isolation-wrapper placeholders;
- T5 is absent from the ordinary scheduler. The campaign command freezes and
  hashes a plan but always refuses before any lease, preflight, simulator, or
  powered action because no reviewed out-of-process watchdog executor is shipped;
- no approved golden replay corpus, labels, calibrated production policy, or
  authorized powered campaign exists in this checkout.

The checked-in promotion files are examples, not evidence that the missing
private T1 inputs or a live boundary exist. See `docs/aigp/durable_improvement_loop.md`
for the exact trust, isolation, and provenance contract.

## Completed Wave 1A shared-contract freeze

Wave 1A was implemented in `fd51af3c587e7c3431719b79c1713344e7cc6d6f`,
hardened in `a6782cd9dcc34aee94e0f064021399985e0f6839`, and integrated
through the evidence-record commit `7b0d84ad9be969b303f8919bba10fd80381c65e5`.
It freezes exact `/1` contracts for host/frame timing, prediction horizons,
latency events, gate authority, raw observations, relative state, command
proposals, supervisor approvals, and ordinary T0-T4 evidence scope. The
production authority seam is now explicit:

```text
CommandProposal -> safety supervisor -> SupervisorApprovedCommand -> transport
```

The transport compatibility projection is pure and performs no send. The
external safety supervisor and stateful transport still own caller trust,
single-use enforcement, pacing, watchdogs, arm state, and cleanup. The mutable
legacy `AttitudeRateCommand` remains an integration boundary, not an authority
token.

Accepted offline evidence, with zero FlightSim access, was:

- 53 direct VQ2 contract tests;
- 183 final evidence/promotion/scheduler tests with 6 expected skips;
- the exact 356-test VQ2 policy in both canonical and isolated-cache runs;
- 1,448 fast tests with 20 expected skips and 42 deselections;
- a reviewed 116-entry trusted manifest with builder identity
  `d6e4cd31177281fe9010eeeeb7df1667c248c464a45444d692d5a8225a6dc033`
  and file SHA-256
  `45514d8edaad2874c79a95946ff4b7632d5b4ada7a0294bf1f08c3f730701253`;
- post-merge `test-vq2`: 356 passed, followed by an empty Git status on main.

These results do not alter the historical M0 counts above and do not claim a
private golden corpus, calibrated production geometry, powered execution, or
official-simulator evidence.

## Wave 1 offline integration record

The three Wave 1 implementation branches were merged on candidate
`361d0060f16dbaec753de00ba491f1a085707eb1` and integrated to main through
`1cf17ea5f4e0a330bee89b0128d30b13657899a2`:

- runtime timing adds exact camera publication timing, one-shot latest-value
  consumption, a no-burst 50 Hz scheduler, latency traces, and percentile
  summaries;
- gate geometry adds opt-in visible/censored inner-aperture fitting with
  conservative uncertainty and an explicit degraded bbox-only fallback that
  the estimator withholds;
- relative estimation adds variable-time feature filtering, innovation gating,
  bounded caller-supplied prediction, coasting/loss, same-session
  contract-validated gate/reset reinitialization, retained replay history, and
  cross-session rejection;
- the cross-workstream test binds a synthetic timed, read-only frame through
  detection, aperture fitting, and exact-source relative-state initialization,
  while proving that bbox-only geometry is withheld from the estimator.

Accepted candidate evidence, with zero FlightSim access, is:

- 187 combined affected tests;
- exactly 418 VQ2 policy tests in both canonical and isolated candidate runs;
- `test-fast` and `test-unit`: 1,510 passed, 20 skipped, 42 deselected each;
- a reviewed 119-entry trusted manifest with semantic identity
  `fd1d09e16c34dd3c77fb45877102dda56ca1da888b8b6c3cf5bf1408ffe0d4b8`
  and file SHA-256
  `44f985274eb41a5c6d12b6fa17e4d553facb14f8b854901d084eabc99c88af5e`;
- `git diff --check`: clean.

These are offline foundations. The scheduler, geometry adapter, and estimator
are not connected to the powered command path. Remaining M1/M2/M4 acceptance
requires production event wiring, recorded replay, IMU derotation and measured
delay evidence, controller integration, and separately authorized simulator
measurements. Main fast-forwarded from
`3de33c3a568bc86638d9d7ac4dac6124f1e15397` through
`1cf17ea5f4e0a330bee89b0128d30b13657899a2`; post-merge `test-vq2` passed all
418 tests in 6.31 seconds and tracked Git status on main was empty.

## Wave 2 offline integration record

The pure predictive controller, offline system-identification tooling, and
mapless guidance candidates were independently hardened and merged with the
integration-owned offline adapter in
`8176cbac20ff16bfa4b8c24764596d9366fe98cf`:

- the controller maps exact `RelativeGateStateV1` inputs to bounded
  `CommandProposalV1` values for Gate 0 approach and Gate 1 recentering, with
  covariance, source, timing, health, dwell, and command-envelope withholding;
- system-ID tooling defines bounded offline experiments and fits/selects
  delay/plant profiles without granting those profiles runtime authority;
- guidance owns gate-scoped active/shadow track ownership and chronology,
  countdown/GO and terminal lifecycle, Gate 0 initialization,
  credit/reacquisition, and conservative clipped-target behavior;
- the adapter is the sole guidance caller in the composition, threads accepted
  memory, latches the Gate 0 pitch basis, exact-binds decision/state/tick
  provenance, maps only the two reviewed controller modes, and emits
  source-less exact zero for every other phase including commit; and
- a generated, already-decoded timed image-space path reaches detector,
  aperture fit, relative estimator, guidance, and controller while proving
  bbox-only geometry remains withheld. It is not JPEG receiver/reassembly or
  recorded replay evidence.

Accepted offline evidence is:

- 36 direct adapter tests and 378 combined affected tests;
- exactly 743 VQ2 policy tests in canonical, isolated-manifest, immutable
  committed-candidate, and post-merge main runs;
- `test-fast` and `test-unit`: 1,835 passed, 20 skipped, 42 deselected each;
- promotion-boundary `test-full-non-live`: 1,876 passed and 21 skipped in
  500.10 seconds; skipped optional coverage is not positive evidence;
- a reviewed 123-entry trusted manifest with semantic identity
  `4b8bae1511225f4ed79baa14ec015721069b8c37e98b81de7454bcabe7388988`
  and file SHA-256
  `79b8769f04902c2b2f87a45109b7a9aaa6b5cbf4ad3c4122593a8347ec57c689`;
  and
- main fast-forwarded from `e9a416714b01c2845786a0a22b168a9037f379ec`
  to the integrated candidate; post-merge `test-vq2` passed all 743 tests in
  29.64 seconds and tracked Git status was empty.

No FlightSim process was launched or contacted and no preflight, reset,
arm/disarm, target, transport, or powered action occurred. The adapter remains
offline-only: attitude and Gate 0 pitch lack timestamp/source correlation,
caller-threaded pure memory is a trust boundary, no approved replay corpus or
final processor is present, and measured delay/plant and tracker-isolation
replay evidence remain absent. Nothing in Wave 2 changes the established live
status or authorizes shadow, runtime, supervisor, transport, or powered wiring.

## Wave 3 control-plane dogfood increment

Offline increment `ab62cde9464442e4b448f293ba8efd31ad601c27` closes the
missing positive `SingleMerger` unit path. A fully promotion-valid synthetic
T0-T4 ledger fixture now proves exact fast-forward to a descendant candidate,
clean target state, unchanged checkpoints, correct nonpowered evidence domains,
and orchestration-lease release. Dirty and divergent targets fail without
advancing. The three new cases passed; the scheduler module passed 62 tests
with 6 skips, `test-fast` passed 1,838 tests with 20 skips and 42 deselections,
and candidate/post-merge `test-vq2` passed all 743 tests.

The 123-entry trusted manifest changed only the scheduler-test hash; its
semantic identity is
`60680aec1f26b3661576b65221ab4aeba4fab5df8959e46076dd8f99fce8fe41`
and file SHA-256 is
`0658ef17b864ce60312917aca208cd66263144f44d8fcc86ba4c37be1ebb2be5`.
This is a unit fixture, not an operational scheduler-run candidate. A genuine
T0-T4 exercise remains blocked by the approved replay corpus, production
processor, and administrator-owned isolation wrapper required by T1. Campaign
and T5 remain outside this work.

## Wave 3 offline IMU provenance and derotation integration record

Behavioral implementation
`f53718da892c4ab5aecc567a61249b21a8cb6ffa`, main reconciliation merge
`e3f386d460d012c7b9710ae440c2ac405447f1f3`, and promotion/trust closeout
`ecaa794aeaed87a169b7b87b284d1440f1768a28` add a deterministic local
HIGHRES_IMU provenance envelope, bounded pure rotation-only camera-ray
correction, and offline outer adapter around the unchanged Wave 2 composition:

- exact session/reset, host-clock, IMU stream/generation, sequence, source-time,
  per-sample host-arrival, camera observation, prediction-target, calibration,
  and model identity are retained and fail closed;
- Gate 0 pitch is derived and latched from exact accepted phase-entry attitude
  evidence, while controller attitude is separately propagated to proposal
  time under conservative extrapolation, age, and angular-uncertainty caps;
- invalid, stale, future, uncertain, relabeled, or incoherent evidence produces
  a source-less exact-zero proposal without advancing invalid visual ownership;
  and
- the ordinary raw-camera estimator remains bit-for-bit unchanged. The
  rotation-corrected bearing is standalone evidence and is not injected into
  its capture-time posterior, guidance, or a frozen `/1` proposal.

Accepted offline evidence is:

- 143 affected IMU-adapter, provenance, derotation, and estimator tests;
- exactly 872 canonical VQ2 policy tests, including the post-merge main run;
- `test-fast` and `test-unit`: 1,967 passed, 20 skipped, and 42 deselected each;
- promotion-boundary `test-full-non-live`: 2,008 passed and 21 skipped;
- a strict 126-file trusted manifest with semantic identity
  `f074019f30858b9fcc5fb06a90a8df7cf57770e84791893ddbaa082861eca5eb`
  and file SHA-256
  `ba07b6ea73b5fc88f99e6c8824ea4d7039c956391de2c5730e6716af76cad9b1`;
- exact VQ2 policy file SHA-256
  `4352163c57b06f8bb12a7b7750c8a279d76b0c45d933dacf3d5149238ee970ef`;
  and
- independent lifecycle and adversarial review cleared. Main fast-forwarded to
  `ecaa794aeaed87a169b7b87b284d1440f1768a28`; post-merge `test-vq2`
  passed all 872 tests and tracked Git status was empty.

No FlightSim process was launched or contacted, and no preflight, external
network access, reset, arm/disarm, target, transport, shadow, simulator, or
powered action occurred. This local evidence does not make a bare `/1`
proposal supervisor-verifiable. Runtime promotion still requires a reviewed
`/2` envelope or supervisor-owned registry, production per-sample arrival
capture, and calibrated camera/IMU timing and extrinsics. Applying the corrected
ray still requires a stable-frame or explicitly time-aligned filter. Measured
command/actuator/gyro delay, approved replay, tracker-isolation, shadow/runtime,
and powered evidence remain absent.

## Wave 3B generated offline runtime integration record

Behavioral implementation
`8eab146e3a9a7a1a1b28070d3e0234adff900595`, main reconciliation merge
`7904fbadbc4b220b81afb846a69b15a7b30ef4bb`, and promotion/trust closeout
`28b7d782404d6b825cebae3b65a8443d756be234` add the deterministic
already-decoded runtime composition without connecting any authority surface:

- one owner contains the exact latest-frame cursor, fixed-rate scheduler,
  local IMU estimator/history, raw-camera relative estimator, Wave 3 adapter
  memory, proposal sequence, and cumulative trace;
- nested camera metadata/storage, safety/camera/IMU epochs, timing plans,
  selected attitudes, pipeline outcomes, diagnostic reasons, and exported
  result/trace correlations fail closed before commit;
- distinct publications run perception once; repeated frames run no perception
  and remain source-less exact zero; deadline and planned-overrun skips never
  advance pipeline state or create catch-up bursts;
- corrected bearing remains standalone evidence and is not applied to the
  raw-camera state, guidance, or proposal; and
- `GYRO_SAMPLE` is an occurrence-only fact. The result and trace reject send,
  actuator, command, simulator, network, and powered authority.

Accepted offline evidence is:

- 38 focused runtime tests and 181 coupled runtime/IMU/adapter/estimator tests;
- an independent 199-test compatibility/adversarial matrix with explicit
  clearance and no remaining tranche-local blocker;
- exactly 910 canonical and isolated-manifest VQ2 policy tests, including the
  post-merge main run;
- `test-fast` and `test-unit`: 2,005 passed, 20 skipped, and 42 deselected each;
- promotion-boundary `test-full-non-live`: 2,046 passed and 21 skipped;
- a strict 127-file trusted manifest with semantic identity
  `cdd0db402b6f1c8bb0c90c1b8d445ca64741d3bfc3aa03a78c3fe4d73c8dcce2`
  and file SHA-256
  `e270a194031d463accfb50b28bd3296eb672004d1c41241fab3cb368bab1640a`;
- exact VQ2 policy SHA-256
  `64cfefc083a52fc925ad98c2e3a99e8f6eefcaebb0f4243d214c1e87729a864c`.

The trust review added only the new runtime test and changed only the policy
hash; no file was removed and the trust root did not expand. No FlightSim
process was launched or contacted, and no preflight, external network access,
reset, arm/disarm, target, transport, shadow, or powered action occurred.
Receiver/reassembly and recorded replay evidence, production per-sample arrival
capture, calibrated timing/extrinsics, a supervisor-verifiable provenance
envelope, measured command/actuator/gyro response, and powered evidence remain
absent.

## Wave 3C proof-bound correlated coast integration record

Behavioral implementation
`84674fd8c7379b327e25725010ca58a57f4fd910` and promotion/trust closeout
`168220ba7060d07743335d0e9c56bcd2d05d669d` add the default-off,
single-successor correlated coast without changing the public dropout default
or connecting any authority surface:

- a healthy accepted active update may arm one lease only when its source
  scheduler tick starts exactly on due; a valid late proposal remains accepted
  but opens no unusable lease after scheduler rebasing;
- the exact immediate repeated-frame successor may advance only constant-
  velocity prediction state to first-dropout `COASTING`, using a strictly newer
  causal same-source attitude and strictly growing marginal uncertainty;
- public guidance, controller, and Wave 2 paths still reject dropout. Only the
  Wave 3-owned private capability path may produce an explicitly uncertainty-
  limited coast proposal;
- skips, new-frame selection or failure, lifecycle mismatch, malformed or
  unavailable evidence, coast success, and reuse consume the lease; the second
  repeat remains source-less exact zero; and
- every consuming result retains and reconstructs its exact prior source
  transition, then binds source/current camera, perception, IMU, scheduler,
  estimator, controller, terminal, and cumulative-trace facts before commit.

Accepted offline evidence is:

- 74 focused runtime tests, 95 Wave 3 adapter tests, 32 relative-estimator
  tests, and a 477-test six-module affected matrix;
- an independent 201-test deep contract matrix with explicit clearance and no
  remaining tranche-local blocker;
- exactly 1,019 canonical and isolated-manifest VQ2 policy tests, including the
  post-merge main run;
- `test-fast` and `test-unit`: 2,114 passed, 20 skipped, and 42 deselected each;
- promotion-boundary `test-full-non-live`: 2,155 passed and 21 skipped;
- a strict 127-file trusted manifest with semantic identity
  `f9118fad5fdbdd8e5e355cf0e153492525b853b9b7c32239ab4d2d81f6d63b2b`,
  file SHA-256
  `29b306e41a6954552ef7693f0e0c3d853cc4b60aeedfb59f6a2c9592ece9d8c6`,
  and exact policy SHA-256
  `29eb2dcd627a8f5dbbea4bf88c249a87ca741ca5c9d743c0c646404f40e8748e`.

The trust review replaced exactly the six changed test hashes plus the policy
hash, with no file addition/removal or trust-root expansion. Main fast-forwarded
to the promotion commit and post-merge VQ2 passed all 1,019 tests. No FlightSim
process was launched or contacted, and no preflight, external network access,
reset, arm/disarm, target, transport, shadow, or powered action occurred.
Receiver/reassembly, recorded replay, production timing/extrinsics, a
supervisor-verifiable proof carrier, stable-frame corrected-ray application,
measured response, shadow/runtime acceptance, and powered evidence remain open.

## Wave 3D stable-reference local-feature integration record

Contract freeze `ede0edca3025ad03db3032e371a37c86dc8fdc00`, behavioral
implementation `c21a742004d1d3bc485a866babb9759b6aee62fb`, and promotion/trust
integration `46df0adee76070e10509fa5e807b986a9469c68e` add a pure,
default-off, bidirectional stable-orientation transform with no production call
site.

The transform deliberately does not reinterpret frozen `/1` finite-quad
`log_scale`. It uses the distinct local differential semantic
`vq2-local-differential-area-v1`, exact capture/target camera-time binding, a
seed-bound stable basis fingerprint, complete camera/authority/IMU/model
lineage, determinant-one projective math, full bearing/rate/expansion chain
rules, an analytic dense `6x6` Jacobian, and separated coordinate, declared
joint-nuisance, model-floor, and total covariance terms. Reference creation
requires a complete visible unclipped seed aperture, while later usable
derotation sources may omit finite-quad summaries because their local feature
is independently supplied.

This is a mathematical prototype, not an estimator input. Its covariance scope
rejects directly returned total-labelled states but is a caller assertion, not
an unforgeable provenance carrier. Acceleration bounds and nuisance dominance
are explicit declarative model assumptions; the module does not derive or
prove the supplied envelope. Production use still requires a reviewed local-
scale measurement/covariance producer from the full fitted homography or a
shape-augmented finite-quad state, calibrated camera/timing models, replay, a
sequential nuisance treatment, estimator integration, runtime acceptance, and
separate authority review.

Accepted offline evidence is:

- 82 direct stable-reference tests and a 186-test compatibility matrix;
- three independent final reviews clearing the math, lifecycle, tests,
  documentation, and no-wiring boundary;
- exactly 1,101 canonical and isolated-manifest VQ2 tests, including the
  post-merge main run;
- `test-fast` and `test-unit`: 2,196 passed, 20 skipped, and 42 deselected each;
- promotion-boundary `test-full-non-live`: 2,237 passed and 21 skipped; and
- a strict 128-file manifest with semantic identity
  `2f70415dd7cdfa0675c6dc778406cdccfdca09757e79b1a8f1a3e0d4752e9268`,
  file SHA-256
  `2c965f2f5a6486f506d51c8e290b09d6a22166f6f277fbff1234690e510d63d9`,
  and policy file SHA-256
  `a98b2d4d618b6999927d1c997ca0a65c63aebef742c53bef31a6c05dcd53b020`.

The trust review added only the new 82-test file, changed only the policy hash,
and removed nothing. No FlightSim process was launched or contacted, and no
preflight, external network, replay, reset, arm/disarm, target, transport,
shadow/runtime, or powered action occurred. All prior live evidence and safety
limits remain unchanged.

## Wave 3E rectified-homography local-measurement integration record

Contract freeze `c7dcb612318eb9d26868fa1364c1a027d2b8edcd`, review-driven
contract correction `aab44d48a032444faeaf5cd1020e90dc9dbd24ed`, behavioral
implementation `ceed9c854b0066d4f00d4add796fb968d449593a`, and promotion/trust
integration `16dd5c84995cafb5158e277d730e670557ba69f2` add a standalone,
immutable reducer with no production call site.

The reducer begins only after an external producer supplies a rectified,
center-gauge-fixed full homography and dense conditional `8x8` covariance. It
returns center bearing plus local differential log scale, an analytic `3x8`
Jacobian, dense conditional covariance by full congruence, exact frame and
authority provenance, raw and canonical covariance views, deterministic
fingerprints, and bounded projective/numerical diagnostics. It does not accept
frozen `/1` observations, corners, rates, states, or total covariance and does
not call Wave 3D, the estimator, runtime, controller, or supervisor.

This closes only the homography-to-local-feature T0 algebra and lifecycle
boundary. The current fitted-observation/observation-adapter path publishes
only a heuristic covariance over five image-space summaries and uncalibrated
geometry. The differential of those five summaries has rank at most five over
eight homography degrees of freedom, so it cannot determine an honest full-
homography covariance or Wave 3E input. Production use remains blocked on
calibrated image-to-ray rectification, an independently reviewed
full-homography fit covariance, shared nuisance treatment, approved recorded
replay, cross-frame association, sequential filtering, estimator/runtime
integration, and authority review.
No production module produces `VQ2RectifiedHomographyInput`; do not derive it
from `/1`, fitted corners, residuals, guessed covariance, or border-normalized
image coordinates, and do not wire its evidence downstream before those
prerequisites are independently reviewed.

Accepted offline evidence is:

- `224` direct reducer tests and a `450`-test compatibility matrix;
- three independent final reviews clearing mathematics/numerics,
  lifecycle/provenance/no-wiring, and API/test coverage;
- canonical, cache-clean isolated-manifest, and post-merge main VQ2 runs each
  passed exactly `1,325` tests;
- `test-fast` and `test-unit`: `2,420` passed, `20` skipped, and `42`
  deselected each;
- promotion-boundary `test-full-non-live`: `2,461` passed and `21` skipped in
  `487.69s`; and
- a strict `129`-file trusted manifest with semantic identity
  `46e77cbbe8a131517444b141293b1fe8c2bab546a6f5630f711ffe0d621d5ea2`,
  file identity
  `e88363ef096bba83fe4660a4903abb6ae063f41682246b38ba9c69481008fffc`,
  and policy file identity
  `7daa46ec4dfd025c18f12076add06d70b6463f07d6320b20487a63bd78d0851e`.

The trust review added only the new `224`-test file, changed only the policy
digest, removed nothing, and independently matched all `129/129` files. Main
fast-forwarded cleanly and post-merge VQ2 passed all `1,325` tests. No
FlightSim process was launched or contacted, and no preflight, external
network, replay, reset, arm/disarm, target, transport, shadow/runtime, or
powered action occurred. All prior live evidence and safety limits remain
unchanged.

## Completed foundation bootstrap

The former dirty-worktree exception is closed:

- the P0-P2 batch was preserved and committed as `b9382da`;
- no later feature worktree needs uncommitted state from the old main tree;
- clean implementation worktrees may branch from `b9382da` or a later clean
  integration commit;
- `c7c37c0` is retained only as the historical pre-foundation reference.

Two operationalization items remain, but neither requires reopening the old
dirty batch or blocks Wave 1A:

1. Exercise a real clean candidate through T0-T4, including scheduler leases,
   immutable checkpoints, interruption/resume, deduplication, exact worktrees,
   and the reviewed merger path.
2. Replace the T1 example placeholders only after an approved private corpus,
   production replay processor, calibrated policy, and administrator-owned
   isolation wrapper exist.

If main ever becomes dirty unexpectedly, stop integration, attribute and
inventory the changes, and move the owned work onto a task branch/worktree before
continuing. Quarantine unexplained state. Never erase it with `reset --hard`,
checkout-based discard, or bulk cleanup merely to make status appear clean.

## Run-to-completion orchestration

The program can now proceed without dirty-bootstrap ambiguity. Use the existing
ledger and scheduler as the durable source of evaluator/trial state, and use the
task manifests, branches, and worktrees below for implementation state. The trial
scheduler does not silently snapshot or own arbitrary uncommitted code.

### Task lifecycle

Each task moves through:

    queued
      -> leased
      -> active
      -> tested
      -> committed
      -> integration_pending
      -> integrated
      -> post_merge_verified

Failure, timeout, or interruption records evidence and returns the task to a
reviewable or resumable state. It does not leave an anonymous dirty worktree.

### Task manifest

Every implementation task should record:

- task and parent identifiers;
- objective and explicit non-goals;
- starting main commit;
- branch and worktree path;
- owned files or module boundary;
- dependencies and interface versions;
- required tests and acceptance metrics;
- simulator access level: none, passive, or powered;
- artifact/cache roots;
- lease owner and heartbeat;
- final commit hash;
- result and failure provenance.

### Start checks

Before an agent writes:

1. Main commit matches the task's declared base or the task is rebased deliberately.
2. The task worktree has an empty status.
3. No other active task owns the same integration-hot file or schema.
4. Required interface/schema versions match.
5. The appropriate cache and artifact roots are isolated.
6. Live simulator lease is held if needed.

If any check fails, the task waits or is replanned; it does not improvise inside
another task's worktree.

### Completion checks

A task is finished only when:

- the worktree is clean;
- all intended changes are committed;
- directly affected tests pass;
- the required promotion tier passes;
- artifacts and metrics have hashes and provenance;
- documentation or decision records are updated when behavior changed;
- the branch is ready for integration without hidden local state.

### Integration checks

Use one integration owner:

1. Rebase or merge the candidate onto current main in its own worktree.
2. Resolve conflicts there, never through uncommitted edits on main.
3. Re-run affected tests and the VQ2 suite.
4. Run any higher promotion gate required by the change.
5. Merge or fast-forward the committed branch into main.
6. Run a short post-merge verification.
7. Verify main is clean.
8. Mark the ledger task integrated and retire the worktree only after evidence
   is safely stored.

Main may advance frequently. Finished, green work should be merged rather than
accumulating a long-lived integration branch.

### Resume and recovery

- Leases expire and may be reclaimed.
- Completed checkpoints are not repeated.
- A dirty abandoned worktree is quarantined and attributed to its task.
- Its diff and untracked inventory are recorded before human or integration-owner
  review.
- Automation never erases an unexplained dirty state.
- Live trials resume only from a fresh reset and preflight, never from an assumed
  in-flight state.

## Target architecture and ownership boundaries

    Sensor ingress
      -> immutable timestamped frame / IMU / race data
      -> perception: GateObservation[]
      -> active and shadow tracking
      -> relative-state estimator
      -> guidance objective
      -> pure controller: CommandProposal
      -> safety supervisor and phase authority
      -> SupervisorApprovedCommand
      -> MAVLink transport

Recording is a non-blocking side channel at each boundary.

Suggested versioned contracts:

GateObservation:

- generation, frame ID, and measurement timestamp;
- candidate identity;
- normalized center;
- visible inner and outer edges;
- inner corners when available;
- apparent scale and skew;
- clipping mask;
- confidence, covariance, and residual diagnostics.

RelativeGateState:

- measurement and prediction times;
- authoritative gate epoch/index;
- normalized bearing and bearing rates;
- log scale and expansion rate;
- optional relative pose and velocity;
- covariance;
- clipping and visibility flags;
- innovation and health state.

CommandProposal:

- proposal timestamp and source-state timestamp;
- requested body rates and thrust;
- phase and reason;
- saturation and uncertainty diagnostics;
- no transport, arm, reset, or cleanup authority.

## Milestones

| Milestone | Status | Outcome | Main prerequisite |
|---|---|---|---|
| M0 | Complete at `b9382da` | Stable, green, committed evaluation foundation | Historical bootstrap |
| M1 | Active; generated scheduler/trace composition integrated, production receiver/runtime trace and simulator dossier pending | Runtime timing and simulator semantics dossier | M0 and frozen Wave 1A timing contracts |
| M2 | Active offline; censored aperture fitter integrated, recorded replay and tracker-isolation acceptance pending | Robust clipped Gate 1 geometry with uncertainty | M0, frozen Wave 1A observation contracts, and replay evidence |
| M3 | Pending | Bounded Gate 1 recentering without passage | M1 and M2 |
| M4 | Active offline; estimator, controller, guidance, exact adapters, timestamped attitude provenance, rotation-only evidence, generated runtime composition, proof-bound one-tick correlated coast, standalone stable-reference math, and the standalone full-homography-to-local-measurement reducer are integrated; calibrated rectification, a dense full-homography covariance producer, approved replay, sequential nuisance/state design, measured delay, estimator wiring, and production runtime evidence remain pending | Predictive relative state and delay-compensated IBVS | M1 and M2 |
| M5 | Pending | Separately reviewed Gate 1 passage | M3 and M4 |
| M6 | Pending | Conservative valid full lap | M5 |
| M7 | Pending | Repeatable baseline across fresh and warm sessions | M6 |
| M8 | Pending | Safe time optimization | M7 |
| M9 | Pending | Advanced pose, mapping, MPC, ILC, or learned residuals | Evidence after M7 |

M0-M4 allow substantial offline parallelism. M5 onward is increasingly
serialized around integration and live evidence.

M0 completion records the committed implementation and green core, slow, and
benchmark tiers. The clean-candidate T0-T4 scheduler/resume/merger exercise below
is still required as operational hardening, but it no longer blocks clean Wave 1A
branches.

## Workstream A: evaluation foundation

State: the foundation is committed and its canonical test tiers are green.

Remaining tasks:

- run a clean committed candidate through T0-T4 using the implemented scheduler,
  and retain evidence for leases, interruption/resume, deduplication, immutable
  checkpoints, exact worktrees, and the merger path;
- provision an administrator-owned pinned isolation wrapper and approved private
  replay corpus before attempting T1 outside its synthetic/mocked tests;
- implement and review the competition-specific replay processor once the final
  interface is known;
- obtain approved replay sessions and calibrate a production policy.

Already verified at `b9382da`:

- test-fast, test-unit, test-vq2, bounded slow, and explicit benchmark tiers are
  green;
- all trusted-manifest files match their committed hashes;
- prepared cold/warm metrics match within declared tolerance;
- no skipped evaluator can satisfy a promotion gate;
- T0/T1 never claim closed-loop completion and T2-T4 identify their synthetic
  nonpowered domain;
- dirty or provenance-mismatched candidates are rejected;
- the canonical Windows command works on the target host.

Operational acceptance still requires the real clean-candidate scheduler exercise
and, for production T1 specifically, the private trust-boundary inputs above.

## Workstream B: runtime timing

State: the offline timing and scheduler foundation is integrated on main. Real
detection-to-actuator/gyro tracing, production send-path scheduling, simulator
load measurements, and calibrated measurement/command-delay models remain
open.

Instrument:

    camera epoch timestamp
      -> first and last packet arrival
      -> reassembly
      -> decode
      -> detection
      -> tracking
      -> estimator update
      -> prediction
      -> controller decision
      -> command send
      -> actuator and gyro response

Measure p50, p95, p99, and maximum latency; command intervals; deadline misses;
repeated-frame ticks; frame drops; queue depth; duplicate packets; stream rates;
simulator/wall ratio; graphics preset; focus state; process uptime; and host load.

Scheduler outcome:

- at least 20 ms between command sends;
- near-50-Hz steady operation when work fits;
- missed ticks are skipped;
- catch-up bursts are impossible;
- perception updates only on distinct frames;
- state predicts at IMU/control rate.

Remain at the reviewed 50-Hz cap initially.

## Workstream C: gate geometry and perception

State: stages 1-2, a bounded image-space subset of stage 3, and gate-scoped
active/shadow guidance ownership are integrated with deterministic synthetic
evidence. The fitter uses an
explicitly censored pixel-square prior, not a calibrated planar/physical-square
or metric-pose model. Gate 0 replay, recorded Gate 1 stability, and
tracker/crossing-residue isolation replay remain open.

Stages:

1. Treat missing top, bottom, left, and right edges as censored observations.
2. Extract inner aperture edges and corners.
3. Fit a planar/square aperture with residuals and uncertainty.
4. Maintain separate active and shadow tracks.
5. Evaluate IPPE/PnP only when corner quality and calibration support it.
6. Add multiple-gate association only after single-gate behavior is stable.

Acceptance:

- Gate 0 replay does not regress;
- the recorded top-clipped Gate 1 has stable center and honest uncertainty;
- partial geometry increases uncertainty instead of inventing precision;
- crossing residue cannot seed the next active track;
- placeholder distance or pose does not reach production control.

If corners remain unreliable under clipping, use edge, bearing, and scale
features rather than forcing a metric pose.

## Workstream D: filtering and state estimation

State: the feature filter, distinct-frame updates, innovation gating,
covariance growth, coasting/loss, bounded prediction, exact local IMU/attitude
provenance, and rotation-only camera-ray correction are integrated offline with
the pure controller/guidance composition. The corrected bearing remains
standalone evidence and is not applied to the raw-camera filter. A reviewed
standalone stable-reference transform now proves the local differential
coordinate math, but it intentionally cannot transform frozen `/1` finite-quad
scale and has no estimator/runtime call site. Wave 3E now proves the separate
full-homography-to-local-measurement algebra and first-order dense conditional-
covariance congruence, but the current detector cannot produce its calibrated
full-homography input and no production consumer is wired. A reviewed producer,
calibrated camera/IMU timing and extrinsics, shared and sequential nuisance
treatment, cross-frame association, calibrated command-effect prediction,
runtime wiring, p95 replay comparison against zero-order hold, and
shadow/runtime IBVS evidence remain open.

Initial estimator:

- small Kalman or alpha-beta feature filter;
- one update per distinct frame;
- normalized bearing, log scale, and their rates;
- IMU derotation from capture to prediction time;
- confidence- and clipping-dependent covariance;
- normalized-innovation gating;
- bounded dropout prediction;
- explicit unhealthy and uncertain states.

Attitude work:

- retain short-horizon gyro propagation;
- measure stationary bias and lap-duration drift;
- gate accelerometer correction by dynamics;
- calibrate camera/IMU temporal offset;
- keep yaw command zero until independently calibrated.

Later candidates:

- gate-relative error-state filter;
- optional VIO in shadow mode;
- gate-based translation/yaw drift correction.

Acceptance:

- deterministic replay;
- no duplicate-frame double update;
- lower future-frame center error than zero-order hold, especially p95;
- bounded dropout uncertainty;
- shadow-mode evidence before command authority.

## Workstream E: control and gate progression

State: the pure controller, bounded Gate 0 approach and Gate 1 recenter proposal
modes, normalized bearing/rate damping, bounded elapsed-time and vertical-error
thrust scheduling/control, uncertainty withholding, saturation/dwell limits,
mapless guidance lifecycle, exact offline adapter, and local timestamped
controller-attitude/Gate 0 pitch provenance are integrated. A frozen `/1`
proposal still cannot carry that provenance to the supervisor. Measured
command-effect timing, supervisor-verifiable provenance, supervisor/runtime
wiring, and any powered Gate 1 recenter or passage remain open.

Extract a pure deterministic controller:

    controller(relative_state, attitude, phase, config)
      -> CommandProposal and diagnostics

Stages:

1. Reproduce the current Gate 0 command behavior.
2. Add a bounded Gate 1 recenter stage after proved 0-to-1 credit.
3. Control normalized bearing and bearing-rate damping.
4. Schedule forward progress from scale expansion/time-to-contact.
5. Predict to measured command-effect time.
6. Add saturation-aware scheduling and slew limits.
7. Generalize phases:
   - acquire;
   - align;
   - accelerate;
   - approach;
   - commit;
   - exact-zero confirmation;
   - post-credit reacquisition.
8. Pass Gate 1 through a separately reviewed stage.
9. Generalize the expected i-to-i+1 loop.

Gate 1 recentering must end at its corridor or timeout and must not attempt
passage. An unexpected gate-index transition aborts the stage.

Relative visual NMPC is a later candidate, not an MVP prerequisite.

## Workstream F: navigation and planning

Mapless core:

- one authoritative active gate at a time;
- keep aperture visible;
- uncertainty-aware crossing margin;
- approximately normal approach when skew is observable;
- shadow next gate without authority;
- generic expected-index sequencing.

Local lookahead:

- small local gate map;
- two- or three-gate horizon;
- crossing point selected partly for the next turn;
- visibility and predicted image velocity objectives;
- center-crossing fallback under uncertainty.

Optional simulator priors:

- keyed by build, course signature, calibration, code, and schema;
- validated before use;
- never override current observations or authoritative sequence;
- mapless cold-start remains functional;
- disabled for the physical/no-prior profile.

Global time optimization begins only after a repeatable full lap and credible
metric state. It should optimize finite apertures and include measured hidden
rate-loop lag. MPCC, aperture-aware trajectory optimization, retiming, ILC, and
learned residuals remain evidence-driven candidates.

## Workstream G: simulator characterization and system identification

State: bounded experiment definitions, offline fitting, and deterministic
profile selection are integrated. Wave 2 ran no FlightSim experiment and
selected no profile for runtime authority; new passive or powered evidence
retains the authorization and simulator-lease boundaries below.

Passive:

- packet and field census;
- race-status cadence and jitter;
- camera/IMU/race timestamps versus wall time;
- graphics preset and focus/minimize A/B tests;
- pilot-feed invariance;
- simulator-time/wall-time behavior;
- reset and initial-state repeatability.

Bounded powered:

- repeat sign identification;
- hover equilibrium;
- small rate doublets/chirps;
- thrust steps;
- per-axis command delay, lag, slew, saturation, and cross coupling.

Later flight identification:

- pitch/thrust to image expansion;
- forward acceleration and braking;
- drag;
- command/gyro/visual/race-event alignment;
- low-speed valid crossings for pass-plane and scoring-time inference.

Fit simple models first and validate them on held-out pulses. Preserve model
uncertainty. Do not deliberately crash merely to map the collision envelope.

## Workstream H: promotion, reliability, and qualification

Evidence ladder:

| Tier | Evidence |
|---|---|
| T0 | Unit and interface correctness |
| T1 | Golden replay perception, estimation, and open-loop commands |
| T2 | VQ2-specific deterministic closed-loop surrogate |
| T3 | Changed-domain scenarios and repetitions |
| T4 | Full non-live matrix and regression suite |
| T5 | Explicit bounded Training-mode FlightSim trial |

The T2 evaluator should include 30-Hz timestamped vision, clipping, IMU
propagation, measured rate lag, latency/jitter, race-status delay, bounds, and
authoritative sequencing. It is regression evidence, not official-simulator
equivalence.

After the first full lap:

- measure warm and fresh-session repeatability;
- interleave baseline and candidate trials;
- record process uptime and trial count;
- stop on baseline drift;
- use successive halving before live evaluation;
- retain conservative and aggressive configurations.

A provisional reliability target is ten consecutive conservative valid laps
across mixed warm and fresh sessions. Adjust the final sample requirement after
measuring simulator repeatability.

Before Qualification:

- freeze commit, configuration, dependencies, calibration, build, and artifacts;
- complete dependency and tool disclosure review;
- disable online search and human interaction;
- preserve a reliable fallback candidate;
- submit only previously promoted frozen candidates.

## Workstream I: sim-to-real optionality

The simulator may use resets and track-specific priors. The physical core must
not require them.

- Version intrinsics, extrinsics, time offsets, IMU characteristics, thrust
  mapping, rate dynamics, and command envelopes.
- Keep mapless relative visual operation as fallback.
- Put reset automation, course priors, and ILC behind optional providers.
- Add held-out robustness tests for delay, loss, blur, lighting, clipping,
  calibration error, and dynamics variation.
- Keep transport-specific MAVLink below the controller.
- Make physical gate-progress authority a separate reviewed provider rather
  than reusing a visual heuristic as simulator race status.

## Worktrees and parallel execution

Recommended root:

    C:\Users\John\aigp-worktrees

Create only the worktrees needed for the active wave:

| Worktree | Ownership |
|---|---|
| wt-runtime-timing | Timebase, latency instrumentation, scheduler tests |
| wt-gate-geometry | Clipping, aperture, corners, replay metrics |
| wt-relative-estimation | Feature filter, prediction, estimator health |
| wt-vq2-control | Pure predictive controller modules |
| wt-system-id | Offline fitting and bounded experiment definitions |
| wt-vq2-planning | Mapless guidance, lookahead, optional priors |
| wt-evaluation | VQ2 surrogate, scoped evidence, tuning |
| wt-vq2-integration | Sole runner, safety, and live integration owner |

Always serialize ownership of:

- scripts\aigp_vq2_run.py and live stage sequencing;
- safety constants, reset, confirmation, and cleanup;
- replay/observation schema changes until versioned;
- benchmark schema and cache-key changes;
- ledger merger and promotion decisions;
- every live simulator connection.

Each worktree uses its own cache/artifact root and no ad hoc shared writable
SQLite database. A locked environment may be shared read-only. Do not install
dependencies during a performance experiment.

### Four-slot execution waves

Wave 0, completed:

- the testing/evaluation foundation was stabilized and committed as `b9382da`;
- the passive simulator characterization and existing Gate 0/Gate 1 evidence
  inventory were completed;
- main returned to a clean state;
- the planned shared contract freeze did not land in the foundation commit and is
  explicitly carried into Wave 1A. No dirty-tree owner remains.

Wave 1A, shared-contract freeze completed:

- Agent 1 drafted the timestamp, frame-identity, prediction-time, and
  latency-event contract;
- Agent 2 drafted versioned `GateObservation` and `RelativeGateState` contracts,
  including clipping, confidence/covariance, health, and reset/gate epochs;
- Agent 3 drafted `CommandProposal`, `SupervisorApprovedCommand`, and
  tier-scoped evidence contracts plus
  compatibility fixtures;
- the coordinating/integration owner reconciled and landed the shared schemas,
  adapters, and contract tests before downstream branches depended on them.

The freeze is implemented by `fd51af3c` and hardened by `a6782cd9`; its exact
reference is `docs/aigp/vq2_contracts.md`. Production T1 inputs were not a
prerequisite for the freeze.

Wave 1 offline foundations are integrated and post-merge verified; M1/M2/M4
acceptance remains incomplete:

- runtime timing: exact camera publication timing, latest-value consumption,
  fixed-rate scheduling, and latency summaries;
- gate geometry: opt-in visible/censored inner-aperture fitting with conservative
  uncertainty;
- relative estimation: variable-time feature filtering, innovation gating,
  bounded prediction, coasting/loss, same-session gate/reset reinitialization,
  retained replay history, and cross-session rejection;
- coordinator: cross-workstream source binding, exact 418-test policy, and
  reviewed 119-entry trusted manifest.

The new scheduler, geometry adapter, and estimator are not connected into the
powered command path, and no official-simulator or powered evidence is claimed.

Wave 2, completed offline and post-merge verified:

- pure predictive IBVS/controller proposals;
- system-identification tooling and bounded experiment definitions;
- mapless guidance/state machine;
- exact offline guidance/controller adapter and generated cross-layer path.

Wave 3, active offline before any separately authorized live work:

- completed: timestamp and source-bind attitude/Gate 0 pitch inputs and add
  bounded rotation-only IMU derotation evidence while leaving the raw-camera
  estimator unchanged;
- completed: connect the exact generated, already-decoded perception and IMU
  path through the Wave 3 adapter and fixed-rate scheduler, terminate at a
  quarantined proposal, and bind honest generated stage/IMU occurrence facts;
- completed: add only a default-off, proof-bound one-tick IMU-correlated coast
  for the immediate repeated-frame tick. Consume the lease on a skip, new-frame
  attempt or failure, lifecycle change, invalid IMU, or reuse; retain exact-zero
  behavior everywhere else;
- completed: add a standalone immutable stable-reference transform for a
  distinctly named local differential feature state, with exact reference
  lineage, forward/inverse rate chain rules, dense covariance congruence, and
  no `/1`, estimator, runtime, or authority wiring;
- completed: add a standalone rectified full-homography reducer for center
  bearing and local differential scale with an analytic `3x8` Jacobian, dense
  conditional-covariance propagation, exact provenance and integrity, and no
  detector, Wave 3D, estimator, runtime, or authority wiring;
- next: provision approved replay inputs/final processing and exercise the
  operational T0-T4 promotion path without relabeling generated evidence;
- next state-estimation prerequisite: provide calibrated rectification and an
  independently reviewed producer of the full homography plus dense covariance
  required by Wave 3E, with approved replay and explicit shared-nuisance
  treatment. A shape-augmented finite-quad state remains a separate redesign;
  do not wire Wave 3E, or use Wave 3E evidence in Wave 3D, until the producer,
  association/rate state, and sequential nuisance model are independently
  reviewed;
- perform recorded replay scoring only after approved inputs and the final
  processor exist;
- keep Gate 1 recentering as an exclusive, separately authorized live stage.

Wave 4, after conservative completion:

- metric pose/VIO;
- local map/lookahead;
- MPC/MPCC comparisons;
- simulator-only course learning/ILC;
- sim-to-real robustness profile.

## Passive simulator findings from 2026-07-18

No reset, arm/disarm, attitude target, graphics, or powered command was sent.

- passive preflight passed at about 31 decoded frames/s;
- six-second census camera rate: 30.36 Hz;
- HIGHRES_IMU: 118.17 Hz;
- ACTUATOR_OUTPUT_STATUS: 95.43 Hz;
- HEARTBEAT: 9.95 Hz;
- race status: 3.990 Hz;
- blocked pose/odometry/gate messages were absent;
- low-load simulator/wall ratio: 1.00006;
- camera timestamps are epoch scale, not reset-relative time;
- the centered square gate appeared about 80 by 81 pixels, supporting fx=fy=320
  and a 90-degree horizontal / 58.7-degree vertical FOV interpretation;
- the simulator produced almost 300,000 camera datagrams in six seconds, of
  which about 292,000 were duplicates;
- the spawn heartbeat armed bit was true despite no actuator demand.

Still requiring controlled evidence:

- final scoring time basis;
- lockstep behavior under load;
- command application tick and saturation;
- exact camera timestamp meaning;
- calibrated FOV/extrinsics;
- pass-plane and collision-envelope details;
- gate-credit delay distribution.

## Immediate next actions

The recorded M0, Wave 1A, Wave 1, Wave 2, Wave 3 control-plane dogfood, Wave 3
IMU, Wave 3B generated runtime, Wave 3C correlated-coast, and Wave 3D
stable-reference and Wave 3E local-measurement integration commits, clean-main
checks, trusted-manifest verification, and explicit canonical test tiers are
complete.
Do not repeat them merely because an agent resumes the plan; rerun affected
gates whenever the base or trust set changes.

1. The reviewed local timestamp/source seam for attitude and the Gate 0 pitch
   basis plus bounded rotation-only derotation evidence is integrated at
   `ecaa794aeaed87a169b7b87b284d1440f1768a28`. Keep the corrected ray
   standalone; Wave 3D proves only the distinct local-feature coordinate
   transform, and Wave 3E proves only a reducer after an external calibrated
   full-homography producer. Neither is an honest `/1` conversion or filter
   input.
2. The exact generated observation/IMU-to-proposal scheduler composition is
   integrated at `28b7d782404d6b825cebae3b65a8443d756be234`. Keep it
   distinct from receiver/reassembly, recorded replay, and measured
   command/actuator/gyro response evidence.
3. The separately reviewed Wave 3C one-tick correlated-coast lease is integrated
   at `168220ba7060d07743335d0e9c56bcd2d05d669d`. Keep it default-off outside
   its proof-bound generated runtime, preserve public lower-layer dropout
   rejection, and do not treat its local proof as supervisor-verifiable.
4. Validate Gate 0 non-regression and recorded top-clipped Gate 1 geometry once
   approved replay inputs and the final processor exist.
5. Wave 3D stable-reference math is integrated at
   `46df0adee76070e10509fa5e807b986a9469c68e`, and the standalone Wave 3E
   homography-to-local-measurement reducer is integrated at
   `16dd5c84995cafb5158e277d730e670557ba69f2`; neither has a production call
   site. Before using Wave 3E evidence in a filter, supply and independently
   review calibrated rectification and a dense full-homography covariance
   producer, approved replay, cross-frame association/rate state, sequential
   nuisance handling, calibrated production timing/extrinsics, and
   estimator/runtime integration and authority review.
6. The positive exact `SingleMerger` path now has unit evidence. Exercise a
   disposable clean candidate through T0-T4 scheduler leases,
   interruption/resume, deduplication, exact worktrees, and merger evidence
   only after the three real T1 prerequisites exist; this operational dogfood
   remains outstanding and does not broaden tier claims.
7. Provision the private golden replay corpus and administrator-owned isolation
   wrapper before enabling production T1. This may proceed independently when a
   slot and the required authorization are available; it does not block the
   remaining synthetic/offline work.
8. Continue only authorized passive probes while offline work proceeds.
9. Integrate finished branches frequently, but only after each branch is committed
   and promoted. Quarantine a dirty failed worktree without touching clean main.
10. Keep the first live change a separately reviewed, explicitly authorized,
   bounded Gate 1 recenter stage. Treat every powered system-identification or
   later Gate 1 stage as a separate,
   explicitly authorized workflow outside the shipped scheduler. Never reinterpret
   synthetic T2-T4 results as FlightSim evidence.

## Open authorization item

The replay tooling requires an explicit attestation before storing full decoded
competition frames. Confirm organizer permission before building the private
golden corpus. Without it, use existing approved captures, isolated stills, and
synthetic/derived test data.

## Definition of program completion

The program is complete when:

- main is clean and contains only reviewed, committed integrations;
- a frozen candidate completes the full VQ2 course in correct sequence;
- completion is repeatable across the chosen reliability campaign;
- every powered result proves reset, countdown/GO, freshness, watchdogs,
  collision/sequence state, and cleanup;
- simulator-only priors are optional and the mapless fallback remains functional;
- performance candidates are promoted by paired evidence rather than a lucky lap;
- qualification artifacts, dependencies, calibration, configuration, and
  disclosures are frozen and reproducible.
