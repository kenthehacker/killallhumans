# Package 2 import/environment recovery

- Task ID: `vq2-package2-import-environment-recovery`
- Parent: `vq2-package2-powered-calibration-pilot`
- State: `F01-A01 consumed and terminal-invalid before simulator launch; immutable
  poison requires a new reviewed recovery task and forbids retry or clearing`
- Starting main commit:
  `e8d1f339c949ea4ab83721746a7a42eda43a1c2e`.
- Branch: `package2-import-environment-recovery`.
- Worktree:
  `C:\Users\John\aigp-worktrees\wt-package2-import-environment-recovery`.
- Owner and integration owner: `/root`.
- Heartbeat date: `2026-07-21`.
- Simulator target: FlightSim build 3385, Training mode.
- Simulator access: `none` through implementation and promotion; conditional
  `powered` for one `calibration-excite` attempt only after this exact
  correction/identity rollover is integrated, its new private root and freeze
  are independently reviewed, and every L0/L1 entry gate below passes.
- Contract-freeze commit: `ea728904c4a4761b677d7d118265a595d5b0d5ac`.

## Failure provenance and authority

The predecessor task's sole `F00-A01` was consumed and poisoned. It is
immutable evidence and must not be cleared, renamed, retried, or used as the
authority for this task. The predecessor failed before scheduled-task query,
simulator enumeration or launch, fixed-port contact, transport, child
creation, frame capture, reset, arm/disarm, target, or powered-command contact.

The deterministic cause was import ordering. Offline admission first matched
the frozen native environment. Import revalidation then loaded
`scripts.aigp_vq2_run`, whose pinned OpenCV loader prepended its DLL directory
to native `PATH`. The later exact spawn-environment check correctly rejected
the changed map after the predecessor attempt and lease had already been
created.

On 2026-07-21 the user explicitly requested this recovery, a fresh
attempt/freeze, and testing in the actual simulator. That instruction is
recorded as conditional authorization for at most one successor bounded
`calibration-excite` session under this newly reviewed exact task/freeze. L0
must bind that instruction into a new simulation-only capture authorization
published create-new in this task's new private root. The authority permits
only task-local recording, sealing, bundle verification, acquisition analysis,
and independent integrity review. It does
not authorize reuse of predecessor evidence or identifiers, another attempt,
another powered stage, Gate 0 or Gate 1 steering/passage, a lap, cooked-PAK
access, Package 1 operational replay, training/corpus reuse, cross-task access,
later fitting without a new hash-bound authority, physical/HIL/submission use,
or public release. Private decoded-frame capture remains limited to this
build-3385 Training calibration session. Any semantic or stage change requires
a new checkpoint.

## Objective and stop condition

Correct the wrapper host-boundary defect and perform the minimum mechanical
identity rollover required to make one fresh attempt possible:

1. revalidate the complete frozen import graph in an exact isolated child
   interpreter so imported native loaders cannot mutate the wrapper process;
2. compare the wrapper's complete native environment to the frozen inventory
   one final time immediately before atomic attempt-directory consumption;
3. seal that exact validated mapping for every launcher/task-query, powered-
   child, and cleanup-fallback subprocess, passing only defensive copies and
   never substituting a later `os.environ` or native-block reread;
4. emit only expected/observed hashes in environment-drift diagnostics;
5. roll every production-bound predecessor task/session/attempt/root/freeze/
   launcher/run/plan identity to the fresh identities frozen below; and
6. prove import-induced or other pre-consumption environment mutation fails
   before attempt, lease, poison, simulator, port, or powered contact.

The implementation phase stops after exact promotion, integration, and post-
merge verification. This same task's L0 boundary stops after the new private
root, identities, deterministic inventories, capture authority, and live
freeze are independently reviewed and published create-new with the freeze
last. The L1 boundary stops after the single bounded session and its cleanup/
evidence result. E0 then validates the exact terminal, seal, bundle, report,
split, and cleanup identities offline and obtains independent acquisition-
integrity review. No result in this task fits data, assesses rank, accepts
calibration, opens successor data access, or opens Gate 1 work. Post-
consumption drift retains the predecessor's fail-closed cleanup, invalidation,
and poison behavior; only failures before atomic consumption are required to
leave attempt, lease, and poison absent.

## Owned and excluded surfaces

Behavioral ownership is limited to:

- this task record;
- `scripts/aigp_vq2_calibration_target.py` and its direct test, only for the
  new non-transferable task/session/private-root capture-authority identity;
- `scripts/aigp_vq2_powered_attempt.py` and its direct test, only for the
  exact identity rollover and unchanged plan/safety contract;
- `scripts/aigp_vq2_powered_calibration_probe.py` and its direct test, for the
  isolated audit, final environment seal, spawn equality gates, launcher name,
  and identity rollover;
- `scripts/aigp_vq2_powered_calibration_analysis.py` and its direct test, only
  for the exact fresh run/split identity; and
- additive compatibility assertions in `tests/test_aigp_vq2_runner.py`,
  `tests/test_aigp_vq2_powered_cleanup.py`, and
  `tests/test_aigp_live_lease.py` only if the mechanically rolled constants
  require them. Generic lease examples and historical build-reference paths
  remain unchanged.

The integration owner may update only the exact corresponding digests and
pass arithmetic in `config/promotion_trusted_files.json` and
`config/t1_pytest_policy.json` after behavioral review. A count change is not
assumed in advance.

The task does not own controller, estimator, detector, runner-stage behavior,
capture schemas, command values/bounds, transport, lease protocol, cleanup
truth table, launcher/payload binaries, nominal target configuration, or the
predecessor private evidence root. It must not whitelist `PATH`, filter a
variable from comparison, accept a post-hoc environment, weaken import-origin
checks, alter the predecessor records, or add a generic powered test command.

## Frozen successor identities

- task: `vq2-package2-import-environment-recovery`;
- session: `F01`;
- sole attempt: `F01-A01`;
- private root:
  `C:\Users\John\aigp-evidence\2026-07-21-package2-import-environment-recovery`;
- attempt directory: `<private-root>\F01-A01`;
- live worktree:
  `C:\Users\John\aigp-worktrees\wt-package2-import-environment-recovery-live`;
- freeze ID:
  `vq2-package2-import-environment-recovery-f01-a01-live-freeze`;
- freeze path: `<private-root>\live-freeze-F01-A01.json`;
- scheduled task: `AIGP-P2-F01-A01-Launch`;
- run ID: `F01-A01/reset-epoch-1/excitation-1`;
- split: `discovery_fit`;
- plan ID: `vq2-build3385-training-f01-excite-v1`; and
- plan canonical-object SHA-256:
  `e3871b782c72bfafdbefd8d2c5138ca008311b61903bfb5b9f5e8e9cc3a63cab`.

The plan changes only its identity from the predecessor; all stage, tick,
timing, command, segment, and safety values are byte-semantically unchanged.
No old F00 identifier or root is admissible to this production attempt.
The predecessor task's proposed future names `F01-F04` are retired and
replanned: this `F01` is permanently assigned to replacement discovery-fit,
is not a post-design collection session, can never be relabeled, and no later
collection may reuse it.

The exact new capture-authority identity is
`conversation-2026-07-21-package2-f01-sim-capture`, authorized on `2026-07-21`
with source `direct_user_instruction`. Relative to the predecessor authority,
only that authority ID/date, `task_id`, `session_ids=["F01"]`, and
`storage.private_root` change. Domain, build/mode, allowed purposes/classes,
all storage booleans, retention, transfer, organizer-credential, and
publication values remain byte-semantically identical. In particular the
task-local `offline_replay_and_analysis` and `derived_replay_and_analysis`
labels authorize only the L1/E0 bundle and acquisition checks described here;
they grant no Package 1 operational replay or successor use.

## Frozen correction contract

The production import revalidation command is exactly the frozen interpreter
with `-E -s -B -m scripts.aigp_vq2_powered_import_audit`, the exact candidate
working directory, no shell, and the prevalidated native environment. It is
deadline-bounded, heartbeat-cooperative, output-bounded, and deterministically
terminates/reaps on failure. Its canonical result must equal the frozen
`python_sha256`, seeds, and complete entries; the isolated derivation itself
proves exact roots, origins, user-site absence, and absence of extra candidate
or venv modules. The offline audit subprocess has no private-root/publication,
simulator launcher, live mutex/lease, socket, or fixed-port provider. The
parent import graph and native environment must remain unchanged across the
call.

The final environment gate is a required spawn-service method called after
attempt material and handles exist but immediately before
`AttemptWorkspace.consume`. It reads the native Windows block once, validates
the complete sorted name/value-hash inventory and the frozen launcher
environment digest, stores an internal defensive copy, and rejects a second
seal. Every later launcher/child/fallback spawn requires the seal and receives
only a defensive copy of the same mapping. Scheduled-task queries and every
other subprocess in the live boundary also derive `SYSTEMROOT` and their
complete environment from that seal, never from `os.environ`. The existing
later offline revalidation before child/fallback capability release still
compares the then-current native environment exactly; sealing is not a waiver
for later drift, and detected post-consumption drift follows normal
invalidation/cleanup/poison handling.

Failures before `AttemptWorkspace.consume` propagate as an unconsumed offline
failure. Cleanup terminates and reaps the offline audit subprocess; zeroizes
the lease-owner, child, and cleanup capability buffers; and closes every
allocated pipe, process, job, output, and retained-wrapper handle. It creates
no attempt directory, wrapper ledger, lease, poison, simulator process,
powered child/fallback, or fixed-port contact. No secret or environment value
may enter diagnostics.

## Verification and review gates

After each behavioral file edit, run its directly corresponding target; the
probe command is:

```powershell
$env:AIGP_PYTHON = 'C:\Users\John\killallhumans\.venv\Scripts\python.exe'
.\scripts\dev.cmd test-target tests/test_aigp_vq2_powered_calibration_probe.py
```

Before behavioral acceptance, run this exact affected matrix:

```powershell
.\scripts\dev.cmd test-target tests/test_aigp_vq2_calibration_target.py tests/test_aigp_vq2_powered_attempt.py tests/test_aigp_vq2_powered_calibration_probe.py tests/test_aigp_vq2_powered_calibration_analysis.py tests/test_aigp_vq2_powered_cleanup.py tests/test_aigp_vq2_runner.py tests/test_aigp_live_lease.py
.\scripts\dev.cmd test-vq2
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
```

Obtain independent import-isolation, environment/spawn, failure-ordering,
identity/capture-authority, powered-authority/cleanup, and test reviews. Commit
behavior before promotion metadata.

At the promotion boundary, update only exact trusted metadata, rehash every
trusted path, and prove collection/pass arithmetic. From two separate fresh
exact worktrees at the unchanged candidate, run `test-full-non-live` and the
isolated hash-pinned VQ2 suite with:

```powershell
$env:AIGP_CANDIDATE_WORKTREE = (Get-Location).Path
C:\Users\John\killallhumans\.venv\Scripts\python.exe -I scripts\aigp_pytest.py vq2
```

Inventory every physical side effect in each promotion worktree. Integrate
that exact candidate, run post-merge `test-vq2`, and prove tracked `main`
clean. Automated tests use only injected kernels/transports, ephemeral ports,
and unique nonproduction mutex names. They must never bind or send through UDP
14550/5600, open or acquire `Global\AIGP-FlightSim-LiveLease-v1`, launch/query
FlightSim, or read/write either private evidence root.

## Successor L0/L1 gates

L0 must use the exact fresh identities above, capture authorization binding
the user's 2026-07-21 instruction, attempt limit one, detached physically
pristine live worktree at the exact integrated commit, and exact Python,
PowerShell, development lock, launcher script, launcher, payload, target
configuration, and plan identities. It requires two isolated audit derivations
with byte-semantic equality; exact `PYTHONNOUSERSITE=1` and
`PYTHONDONTWRITEBYTECODE=1`; absent `PYTHONHOME`, `PYTHONPATH`, and
`PYTHONSTARTUP`; exact implementation/environment/import inventories;
exclusive lease/fixed-port rules; phase deadlines; outbound allowlist; and
cleanup/invalidation rules. Independent reviewers must bind the one exact
no-selector wrapper argv and hashes. All root files are create-new and the
freeze is last. Every ancestor/final path component must pass exact-path,
local fixed-volume, non-reparse, retained-handle, final-path, and volume checks.
The new private root and its private directories/files must additionally be
current-user owned with inheritance disabled and the frozen current-user-only
effective DACL; every private file must be one-link with stable readback. No
unexplained entry is permitted. The new attempt and poison must be absent
before L1.

L0 also hash-binds the immutable predecessor attempt-invalid
`c9dbd9e60a940279deeb3052b5e1d763809a8815d54d9e04b2b1e18d6eec47e6`
and poison
`a725abc8f89696f398b73d97a949a817efdda195447bb46fee358788137b8fd0`
as failure provenance without repairing either. Immediately before the new
attempt it proves current OS absence of the old wrapper, child/fallback,
scheduled task, FlightSim/payload, and fixed-port owners. That observation
never retroactively supplies the predecessor's missing lease-release proof;
the new live gate must independently acquire and prove its own exclusive
lease.

L1 is only the exact F01 plan above: 245 absolute ticks at 20 ms (4.9 s
nominal, 5.0 s hard expiry), thrust `0.235`, yaw rate exact zero, and these
nonzero roll/pitch segments: ticks 30-44 `(+0.08,0)`, 54-73 `(-0.06,0)`,
86-105 `(0,+0.07)`, 116-133 `(0,-0.08)`, 150-164 `(+0.06,+0.04)`,
165-179 `(-0.06,+0.04)`, 180-194 `(-0.06,-0.04)`, and 195-209
`(+0.06,-0.04)` rad/s; every other tick is zero-rate. Missed ticks drop and
never replay; pacing is at most 50 Hz.

Before power it must prove exact build-3385/Training topology plus the local
interactive Training challenge, exclusive lease and free ports, same loopback
peers, stopped vision during reset, race and IMU rollback with at least two
advancing samples, restarted vision only in that epoch, normalized/proved
disarmed state, countdown and GO+150 ms, stable actual 640x360 decoding, and
three stable target frames. Arm/disarm confirmation uses only a newer
heartbeat. Before every send it rechecks parent, lease, deadline, fresh
heartbeat/actuator/camera/target/capture health, advancing IMU/race data,
estimator, collision, gate index zero, and the command envelope. Any gate-index
change, collision (with no launch-pad exception), second source, unknown
outbound, drop/overflow, target loss, source/lineage/identity change, corridor/
size/area violation, or stale/invalid evidence aborts. Target center stays
inside closed `[0.10W,0.90W] x [0.10H,0.90H]`; bbox width and height stay at
most 160 px and area at most `2*A0`, where `A0` is the first three-frame-
confirmed bbox area immediately before arm. Ordinary in-bounds bbox motion is
not an identity change. Roll and pitch excursion from start stay at most 0.05
rad.

Allowed outbound category literals remain exactly `arm`, `attitude_target`,
`disarm`, `gcs_heartbeat`, `sim_reset`, and `timesync`; unknown categories
invalidate. Completion or abort permanently latches production, sends exact-
zero rate and zero thrust when required, then uses newer-heartbeat disarm,
reset, clean advancing race/IMU epoch, final disarmed state, closed vision/
MAVLink/workers/handles, child-tree/port/task/topology proof, and proved lease
release. Any fallback use invalidates the collection even when cleanup
succeeds; any cleanup failure fails the stage. It never approaches or passes a
gate and has no crossing-confirmation path.

E0 opens only the sealed F01 artifacts needed to validate the attempt,
terminal/poison choice, lifecycle, cleanup, capture seal, replay-bundle
integrity, acquisition report, and immutable discovery-fit split. Independent
review must clear provenance, safety accounting, cleanup, and acquisition
integrity before task close. F01 remains sealed quarantine afterward; fitting,
rank/identifiability claims, held-out design, and any successor access require
a separate new hash-bound authority and task.

## Escalation conditions

Stop for review if the isolated audit differs from the frozen graph, the
parent native environment changes, the final seal does not exactly match the
freeze, any spawn can bypass the seal, a pre-consumption failure creates
attempt/lease/poison state, trusted counts or hashes drift unexpectedly, the
candidate is not physically pristine, L0 would reuse predecessor state, the
simulator is not exact build 3385 Training, or any safety/cleanup gate fails.

## F01-A01 terminal result

The import/environment correction was promoted at
`0970f4b74c8529fb6fcd72d37c765d7d18025c3a`. Its fresh L0 publication and
pre-L1 recovery succeeded; the final pre-L1 attestation is 3,962 bytes with
SHA-256 `92cf96b9a0947df648c9c8abd6888be928841fda1aabd3d8aae17f4eca3a1137`.
The sole authorized F01-A01 wrapper was then consumed once.

F01-A01 failed `launcher_return` in about 0.278 seconds because the frozen
`launch_sim.ps1` rejected `query.exe session` exit code `1`. On this exact host,
that command emitted a complete table with one current row proving
`console / John / session 1 / Active`, despite the status. The launcher failed
before task creation and before its first mandatory 500 ms post-run wait.

No powered child or fallback, transport, capture, replay, reset, arm/disarm,
target, or flight command exists. Passive post-exit observation found the
wrapper, FlightSim/payload, exact scheduled task, and UDP 14550/5600 owners
absent. That later cleanliness cannot replace the missing contemporaneous
post-launch topology and lease-release proof. The immutable terminal identities
are:

- attempt `e939c1f35e5a41ff350a7319f3f7f8c22b4a733c9104b428682e1ac38d12eab5`;
- lifecycle `218fe368f6971ab309be4c09c7a87e4d86b9d49624590641fe7d4a5083bc880f`;
- attempt-invalid `edf2424bbf60fb305fc805a28b561fa37982c77e9f8321ed35c91305e3dbaeb0`;
- poison `009edcb1b7f48d120c1f46b393a9094073e378ed122d3ac75c2a565a65fae91d`;
- last lease generation `9e69def312f13f7b90b5a265851e0f83f91ec8419780f5df37836c41286fb898`.

The lease chain has acquisition and heartbeats but no release intent, released
generation, or final lease record. Cleanup therefore failed under the VQ2
safety contract. F01-A01 and its poison are permanent evidence. A successor
must use a new task, attempt, private root, freeze, review, and explicit powered
checkpoint.
