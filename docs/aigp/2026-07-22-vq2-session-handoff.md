# VQ2 session handoff — 2026-07-22

Start here. Do not attempt to recover or rerun `F02-A01`.

## What actually happened

- `origin/main` was fetched and is still `7be5111199161adbf545fd4e58034a31bd2f3a0a`.
- Local `main` started at `1479a85827bca72613b63bfecc81a395136cb36a`, ahead of the remote. Its flight runtime was identical to the tested F02 candidate except for F02 identity metadata.
- FlightSim build 3385 was left running. Last verified topology was launcher PID `7072`, payload PID `26176`.
- `F02-A01` entered the production wrapper but failed before simulator launch/contact, port binding, reset, arm, target, or powered commands.
- The exact failure is a real code bug: `Win32ProcessOperations.query_process_identity()` hashed the 91,968,000-byte payload with `stable_file_identity()`'s 64 MiB default ceiling. It raised `StableFileError` during `launcher_return`.
- F02 is terminally invalid and poisoned. Preserve it exactly:
  - evidence root: `C:\Users\John\aigp-evidence\2026-07-21-package2-f02-powered-calibration-attempt`
  - `attempt-invalid.json`: `d64b943184796c10bc91cdc257f57e5cfab16765de8d92a9e6889319604c481b`
  - `live-poison.json`: `63a215a250bedde84cfdd2714745e7762b2b97e27ed0a2782e4283592e9c2005`
  - wrapper lifecycle: `c82528d2ed4bd766823b74694b9ba04e8c57a413f0b765a48c30593dc0a6d71a`

## Current working tree

The tree is intentionally incomplete and uncommitted. Inspect the diff before editing.

- Implemented the process-image fix in `scripts/aigp_vq2_powered_runtime.py`:
  - `MAX_PROCESS_IMAGE_BYTES = 128 * 1024 * 1024`
  - process identity passes that explicit limit while the generic 64 MiB default remains unchanged.
- Added a portable regression test in `tests/test_aigp_vq2_powered_runtime.py`.
- That exact regression test passed: `1 passed`.
- Began rolling the four production identity modules from F02 to F03:
  - `scripts/aigp_vq2_calibration_target.py`
  - `scripts/aigp_vq2_powered_attempt.py`
  - `scripts/aigp_vq2_powered_calibration_analysis.py`
  - `scripts/aigp_vq2_powered_calibration_probe.py`
- Matching test, policy, manifest, documentation, commit, worktree, and freeze updates are not complete.

## Clean next step

Finish one fresh F03 candidate; do not add another recovery state machine.

Use these identities:

```text
task:      vq2-package2-f03-powered-calibration-attempt
session:   F03
attempt:   F03-A01
root:      C:\Users\John\aigp-evidence\2026-07-22-package2-f03-powered-calibration-attempt
worktree:  C:\Users\John\aigp-worktrees\wt-package2-f03-powered-calibration-attempt-live
task name: AIGP-P2-F03-A01-Launch
plan ID:   vq2-build3385-training-f03-excite-v1
```

Unchanged waveform hashes with the F03 plan ID:

```text
object: 73a1134906edeb6480e189cdf9df1d9d30eac697537d3da44d5d0de075237e7b
file:   d69ea76c6f3d4d44b8ec17c70e813eb79fe8530e995ced4e65a91a77333c221a
```

Complete the four matching test rollovers plus the runtime regression, update `config/t1_pytest_policy.json` from 2170 to 2171 expected passes, refresh affected promotion hashes, run directly affected tests and `test-vq2`, then create a fresh F03 commit/worktree/freeze.

The production wrapper natively validates and adopts an already-running exact simulator. Do not close FlightSim merely to satisfy the old optional external absence checkpoint.

External F02 debug launch files are under:

```text
C:\Users\John\aigp-review\2026-07-21-package2-f02-powered-calibration-attempt
```

They are not part of the candidate and should not be reused for F03.
