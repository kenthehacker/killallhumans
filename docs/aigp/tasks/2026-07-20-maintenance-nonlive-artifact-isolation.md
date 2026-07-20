# Non-live telemetry-artifact isolation

- Task ID: `vq2-maintenance-nonlive-artifact-isolation`
- Parent: `2026-07-20-vq2-development-continuation-handoff`
- State: `committed`
- Objective: keep the bounded VQ1 dry-run slow test from writing an ignored
  timestamped telemetry capture into the repository while preserving the
  runner's intentional default recording behavior.
- Explicit stop condition: stop after the test-only artifact isolation is
  promoted, integrated, and post-merge verified. Do not continue into runner
  semantics, replay, capture, simulator, runtime, transport, perception,
  estimation, control, or live work.
- Starting main commit:
  `8472869264e70d0a3c06890423fc80b7af94ff59`.
- Branch: `maintenance-nonlive-artifact-isolation`.
- Worktree:
  `C:\Users\John\aigp-worktrees\wt-nonlive-artifact-isolation`.
- Owner: `/root`.
- Heartbeat date: `2026-07-20`.
- Simulator access: `none`.

## Entry audit

- The declared base is the reviewed local `main`, not `origin/main` or the
  earlier `e71d284` closeout reference. `8472869` is the documentation-only
  continuation-handoff child of `e71d284`.
- Tracked status on `main` and the fresh task worktree was empty before the
  task began. The task worktree contained no ignored state.
- The fifteen historical non-main worktrees remain registered and inactive;
  none was reused or removed.
- The main worktree has pre-existing ignored captures in addition to its
  documented bytecode. They are unexplained task-external state and will not
  be modified, deleted, copied into the candidate, or treated as evidence.
- The fresh task worktree's initial `captures/` inventory contained exactly
  the five tracked historical telemetry files. Its canonical sorted
  `relative-path|size|sha256` inventory SHA-256 was
  `d8beecdd3abc1c8b9668b00bceb8435e91f1bad26593ab103f2090a79cdd501d`.

## Ownership and exclusions

Behavioral ownership is limited to:

- `tests/test_aigp_vq1_runner.py`; and
- this task record.

After behavioral acceptance, the integration owner alone may update:

- `config/promotion_trusted_files.json`.

This task explicitly excludes:

- `scripts/aigp_vq1_run.py` and its default recording semantics;
- `config/t1_pytest_policy.json` and all discovery/count policy;
- every file under `captures/`, including tracked historical captures;
- dependencies, schemas, calibration, private data, replay corpora, and
  generated dependency inventories;
- VQ2 production/runtime/controller/supervisor/transport wiring; and
- preflight, simulator launch or connection, reset, arm/disarm, targets,
  shadow stages, and all powered or live surfaces.

No new dependency, schema, calibration input, replay input, private capture,
or simulator authority is required or authorized.

## Artifact and cache isolation

- Task evidence root:
  `C:\Users\John\aigp-worktrees\artifacts\maintenance-nonlive-artifact-isolation`.
- External cache root:
  `C:\Users\John\aigp-worktrees\.artifact-cache\maintenance-nonlive-artifact-isolation`.
- Python: the reviewed
  `C:\Users\John\killallhumans\.venv\Scripts\python.exe` through
  `AIGP_PYTHON` and the canonical `scripts\dev.cmd` surface.
- Test telemetry: a pytest-managed `tmp_path` outside the worktree, passed
  explicitly as `record=str(record_path)` and checked as a nonempty temporary
  artifact.
- Canonical launcher bytecode and pytest caches remain process-scoped outside
  the repository. Promotion candidate-local caches or bytecode are not
  permitted; the development worktree is not a strict promotion candidate.

## Frozen behavioral change

Change only `test_dry_run_full_flow` so it accepts `tmp_path`, passes an
explicit temporary `.jsonl.gz` record path to `run_vq1`, and validates the
temporary telemetry artifact. Do not change the runner's fallback path or any
production behavior. Repository `captures/` inventories taken before and
after the target test, slow tier, and promotion-boundary full non-live tier
must be identical and contain no new ignored telemetry.

## Verification and acceptance

Required gates, in order:

1. `test-target tests/test_aigp_vq1_runner.py::test_dry_run_full_flow`;
2. `test-slow`;
3. `test-fast`;
4. `test-unit`;
5. `test-vq2`, still exactly `1,325` passes;
6. one fresh exact promotion worktree running `test-full-non-live` with no
   repository-local capture delta; and
7. a separate fresh exact, physically audited worktree running the isolated
   hash-pinned VQ2 suite, followed by post-merge `test-vq2` on clean `main`.

Directly affected evidence is the target node. Compatibility evidence is the
complete slow tier. Broad evidence is `test-fast` plus `test-unit`. Canonical
VQ2 evidence is `test-vq2`; the full non-live tier is promotion-only.

Acceptance additionally requires:

- only the slow test changes behavior;
- capture inventories remain exactly equal before/after each artifact-sensitive
  run;
- no production, live, replay, simulator, or network surface is invoked;
- independent API/test and promotion/trust review;
- clean committed candidates with no ignored executable or extra physical
  state; and
- exact trust-delta and pass-count arithmetic.

Observed pre-promotion evidence:

- affected target: `1` passed in `1.28s`;
- slow compatibility tier: `2` passed, `2,480` deselected in `3.96s`;
- `test-fast`: `2,420` passed, `20` skipped, `42` deselected in
  `99.96s`;
- `test-unit`: `2,420` passed, `20` skipped, `42` deselected in
  `99.89s`;
- canonical `test-vq2`: exactly `1,325` passed in `32.83s`; and
- capture inventories before and after the successful target and complete
  slow-tier runs remained exactly five files at
  `d8beecdd3abc1c8b9668b00bceb8435e91f1bad26593ab103f2090a79cdd501d`.

Independent behavioral/API and lifecycle/authority reviews found no scope,
production-semantics, replay, simulator, live-authority, or test-design
blocker after task-record corrections. The development worktree contains the
explained pytest cache produced by these runs and is not eligible as the
strict promotion candidate; promotion uses new exact worktrees.

## Frozen trust delta

The VQ2 policy must remain byte-identical with file SHA-256
`7daa46ec4dfd025c18f12076add06d70b6463f07d6320b20487a63bd78d0851e`,
canonical JSON SHA-256
`b8bc5228b12eafc75c10b3d2aa658cfe57a0d1ed820b3fefa6e0317d7c5cdc90`,
31 sorted unique test paths, and expected count `1,325`.

The trusted manifest must remain at 129 sorted, unique, case-safe paths. Only
the mapping for `tests/test_aigp_vq1_runner.py` may change from
`af3612764af47f7e645893c90e2f8688a02f74662fa5c7713dc712a42a32ccb8`.
The tested replacement file SHA-256 is
`977f2431aaa07b762eab7888451f0b6aa82dc5aa6f387d940d3862d3ecb9cf07`.
No path may be added or removed. The new manifest file and canonical JSON
identities must be recorded after independent `129/129` rehash verification.

## Lifecycle evidence

- Contract/task-record commit:
  `060fd479988cf1214039f1618c7fc4f4d083e44d`.
- Behavioral commit: `5a9fa4ae231fec4ede4476157b72b090249a59a9`.
- Promotion/trust commit: pending.
- Integration commit: pending.
- Post-merge verification: pending.
- Result: committed; all required pre-promotion gates passed and the bounded
  behavior is frozen for trust promotion.
- Failure provenance: the first affected-target attempt failed while opening
  the explicit record because its pytest `tmp_path` directory disappeared
  during the run. The repository capture inventory was unchanged. With no
  intervening code edit, the identical command passed on immediate rerun with
  the same unchanged inventory; this is retained as a transient shared-host
  temporary-directory event, not accepted pass evidence.
