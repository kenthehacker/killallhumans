# Non-live telemetry-artifact isolation

- Task ID: `vq2-maintenance-nonlive-artifact-isolation`
- Parent: `2026-07-20-vq2-development-continuation-handoff`
- State: `post_merge_verified`
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
  the five tracked historical telemetry files. Its canonical ordinal-sorted
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
- Canonical launcher bytecode remains process-scoped outside the repository.
  Pytest may create an explained `.pytest_cache` in development or full-suite
  worktrees. Candidate-local caches or bytecode are forbidden at the pre-run
  exactness audit. An audited post-run cache disqualifies that physical tree
  from reuse; the separate strict candidate must remain cache-free before and
  after its run.

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
No path may be added or removed.

The canonical builder first published to an external review path and then to
the tracked path. Independent strict audits of both outputs matched all
`129/129` direct regular files and found exactly the one frozen digest change,
with no addition or removal. The tracked manifest identities are:

- file SHA-256:
  `3855243e7b3675ebff14731bbd073b7850bb87fb9d9d35267b7ca0fa2982d08f`;
  and
- canonical JSON SHA-256:
  `ac2700e5cfed1c9aece92446d7aef665ddfff923d790e62628c35cbbbf4978a2`.

These identities freeze the candidate trust metadata but do not replace the
fresh full non-live and isolated hash-pinned promotion runs.

## Promotion and integration evidence

The complete immutable promotion candidate is
`ef92041bb3f05b1d8f3ef69182db8d51184c9cce`. A fresh detached worktree at
that exact commit passed the promotion-boundary `test-full-non-live` tier with
`2,461` passes and `21` skips in `483.73s`. Its `captures/` inventory remained
exactly the five tracked files at
`d8beecdd3abc1c8b9668b00bceb8435e91f1bad26593ab103f2090a79cdd501d`.
The only candidate-worktree physical side effect was an eight-entry
`.pytest_cache` tree; every path and file digest was inventoried, tracked
status remained empty, and that worktree was not reused as the strict
candidate.

A separate fresh detached worktree at the same unchanged commit passed the
pristine lexical/physical audit, contained `807` non-`.git` physical entries,
and passed the isolated hash-pinned VQ2 suite with exactly `1,325` tests in
`34.70s`. Its physical inventory remained exactly `807` entries before and
after, with empty tracked and ignored status.

Local `main` remained at the declared starting commit and tracked-clean until
the exact candidate fast-forward. It then advanced directly from `8472869` to
`ef92041`. Post-merge canonical `test-vq2` passed exactly `1,325` tests in
`34.46s`; tracked status was empty. The main worktree's pre-existing 45-file
capture inventory remained byte-identical at aggregate SHA-256
`e3ece19f6b58b235d8c78b8041c287939efd0f6c29bb0072935271336aed747e`.
That identity is SHA-256 over newline-joined UTF-8
`relative-path|size|sha256` rows in ordinal relative-path order, with no
trailing newline.

No FlightSim process was launched or contacted, and no preflight, external
network, replay corpus, private/full-frame/live/repository-local capture,
reset, arm/disarm, target, transport, shadow/runtime, or powered action
contributed to this task. The temporary synthetic pytest artifact is the only
new telemetry file created by the accepted test.

## Lifecycle evidence

- Contract/task-record commit:
  `060fd479988cf1214039f1618c7fc4f4d083e44d`.
- Behavioral commit: `5a9fa4ae231fec4ede4476157b72b090249a59a9`.
- Behavioral-acceptance record:
  `1fb409706893c8d00c8d2d1ed516569c97692574`.
- Promotion/trust commit:
  `ef92041bb3f05b1d8f3ef69182db8d51184c9cce`.
- Integration commit:
  `ef92041bb3f05b1d8f3ef69182db8d51184c9cce` (exact fast-forward).
- Post-merge verification: exactly `1,325` VQ2 tests passed in `34.46s` with
  empty tracked main status and no capture-inventory change.
- Result: `post_merge_verified`; the bounded maintenance stop condition is
  reached. Default runner recording, production code, VQ2 policy/count, trust
  path inventory, and historical captures remain unchanged. The task improves
  promotion hygiene only and advances neither M1, M2, nor M4. No further
  implementation task is authorized by this record.
- Failure provenance: the first affected-target attempt failed while opening
  the explicit record because its pytest `tmp_path` directory disappeared
  during the run. The repository capture inventory was unchanged. With no
  intervening code edit, the identical command passed on immediate rerun with
  the same unchanged inventory; this is retained as a transient shared-host
  temporary-directory event, not accepted pass evidence.
