# AIGP VQ2 development continuation handoff — 2026-07-20

This is the operational starting point for development after Wave 3E. This
document alone grants no simulator, capture, replay, production, transport, or
powered authority. The exact Package 2 task contract records only the
conditional simulator/private-capture authority and single powered pilot; it
grants no replay, production, or transport authority.

The current integrated local-main baseline is
`f33e4361fd98c558d046879e5762b94600a9a3bb`. It contains the first promoted
Package 2 powered-calibration implementation. L0 stopped before publishing a
live freeze, creating attempt A01, contacting FlightSim or fixed ports, or
capturing a frame because two pre-live integrity gaps remained. Correction
behavior commit `e87c5e6` closes those environment/import-freeze gaps and is
under exact-candidate promotion in this successor. The earlier Package 2A /
Package 3B passive tranche at `cb2b1decdfe7d2fd82de803fc1510ee569cc4ef0`
remains accepted historical evidence and must not be repeated.

## Precedence and required reading

Use this order when resuming:

1. `Agents.md` and the user's current explicit authorization.
2. `docs/aigp/2026-07-18-vq2-handoff.md` for build-3385 live state, the
   verified interface, and the safety contract.
3. This handoff for current development sequencing and resume gates.
4. `docs/aigp/tasks/2026-07-20-package2-powered-calibration-pilot.md` for the
   exact approved scope, executable contract, evidence, and stop conditions.
5. `docs/aigp/2026-07-18-vq2-execution-plan-handoff.md` for program history,
   ownership, milestones, and orchestration.
6. `docs/aigp/vq2_local_differential_measurement.md` and
   `docs/aigp/tasks/2026-07-19-wave3e-local-differential-measurement.md` for
   the Wave 3E contract and accepted evidence.
7. `docs/aigp/durable_improvement_loop.md` for T0-T5 trust, replay, promotion,
   isolation, and campaign semantics.

Stop for human review if these sources conflict. Verified build-3385 VQ2
findings override older public VQ1 pose/map assumptions. External research may
inform a design, but it cannot replace empirical build-specific evidence.

## Resume in one pass

1. Read `git status --short --branch`, `git rev-parse HEAD`, and
   `git worktree list --porcelain`; do not mutate unexplained state.
2. Confirm the base is the current reviewed local `main`, not stale
   `origin/main` or an old feature worktree.
3. Inventory which concrete entry gate below is actually satisfied.
4. If no new data, calibration, replay infrastructure, or simulator authority
   exists, report the exact missing prerequisite and stop behavioral
   development. The bounded telemetry-artifact isolation task is complete.
5. If an entry gate is satisfied, freeze one package contract in a fresh
   worktree, record its
   access level and non-goals, obtain the required reviews, and implement only
   that package.
6. Stop when its exit boundary is reached. Do not silently continue into a
   filter, runtime, transport, replay, capture, or live stage.

## Current integrated baseline

- Target: FlightSim build 3385 in Training mode.
- Current reviewed implementation/evidence baseline:
  `f33e4361fd98c558d046879e5762b94600a9a3bb`.
- Package 2 behavior commit:
  `508d43d49c4f2b6704b3d6d51e071157de4ce2a3`.
- Live-freeze-readiness correction behavior commit: `e87c5e6`; its trusted
  metadata and exact-candidate promotion are the active successor gate.
- `cb2b1de` is the Package 2A / Package 3B passive capture-ingress evidence
  closeout and remains a historical passive reference.
- Accepted behavior and live-evidence candidate:
  `a50f4ea0e18b2f5a295fdb1cc94e183734616601`.
- Passive timing task record:
  `docs/aigp/tasks/2026-07-20-package3b-m1-passive-timing.md`.
- Exact distributions, artifact identities, and remaining unmeasured claims:
  `docs/aigp/vq2_runtime_timing.md`.
- Package 2 calibration/readiness contract:
  `d52652452de130e8d3990316e156c340dac5e785`.
- `d526524` is documentation-only evidence that the current passive/static
  inputs fail Package 2 intrinsic, extrinsic, and temporal-alignment readiness;
  Package 2 is not accepted.
- `ef92041` is the bounded non-live telemetry-artifact isolation and trust
  integration.
- Continuation-handoff introduction:
  `8472869264e70d0a3c06890423fc80b7af94ff59`.
- `e71d284` is the documentation-only Wave 3E closeout.
- Wave 3E code and trust integration:
  `16dd5c84995cafb5158e277d730e670557ba69f2`.
- Wave 3E behavior:
  `ceed9c854b0066d4f00d4add796fb968d449593a`.
- Review-driven contract correction:
  `aab44d48a032444faeaf5cd1020e90dc9dbd24ed`.
- Contract freeze:
  `c7dcb612318eb9d26868fa1364c1a027d2b8edcd`.
- Wave 3D closeout:
  `f8b0e4095a15413bf04601bc5264f12842bdbc66`.

Earlier integrated milestones remain recorded in the execution-plan handoff:

- foundation: `b9382da162c1c1e2984288ad7f3cfa7e5a1b11f8`;
- Wave 1A: `a6782cd9dcc34aee94e0f064021399985e0f6839`;
- Wave 1 offline record: `1cf17ea5f4e0a330bee89b0128d30b13657899a2`;
- Wave 2: `8176cbac20ff16bfa4b8c24764596d9366fe98cf`;
- Wave 3 control plane: `ab62cde9464442e4b448f293ba8efd31ad601c27`;
- Wave 3 IMU provenance: `ecaa794aeaed87a169b7b87b284d1440f1768a28`;
- Wave 3B generated runtime: `28b7d782404d6b825cebae3b65a8443d756be234`;
- Wave 3C correlated coast: `168220ba7060d07743335d0e9c56bcd2d05d669d`;
  and
- Wave 3D stable reference: `46df0adee76070e10509fa5e807b986a9469c68e`.

M0 and Wave 1A are complete. Waves 1 through 3E are integrated and
post-merge verified offline, the bounded promotion-hygiene maintenance is
complete, and the passive capture-ingress tranche has accepted 2026-07-20
FlightSim evidence. M1, M2, and M4 runtime acceptance remain incomplete. The
passive receiver evidence is capture-loaded and passive only. Package 2's
current implementation and correction add no live simulator evidence. The
authoritative 2026-07-18 handoff still governs live flight state and every
powered safety rule.

The system is not race-ready. Gate 0 has one credited collision-free pass and
Gate 1 has three-frame post-credit reacquisition evidence, but no control
command has steered toward or attempted to pass Gate 1 and no lap exists.

Local `main` is materially ahead of the cached `origin/main` reference. No
network fetch or push was performed during this closeout. Use the reviewed
local `main`, not `origin/main`, as the successor base; do not reset, rebase, or
pull away the local history.

## Current technical truth

The build-3385 production boundary remains:

```text
UDP JPEG vision + HIGHRES_IMU + race status
                  -> target tracking and IMU attitude estimation
                  -> safety-gated body-rate/thrust commands
```

There is no usable pose or gate-map stream in VQ2. The current fitted-
observation path publishes a heuristic covariance over five image-space
summaries. Their differential has rank at most five over the eight projective
homography degrees of freedom, so current `/1` evidence cannot determine an
honest full-homography or local-scale covariance.

Wave 3D and Wave 3E remain standalone T0 mathematical evidence with no
production call site:

- Wave 3D proves a stable-reference transform for the distinct
  `vq2-local-differential-area-v1` feature semantic. It does not convert frozen
  `/1` finite-quadrilateral scale.
- Wave 3E begins only after an external producer supplies rectified,
  center-gauge-fixed full-homography evidence and a dense gauge-fixed `8x8`
  `CONDITIONAL_FIT` covariance. It proves the first-order local measurement,
  analytic `3x8` Jacobian, conditional-covariance congruence, provenance, and
  integrity boundary.
- No production module produces `VQ2RectifiedHomographyInput`.
- The corrected ray remains outside the current filter.
- Wave 3C remains default-off outside its proof-bound generated runtime.

Do not derive a Wave 3E input from `/1`, fitted corners, aggregate residuals,
border-normalized pixels, heuristic floors, diagonal independence, bootstrap
guesses, or arbitrary homography scale. Separately, do not create a downstream
rate/state by inserting zero rates or naive finite differences. Do not pass
Wave 3E evidence into Wave 3D, a filter, runtime, controller, supervisor, or
transport without the later reviewed contracts below.

## Verification and trust baseline

The current integrated hash-pinned trusted state is Package 2 commit
`f33e436`. Its accepted exact-candidate evidence is:

| Evidence | Observed result |
|---|---|
| Package 2 behavior commit | `508d43d49c4f2b6704b3d6d51e071157de4ce2a3` |
| Trusted-metadata/integration commit | `f33e4361fd98c558d046879e5762b94600a9a3bb` |
| Fresh exact-commit `test-full-non-live` | `3,357` passed, `22` skipped |
| Fresh isolated hash-pinned VQ2 | `2,133` passed |
| Detached post-merge VQ2 | `2,216` passed, `1` skipped |

The live-freeze correction at `e87c5e6` has `246` directly affected passes,
`2,225` canonical VQ2 passes with `1` skip, and `3,325` fast/unit passes with
`21` skips and `42` deselections. Its reviewed 42-test trusted policy expects
`2,142` passes, which a local strict validation achieved. Those are
pre-promotion correction results; the fresh exact-candidate full-suite and
hash-pinned runs, integration, and post-merge gate remain mandatory.

The following table is historical accepted passive-tranche evidence at
`cb2b1de`, not the current trust identity:

| Evidence | Observed result |
|---|---|
| Direct replay/runner review | `225` passed, `4` skipped |
| Canonical candidate VQ2 | `1,480` passed |
| Fresh isolated hash-pinned VQ2 at `a50f4ea` | `1,480` passed |
| `test-fast` | `2,578` passed, `20` skipped, `42` deselected |
| `test-unit` | `2,578` passed, `20` skipped, `42` deselected |
| Development promotion `test-full-non-live` | `2,619` passed, `21` skipped |
| Fresh exact-commit promotion `test-full-non-live` at `cb2b1de` | `2,619` passed, `21` skipped |
| Fresh isolated hash-pinned VQ2 at `cb2b1de` | `1,480` passed |
| Post-merge VQ2 at `cb2b1de` | `1,480` passed |

The exact-commit promotion worktree gained only ignored `.pytest_cache/` and
no source-adjacent bytecode, tracked change, or non-ignored untracked artifact.
The separate isolated VQ2 worktree remained physically clean. The previous Wave
3E/maintenance evidence remains historical context:

| Evidence | Observed result |
|---|---|
| Wave 3E direct | `224` passed |
| Focused compatibility | `450` passed |
| Canonical candidate VQ2 | `1,325` passed |
| Final cache-clean isolated hash-pinned VQ2 | `1,325` passed in `33.50s` |
| Post-merge VQ2 at `16dd5c8` | `1,325` passed in `33.72s` |
| Post-closeout VQ2 at `e71d284` | `1,325` passed in `33.51s` |
| Maintenance affected target | `1` passed in `1.28s` |
| Maintenance slow tier | `2` passed, `2,480` deselected in `3.96s` |
| `test-fast` | `2,420` passed, `20` skipped, `42` deselected |
| `test-unit` | `2,420` passed, `20` skipped, `42` deselected |
| Maintenance candidate VQ2 | `1,325` passed in `32.83s` |
| Maintenance promotion `test-full-non-live` | `2,461` passed, `21` skipped in `483.73s` |
| Maintenance isolated hash-pinned VQ2 | `1,325` passed in `34.70s` |
| Maintenance post-merge VQ2 | `1,325` passed in `34.46s` |

The independent numerical matrix covered `2,000` admitted homographies,
one-ULP boundaries, dense covariance scales from `1e-250` through `1e6`, and
tolerated versus material negative modes. This remains synthetic T0 evidence,
not covariance calibration or replay acceptance.

Historical passive-tranche trust identities were:

| Item | Identity |
|---|---|
| Policy file SHA-256 | `d513a88faddf2050d3f41d69cc425173d2fcc491e0f41df98bdafc1a7b326e0e` |
| Policy canonical JSON SHA-256 | `e3cc46056c16d3e7b09dc2e02fa13ff615f0884614ab671c4db1cbad38d92161` |
| Trusted manifest file SHA-256 | `ce662e46c460bf4930e8a1a22e38e893adf877efc0553a17dd01ed21c406afa7` |
| Trusted manifest canonical JSON SHA-256 | `c1de0131b4ecb6f5563835098c66d29888b06c163ea78006315b7a2ea226220c` |
| Wave 3E test SHA-256 | `683aa081103e6e9ae22281b1e1f573bc821218f57df75cdfa688587b0ad84382` |
| VQ1 runner test SHA-256 | `977f2431aaa07b762eab7888451f0b6aa82dc5aa6f387d940d3862d3ecb9cf07` |

That historical policy contains `36` sorted, unique test files plus `2` discovery inputs
and expects `1,480` passes. The trusted manifest contains `142` sorted,
case-unique paths; independent rehash matched all `142/142` regular on-disk
files. Canonical JSON identities are SHA-256 over parsed JSON serialized with
sorted keys and compact separators.

That `1,480`-test result is not operational T1 replay. The checked-in promotion
command and ladder identity files were examples with placeholders or zero
hashes. At that passive checkpoint, the required production command,
identities, and candidate documents were absent, and the ignored schema-v2
`.aigp-loop/trials.sqlite3` ledger was operationally empty. Those are
historical facts, not a current Package 2 gate assessment.

## Repository, worktrees, and physical hygiene

Git tracked status on `main` was empty at Package 2 correction entry. Multiple
historical, readiness, development, and promotion worktrees remain registered;
their count is not a durable identity. The active correction worktree is
`wt-package2-live-freeze-readiness`. The designated
`wt-package2-powered-calibration-live` worktree remains unused: it has no live
freeze and no consumed attempt. Do not resume or remove another worktree
automatically. Inspect the current facts with:

```powershell
git worktree list --porcelain
git status --short --branch
```

Historical worktrees are evidence, not reusable candidate bases. Git-clean is
also not promotion-pristine: inventory ignored executable/cache state at each
gate and do not rely on a stale exact count. Do not delete unexplained ignored
state merely to make a report clean, and do not use `main` as a strict
promotion candidate. Create fresh exact worktrees for full-suite and
hash-pinned promotion.

At maintenance entry, the main worktree also contained 40 ignored capture
files alongside five tracked historical captures. They were preserved as
task-external state and were not used as evidence. Post-merge verification
proved the complete 45-file inventory byte-identical at aggregate SHA-256
`e3ece19f6b58b235d8c78b8041c287939efd0f6c29bb0072935271336aed747e`.
The digest uses newline-joined UTF-8 `relative-path|size|sha256` rows in
ordinal relative-path order, with no trailing newline.
Do not delete, relabel, or infer provenance for those files automatically.

Use the current deliberately reviewed local `main` containing this handoff as
the base. `f33e436` is the Package 2 implementation/trust integration baseline;
`cb2b1de` remains the passive historical reference. A future task starts as
follows only after one entry gate is actually satisfied, using unique names
confirmed not to exist:

```powershell
$vq2TaskBase = (git rev-parse main).Trim()
$vq2TaskBranch = 'replace-with-unique-authorized-task'
$vq2TaskWorktree = 'C:\Users\John\aigp-worktrees\wt-unique-authorized-task'
git worktree add -b $vq2TaskBranch $vq2TaskWorktree $vq2TaskBase
git -C $vq2TaskWorktree status --short --branch
```

Do not reuse an old worktree without a fresh exactness audit. Promotion
candidates may contain only tracked tree content and the validated `.git`
worktree indirection. Reject ignored modules, source-adjacent bytecode, extra
files/directories, symlinks or junctions, alternate discovery controls, and
mutable candidate-local caches.

Keep private captures, calibration data, credentials, simulator data, trial
databases, candidate configs, and generated dependency inventories outside
Git. Preserve tracked historical captures. Before deleting any ignored
artifact, resolve its exact path, identify its provenance and contents, and
confirm it is disposable.

## Canonical task lifecycle

Future implementation task records use the canonical lifecycle:

```text
queued
  -> leased
  -> active
  -> tested
  -> committed
  -> integration_pending
  -> integrated
  -> post_merge_verified
```

Behavioral and promotion reviews are gates between lifecycle transitions, not
new terminal state names. Contract freezes and review-driven corrections are
immutable evidence commits.

Each task record must include:

- task/parent IDs, objective, non-goals, and explicit stop condition;
- starting main commit, branch, worktree, owner, and heartbeat;
- owned and explicitly excluded files/interfaces;
- dependencies, schema versions, calibration/data prerequisites;
- simulator access: `none`, `passive`, or `powered`;
- isolated artifact/cache roots;
- directly affected, compatibility, canonical, broad, and promotion tests;
- final behavior and promotion commits; and
- result, failure provenance, and post-merge verification.

Three historical Wave 2 child task headers retain stale pre-integration labels:
mapless guidance (`integration_pending`), system ID (`candidate_complete`), and
predictive control (`committed_integration_pending`). Their composite Wave 2
integration is post-merge verified on `main`. Treat those labels as historical
metadata, not active leases or valid resume points.

## Canonical offline development loop

For a worktree without its own `.venv`, point the launcher at the reviewed
development environment:

```powershell
$env:AIGP_PYTHON = 'C:\Users\John\killallhumans\.venv\Scripts\python.exe'
.\scripts\dev.cmd test-target <affected-test-path-or-node-id>
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
.\scripts\dev.cmd test-vq2
```

`test-target` always excludes `live` but deliberately permits an explicitly
targeted slow or benchmark test. Run affected tests after each edit. Run
`test-vq2` for each accepted candidate. Use `test-slow` and `test-benchmark`
only for their explicit tiers. Use `test-full-non-live` only at a promotion
boundary; the current full suite has taken about thirteen minutes, so the
calling shell needs an outer ceiling longer than the suite.

For the final strict candidate only, after verifying a fresh exact worktree:

```powershell
$env:AIGP_CANDIDATE_WORKTREE = (Get-Location).Path
C:\Users\John\killallhumans\.venv\Scripts\python.exe `
  -I scripts\aigp_pytest.py vq2
```

Do not append to `benchmark_history.jsonl`. Never run
`python -m pytest -m live`, `preflight`, a simulator launcher, or any powered
stage as a side effect of ordinary development testing.

## Completed bounded maintenance task

The slow dry-run telemetry isolation is integrated and post-merge verified at
`ef92041bb3f05b1d8f3ef69182db8d51184c9cce`; its exact lifecycle and failure
provenance are recorded in
`docs/aigp/tasks/2026-07-20-maintenance-nonlive-artifact-isolation.md`. Do not
repeat this task.

The accepted test-only change passes an explicit pytest `tmp_path` record to
`run_vq1(dry_run=True, ...)` and checks the nonempty temporary artifact. The
runner's intentional default recording behavior is byte-unchanged. The target,
slow, fast, unit, canonical VQ2, full non-live, isolated hash-pinned VQ2, and
post-merge VQ2 gates all passed with the counts recorded above.

The promotion full suite left the fresh candidate's five tracked captures
byte-identical and produced only an inventoried `.pytest_cache`. A separate
fresh exact candidate retained an identical physical inventory across the
hash-pinned VQ2 run. At maintenance commit `ef92041`, the manifest contained
129 entries and changed only the VQ1 runner-test digest; the then-current VQ2
policy remained byte-identical at 1,325 expected passes. Those counts are
historical, not the later passive-tranche historical identities or current
Package 2 identities recorded above.

This maintenance improves promotion hygiene only. It advances neither M1, M2,
nor M4 and unlocks no replay, calibration, producer, estimator, runtime,
transport, simulator, or powered stage. If no consequential package entry gate
below is now satisfied, stop behavioral development and report the missing
prerequisite honestly.

## Current resume decision

The Package 2 separately authorized-collection entry gate is satisfied for
exactly one bounded build-3385 Training `calibration-excite` pilot after all
offline promotion and L0 freeze checks pass. The user approved routine
in-contract work without repeated permission prompts, no cooked-PAK
extraction, configurable and changeable nominal gate facts, simulator-observed
camera/IMU/timing/stream facts, and private decoded capture for this simulation
task only. That authority does not transfer to physical/HIL/submission/public
use, a second attempt, another powered stage, or Gate 0/Gate 1 work.

The first Package 2 implementation was promoted and integrated at `f33e436`.
L0 then stopped before freeze publication, A01 creation, simulator/port
contact, or capture because the launcher environment was not cross-bound to
the native frozen environment and no production initial import-inventory
builder existed. Correction behavior `e87c5e6` fixes those gaps, and its
trusted metadata is regenerated and independently reviewed. The remaining
gate sequence is fresh exact-candidate full non-live and hash-pinned VQ2,
exact integration, and detached post-merge VQ2. Only then may L0 derive and
independently review the live freeze. No attempt has been consumed.

The earlier Package 3B timing tranche remains integrated historical evidence
with three accepted passive captures. It must not be described as blocked or
repeated.

M1 itself is not complete. Consequential follow-on work still requires the
specific entry gate for that package: full Package 2 calibration inputs
(including the required camera/IMU calibration and timing/measurement model),
the no-send 50 Hz shadow scheduler and production handoff/deadline/skip
evidence, and—before operational replay—the approved labeled corpus, final
processor, calibrated policy, and pinned OS isolation wrapper. The exact
Package 2 pilot already has conditional task-specific authorization; any other
powered FlightSim stage requires its own frozen contract and authorization.
Absent one of those remaining prerequisites, do not reinterpret the passive
tranche, canonical tests, or Wave 3D/E synthetic evidence as calibration,
operational replay, completed runtime, or powered evidence.

The current continuation selected Package 2 because its entry gate permits
separately authorized collection, then froze and independently reviewed the
algorithm-independent readiness contract at `d526524`. That checkpoint was
subsequently followed by implementation at `508d43`/`f33e436` and correction
behavior at `e87c5e6`, but no new collection followed. Accepted passive
sessions 02 through 04 contain `182`/`181`/`181` frames but only `6`/`5`/`5`
unique decoded hashes, one exact target box `(282, 134, 80, 80)`, and `2,110`
exact-zero gyro vectors. They contain no independent known-geometry label.
They therefore provide no across-image intrinsic support, hand-eye rotation,
or temporal-offset excitation and cannot be upgraded into Package 2
calibration.

Package 2 remains incomplete. Do not repeat the static passive collection or
implement a detached fitter against it. The exact separately frozen powered
pilot is now the active path, subject to correction promotion and independent
L0 freeze review. The earlier passive authority does not transfer; simulator
and private-capture authority comes only from the exact Package 2 task contract.

## Consequential development roadmap

No package below is authorized merely by appearing here. Its entry gate must
be satisfied and recorded first.

```text
Package 1 replay/data trust ----+-> Package 4 full-homography producer
                                |                  |
Package 2 calibration/timing ---+                  +-> Package 6 sequential state
                                                                   |
Package 1 replay/data trust --------> Package 3A M2 geometry -------+
                                                                   +-> Package 7 replay

authorized simulator evidence -----> Package 3B M1 timing ---------+-> Package 8 runtime
Package 7 replay ---------------------------------------------------+

Package 3A M2 + Package 3B M1 + explicit authorization -> bounded Gate 1 recenter
```

The bounded Gate 1 path is independent of Wave 3E. The Wave 3E path does not
reach runtime until the producer, sequential-state, replay, and proof-carrier
gates are independently accepted.

### Package 1 — replay and data trust boundary

Entry gate:

- organizer/user attestation permitting storage and use of full decoded
  competition frames;
- an approved private content-addressed corpus with immutable build/mode/session
  provenance and reference labels;
- a final production `module:function` replay processor;
- a calibrated replay policy; and
- one pinned administrator-owned OS isolation wrapper path and hash used at
  both candidate-code boundaries.

Required evidence:

- immutable train/calibration/held-out session splits;
- Gate 0 plus recorded top-clipped Gate 1 coverage;
- visibility, clipping, geometry, tracker/crossing-residue, failure, and label-
  uncertainty annotations;
- deterministic processing and exact corpus/processor/policy/wrapper hashes;
- bounded causal T1 full-stack replay using only publication-order inputs; and
- operational T0-T4 scheduler/lease/resume/merge evidence only after every real
  T1 prerequisite exists.

Generated data must never be relabeled as recorded replay. A frame corpus
without labels or an independent reference-measurement protocol cannot validate
covariance coverage. Keep all private corpus data and operational configs out
of Git.

### Package 2 — production calibration and timing dossier

Entry gate: approved build-3385 calibration inputs or separately authorized
collection.

Required artifacts and evidence:

- pixel-to-camera-FRD ray rectification and Jacobians;
- exact calibration artifact ID, content hash, resolution, units, conventions,
  validity domain, and capture provenance;
- camera-to-body extrinsic with independently verified axis/sign conventions;
- camera/IMU time-offset model, per-sample host-arrival semantics, and bounded
  uncertainty;
- explicit separation of fixed/shared calibration nuisance from per-frame
  conditional fit covariance;
- held-out reprojection/angular residuals across the image;
- forward/inverse round trips and analytic Jacobians against finite
  differences; and
- repeat-session stability with pass limits frozen before held-out evaluation.

Do not reuse historical VQ1 intrinsics or Wave 3D's synthetic identity
calibration. Package 2 exits at reviewed calibration evidence; it does not by
itself produce a homography, estimator input, runtime selection, or command.

### Package 3 — parallel M2 and M1 evidence

Package 3A, recorded M2 geometry/tracker acceptance, requires Package 1:

- prove Gate 0 non-regression;
- demonstrate stable center and honest uncertainty for recorded top-clipped
  Gate 1;
- prove crossing residue cannot initialize the next active track;
- stratify results by clipping, visibility, perspective, and detector health;
  and
- withhold on insufficient evidence; placeholder distance or pose never
  reaches control.

Package 3B, the M1 receiver/runtime timing dossier, requires the appropriate
simulator access level and exclusive lease:

- measure receiver/reassembly behavior and load, real per-sample arrivals,
  p50/p95/p99/max stage timing, missed deadlines, and skip behavior;
- keep generated scheduler traces distinct from production receiver evidence;
  and
- treat actual send-to-actuator/gyro causal delay as a separately authorized
  powered experiment.

M1/M2 evidence can unlock the existing bounded Gate 1 path independently of
Wave 3E. It grants no powered authority by itself.

### Package 4 — full-homography producer feasibility and contract

Entry gate: accepted Packages 1 and 2, including a reference protocol adequate
to assess covariance honesty. Do not choose or fabricate an algorithm before
the algorithm-independent obligations are frozen and independently reviewed.

Required producer obligations:

- output rectified camera-FRD slopes;
- exact center gauge `H[0,0] == 1.0` and Wave 3E's frozen eight-parameter order;
- dense `8x8` `CONDITIONAL_FIT` covariance with all correlations retained;
- source observation/image identity, visibility/clipping, detector health,
  support/inliers, residuals, quality, algorithm/configuration identity, and
  calibration identity;
- fail-closed geometry, conditioning, gauge, convexity/orientation, and
  covariance checks; and
- separate treatment of calibration, timing, detector/model, common-mode,
  nonlinear-remainder, attitude, and sequential nuisance.

Required acceptance evidence:

- independent synthetic truth/oracle and degeneracy tests;
- analytic versus finite-difference derivatives where applicable to the
  selected covariance construction, plus dense off-diagonal influence;
- empirical held-out replay calibration/consistency stratified by perspective,
  visibility, clipping, and conditioning;
- independent mathematics, lifecycle/provenance, replay, and API/test reviews;
  and
- a quarantined offline composition that may call Wave 3E but does not create
  an estimator measurement or production consumer.

If identifiability or covariance honesty cannot be demonstrated, stop and
explicitly reconsider a shape-augmented finite-quadrilateral branch. That is a
new contract requiring complete shape, rates, dense covariance, and provenance;
current `/1` is not the fallback input. Choosing it terminates the Wave 3E
producer branch: re-plan Packages 6 and 7 for the new semantic, and never feed
shape-augmented evidence through Wave 3E.

### Package 5 — corrected-ray offline estimator study

This is parallel to, not dependent on, the homography producer. Entry requires
accepted camera/IMU timing and extrinsics plus approved replay.

Acceptance requires exact causal IMU lineage, reviewed attitude/timing/
extrinsic uncertainty, paired held-out comparison with the raw-camera bearing
path, and fail-closed innovation/withholding behavior under jitter, dropout,
and calibration perturbation. Exit is offline estimator evidence only; no
runtime or command selection.

### Package 6 — sequential local-differential state

Entry gate: accepted producer evidence, calibration/timing, and association
data.

The contract must decide and prove:

- cross-frame association and exact gate/track/reset lifecycle ownership;
- rate/state construction without zero-rate placeholders or naive finite
  differences;
- observability and bootstrap behavior for initially uncertain rates;
- separation of Wave 3E conditional fit covariance from calibration, timing,
  detector/model, attitude, nonlinear, and sequential nuisance;
- an augmented/Schmidt nuisance state, retained cross-covariance, or another
  independently proved common-mode construction;
- fixed reference/calibration error does not average toward zero;
- Wave 3D's one-shot nuisance envelope is not reused as independent noise; and
- complete seed/reference, distinct-frame, retirement, dropout, gate-change,
  and reset rules.

Acceptance includes analytic/finite-difference Jacobians, PSD and cross-
covariance invariants, known-truth sequences with motion/jitter/dropout/
clipping/swaps/resets/shared nuisance, and proof that repeated observations do
not spuriously shrink common uncertainty. Exit remains standalone estimator
evidence with no production consumer.

### Package 7 — approved replay composition

Entry gate: accepted Packages 1, 2, 4, and 6 plus the applicable Package 3A
recorded geometry/tracker evidence. Package 5 is required only if the corrected
ray is part of this composition. Package 3B/M1 is a later runtime gate, not a
prerequisite for this offline replay composition.

```text
approved replay processor
  -> reviewed full-homography producer evidence
  -> indivisible Wave 3E evidence
  -> reviewed sequential state
  -> optional Wave 3D transform
  -> quarantined diagnostics or proposal
```

Compare this composition pairwise against the current raw-camera filter and
zero-order hold. Require Gate 0 non-regression, top-clipped Gate 1 stability,
association and tracker-isolation results, calibrated innovation/coverage
diagnostics, p95 prediction comparison, and fail-closed health behavior. Label
T1 open-loop replay and synthetic T2-T4 evidence separately. Claim no receiver,
supervisor, send, actuator, FlightSim, or powered result.

### Package 8 — runtime and supervisor proof carrier

Entry gate: accepted replay candidate plus M1 timing evidence.

Before implementation, independently review the architecture choice between a
versioned proposal envelope and a supervisor-owned registry. The supervisor
must verify complete provenance without trusting detached values or caller-
asserted labels. Retain production per-sample arrivals, calibrated timing/
extrinsics, measured delay, exact nuisance/state identity, bounds, and
withholding diagnostics.

Acceptance proceeds through fail-closed supervisor tests and shadow evidence
before transport-selection review. Simulator-connected shadow evidence requires
its own declared simulator access level, exclusive lease, and applicable
authorization; replay acceptance and M1 evidence do not grant that access.
This package still grants no reset, arm, send, or cleanup authority.

## Authority ownership

| Owner | Sole authority |
|---|---|
| User/operator | Authorize each powered stage and any new private full-frame capture or corpus use |
| Integration owner | Promotion, merge, live-lease coordination, post-merge verification, and release of a reviewed candidate to its next gate |
| Safety supervisor | Reset epoch; countdown/GO; arm/disarm freshness; watchdogs and command bounds; collision and sequence handling; zero crossing; cleanup; timeout and latched-abort decisions |
| Race status | Gate-pass authority |
| Perception | Evidence and health only; never passage or command authority |
| Planning/control experiments | Objectives or proposals only; never reset, arm, send, or cleanup authority |

These authorities are not transferable through a function call, data label,
task handoff, or test result. A production consumer must preserve the complete
reviewed proof carrier required by its authority owner.

## Separate bounded live path

The Package 2 `calibration-excite` pilot is the active separate bounded live
path. It is discovery-only and grants no Gate 0/Gate 1 approach or passage
authority. The independently reviewed Gate 1 recenter stage remains a later,
separate path: Wave 3E is neither a prerequisite nor an excuse to broaden it,
and its entry still requires M1 timing/simulator evidence, M2
replay/tracker-isolation acceptance, a frozen stage plan, the exclusive
simulator lease, and its own authorization.

The stage moves the captured high-right Gate 1 target toward a conservative
image corridor, then stops on corridor or timeout and proves cleanup. It does
not attempt passage and aborts on any unexpected gate-index transition.

`preflight` is passive and sends no arm or flight target, but it still contacts
the simulator and binds live ports, so it requires lease coordination. Every
`sign-id`, `hover`, `gate0`, `gate0-observe`, Gate 1, system-ID, or powered
stage other than the exact Package 2 pilot requires its own contract and
authorization and must run only from its designated integration worktree. This
handoff makes no claim that FlightSim is currently running or responsive.

The authoritative live safety contract remains unchanged:

1. Prove both race-clock and IMU-clock rollback after `SIM_RESET`, observe
   advancing samples, and witness countdown.
2. Stop vision during reset and restart only after the new epoch is proved;
   camera frame IDs and timestamps do not roll back in this build.
3. Wait until GO plus `150 ms` before arming or powered commands.
4. Confirm arm/disarm only on a heartbeat newer than the request.
5. Pace at no more than `50 Hz` and drop missed ticks.
6. Abort on stale streams/target, estimator failure, unsafe collision,
   nonfinite or out-of-envelope commands, or excessive angle/rate.
7. Keep yaw rate at exact zero and clamp roll/pitch command rates to
   `0.25 rad/s`.
8. Cleanup always sends reset and proves the clean epoch and disarmed state; a
   cleanup failure fails the stage.
9. A credible close crossing that loses the target latches exact zero rate and
   zero thrust and waits at most `0.40 s` for a strictly newer race packet.
   Accept only a newer gate-index `1` result; a newer `0`/invalid result,
   timeout, stale stream, or collision aborts. Vision never declares passage.
10. `hover`/`gate0` launch-pad tolerance remains at most `12` tiny contacts,
    total impulse `0.05`, during the first `0.35 s` only.
11. Post-credit observation requires a proved same-epoch gate `0 -> 1`
    transition, resets only the stale target tracker while preserving source
    watermarks and every non-target watchdog, and sends only exact zero
    commands. It requires three distinct newer frames and ends by the earliest
    of pass + `0.20 s`, crossing confirmation + `0.40 s`, or flight start +
    `5.0 s`. It never authorizes a Gate 1 attempt.

Training can initially report heartbeat `base_mode=193` with the armed bit set
while actuator demand is zero. Powered stages must normalize and prove disarmed
after their reset instead of trusting the boot state.

Vision never declares passage. Planning never sends commands. Experimental
control never resets, arms, disarms, sends, or performs cleanup. Race status is
the sole gate-pass authority. Live work resumes from a fresh reset and passive
preflight, never from an assumed simulator or in-flight state.

## Promotion and integration checklist

Behavioral branches do not edit `config/t1_pytest_policy.json` or
`config/promotion_trusted_files.json`. Those remain integration-owner files
updated only after behavioral review.

For each accepted candidate:

1. Record the task contract and exact base before writing.
2. Run directly affected tests after each edit.
3. Run focused compatibility tests and canonical `test-vq2`.
4. Run `test-fast` and `test-unit` as required by scope.
5. Obtain independent mathematical, lifecycle/provenance, API/test, and
   authority/replay reviews appropriate to the change.
6. Commit behavior before editing promotion metadata.
7. Update policy/manifest only for the exact accepted delta.
8. Verify exact collection/pass arithmetic and sorted, unique, case-safe paths.
9. Independently rehash every trusted file and record raw/canonical JSON
   identities.
10. Commit the complete promotion candidate and prove its tracked status empty.
11. Create a fresh promotion-test worktree at that exact commit, run
    `test-full-non-live`, and inventory every physical side effect of the run.
12. Create a separate fresh exact worktree at the same unchanged commit, audit
    its physical contents, and run the isolated hash-pinned VQ2 suite there.
13. Fast-forward or merge that exact commit through one integration owner.
14. Run post-merge VQ2 verification and prove tracked `main` status empty.
15. Only then mark `post_merge_verified` and update shared handoffs.

T2-T4 remain synthetic nonpowered evidence. The shipped scheduler has no
powered T5 executor. Never reinterpret canonical tests, synthetic evaluation,
generated traces, or open-loop replay as FlightSim or closed-loop evidence.

## Stop and escalate conditions

Stop rather than manufacture progress when:

- the selected task's concrete calibration, producer, replay, isolation, or
  simulator-authority prerequisite is not actually satisfied;
- proposed work adds only more detached synthetic scaffolding without advancing
  M1, M2, M4, or promotion hygiene;
- anyone proposes deriving Wave 3E input from `/1`, fitted corners,
  border-normalized pixels, residual heuristics, guessed covariance, or
  arbitrary homography scale;
- anyone proposes a downstream rate/state built from zero-rate placeholders or
  naive finite differences without a reviewed sequential model;
- a candidate differs from its declared base, a candidate contains ignored
  executable source state, tracked `main` is unexpectedly dirty, or any
  new/unattributed ignored executable state appears;
- a policy count, trusted digest, discovery input, dependency, or required test
  result drifts;
- work would wire Wave 3D/E, corrected-ray evidence, scheduler output, or a
  proposal into production without its reviewed provenance and authority
  boundary;
- replay or full-frame capture lacks explicit permission;
- simulator access lacks the exclusive lease or powered authorization;
- reset/GO/freshness/watchdog/command/cleanup proof fails; or
- an older public interface conflicts with verified build-3385 behavior.

Quarantine and inventory unexplained state. Never erase it with `reset --hard`,
checkout discard, bulk cleanup, or automatic worktree deletion. If no entry
gate is satisfied, report the exact missing prerequisite and stop behavioral
development honestly; the completed maintenance is not fallback work.
