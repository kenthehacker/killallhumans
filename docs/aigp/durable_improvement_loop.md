# Durable replay-first improvement loop

This is the operating contract for the P2 tooling. It does not change VQ2
flight behavior, start FlightSim, or authorize recording. The live-flight
safety source remains `docs/aigp/2026-07-18-vq2-handoff.md`.

## Honest boundary

- The normal scheduler supports T0-T4 only. It rejects T5.
- No powered or live run is part of this workflow.
- `AIGP_TRIAL_OFFLINE=1` and the scrubbed child environment are advisory
  process inputs, not an OS network firewall.
- T1 candidate code is acceptable only behind a separately reviewed, pinned
  OS isolation wrapper that attests denied networking, a read-only worktree,
  non-interactive execution, denied access to the trusted host process, and
  kill-on-wrapper-exit descendant containment.
  This repository does not ship such a Windows wrapper, so T1 deliberately
  fails closed until an operator supplies one.
  The pathname/hash recheck is not an atomic verified-execution primitive:
  the wrapper must remain at an administrator-owned location that neither a
  candidate nor ordinary user process can replace for the entire run.
- T2-T4 invoke only hash-pinned trusted synthetic evaluators. They prove a
  nonpowered kinematic domain; they do not prove official-simulator behavior.
  Evaluator hashing is not a general sandbox: those tiers import candidate
  control/planning code and assume an operator-reviewed, non-hostile commit.
  Do not run an untrusted autonomous commit at T0 or T2-T4 without adding an
  external OS sandbox that deliberately permits only the required cache/temp
  writes. Never mount or expose a private replay corpus to those unwrapped
  tiers.
- No powered executor or watchdog supervisor is shipped. A backend-supplied
  mapping and authorization phrase are planning metadata, not proof of hard
  containment; `WarmCampaign.run()` always refuses before a lease, preflight,
  candidate application, or powered occurrence is started.

## Promotion order

| Tier | Exact role |
|---|---|
| T0 | Directly affected tests/import checks for every edit |
| T1 | Scoped VQ2 suite plus isolated, labeled full-stack replay for every T0 passer |
| T2 | Exact warm `race_01` prepared synthetic evaluation |
| T3 | Reviewed changed-domain subset |
| T4 | Exact seven-track matrix plus the explicit hash-reviewed non-live test inventory |
| T5 | Reserved for a future separately supervised official-simulator executor; unavailable here |

All T0-eligible candidates advance to T1. Quality-based successive halving
starts at T1, where labeled replay quality exists. Hard gates are evaluated
before quality: no collision/disqualification/stale-stream/cleanup failure,
then correct sequence and completion, then centering/stability, then time.
Safety is never blended into a scalar speed score.

## Build the reviewed T0-T4 configuration

All commands below work either as modules or as direct scripts from the repo
root. Keep the ledger, worktrees, private corpus, and isolation wrapper outside
Git.

First create and review the trusted evaluator manifest. The checked-in example
was generated from these paths; regenerate it after any listed source/test
change:

```powershell
.\.venv\Scripts\python.exe scripts\aigp_trials.py `
  --ledger .aigp-loop\trials.sqlite3 build-trusted-manifest `
  --repo . --out config\promotion_trusted_files.json --overwrite `
  .gitattributes `
  aigp_loop\__init__.py aigp_loop\_util.py aigp_loop\ledger.py `
  aigp_loop\campaign.py aigp_loop\nonlive.py aigp_loop\promotion.py `
  aigp_loop\replay.py aigp_loop\scheduler.py `
  scripts\aigp_nonlive.py scripts\aigp_pytest.py scripts\aigp_replay.py `
  scripts\benchmark.py `
  scripts\benchmark_matrix.py planning\__init__.py `
  planning\artifact_cache.py `
  sim_pybullet\configs `
  config\t1_pytest.ini config\t1_pytest_policy.json `
  conftest.py pyproject.toml tests competition\tests control\tests `
  estimation\tests flight_control\tests gate_detection\tests `
  gate_sequencing\tests planning\tests sim_pybullet\tests simulation\tests
```

Both manifest/config builders publish atomically and refuse an existing output
unless `--overwrite` is explicit. Trusted inputs may not be symlinks, including
symlinks nested under a selected directory, and every resolved file must stay
inside the repository. Generate to a review path first when changing the trust
boundary, compare it, then use the explicit overwrite flag.

`aigp_nonlive` refuses a trusted manifest that omits its complete local import
bootstrap (`aigp_loop/__init__.py`, `_util.py`, `ledger.py`, `nonlive.py`, and
`promotion.py`, plus the non-live/benchmark scripts) and verifies stable byte
snapshots before any repository package import. The exact seven track JSON
files and their directory inventory are pinned too; extra or case-colliding
configs fail closed. T4 also compares every file
below its ten explicit test roots, including non-Python fixtures/assets, plus
all pytest/startup discovery inputs, with that manifest. It uses
`python -I -m pytest`, explicit timeout-plugin loading with plugin autoload
disabled, no cache provider, the reviewed root config, and only those roots.
All tiers reject adjacent executable bytecode and alternate package/native
extension paths that could shadow a pinned Python module. The manifest builder
also rejects Git-tracked bytecode anywhere in the repository. T1 and T2-T4
install fresh external bytecode-cache prefixes before repository imports; the
trial launcher does the same so it cannot create a cache in the trusted base
before starting T1. Start promotion from a cache-clean trusted checkout: the
T1 replay evidence bootstrap rejects ignored/untracked `__pycache__` and
`.pyc`/`.pyo` spellings on its import boundary without invoking Git. T4 pins
the nested pytest working directory to the reviewed repository and scrubs all
inherited `PYTEST_*` controls. Canonical Windows `scripts/dev.ps1` Python tasks
also use a unique process-scoped cache prefix under the OS temp directory, so
recommended pre-promotion testing does not poison the trusted source tree.

Copy `config/promotion_commands.example.json`. Replace every T1 placeholder
with an approved private corpus manifest, the candidate `module:function`, and
the same absolute reviewed isolation-wrapper path/hash at both candidate-code
boundaries. The scoped pytest step is wrapped by the scheduler; the replay host
is run as `python -I` from the hash-pinned base checkout, never from the
candidate worktree, and launches only the candidate worker through the pinned
wrapper. Both the command document and replay argv must name exactly
`config/promotion_trusted_files.json`, whose own digest is command-bound. The
replay script parses effective argv and verifies its canonical manifest before
importing `aigp_loop`. The candidate worktree path is passed explicitly to the
worker; it is not placed on the trusted host's import path. A zero wrapper hash is
intentionally non-runnable. T0 should be
narrowed to the actual affected tests for the candidate and must run without a
private corpus mounted or otherwise accessible.

Run each reviewed evaluator once to obtain its deterministic
`evaluation_input_hash`, `evaluation_config_sha256`, seed, repetitions, and
evaluator version. Put those exact values in a copy of
`config/promotion_ladder_identities.example.json`. T0's identity is the
reviewed affected-test set/config; T1 uses corpus score identity; T2-T4 use the
corresponding non-live evidence. Then materialize the candidate config:

```powershell
.\.venv\Scripts\python.exe scripts\aigp_trials.py `
  --ledger .aigp-loop\trials.sqlite3 prepare-ladder-config `
  --base-config config\promotion_candidate_base.example.json `
  --tier-identities config\promotion_ladder_identities.json `
  --commands config\promotion_commands.json `
  --out .aigp-loop\candidate.json
```

The builder computes each exact `TierCommand` hash from the reviewed command
document and embeds it in ladder-manifest schema 2 before hashing the full
TrialKey identity. Scheduler execution/resume, merge, and campaign planning all
require checkpoint command hashes to equal those frozen values; a different
but internally self-consistent command plan is rejected.

The command prints the only valid `config_hash`, full-ladder `dataset_hash`,
and `evaluator_version` for enqueue:

```powershell
.\.venv\Scripts\python.exe scripts\aigp_trials.py `
  --ledger .aigp-loop\trials.sqlite3 enqueue --repo . `
  --config .aigp-loop\candidate.json `
  --dataset-hash <printed-dataset-hash> `
  --evaluator-version <printed-evaluator-version> --seed 42
```

Enqueue requires a clean commit because execution occurs in an exact detached
worktree. Run cohort rounds in order with a shared external worktree root:

```powershell
0..4 | ForEach-Object {
  .\.venv\Scripts\python.exe scripts\aigp_trials.py `
    --ledger .aigp-loop\trials.sqlite3 round --repo . `
    --worktree-root C:\Users\John\aigp-worktrees `
    --commands config\promotion_commands.json --tier $_ `
    --keep-fraction 0.5
}
```

T0 ignores the halving fraction and advances every eligible member. Reissuing
an interrupted round reuses immutable completed checkpoints and terminally
reconciles a durable failed checkpoint; it does not rerun either. Planned and
decided round records are idempotent and immutable.

## Scheduler integrity

The SQLite ledger deduplicates the five-part TrialKey and records leases,
heartbeats, provenance, timings, metrics, artifacts, failures, and bounded
output tails. Checkpoint completion/failure is immutable. Full-ladder
checkpoints bind both the manifest tier identity and the command/trusted-file
identity; merge and T5 revalidate the complete T0-T4 chain.

Every command receives a minimal case-insensitive environment allowlist plus
explicit trial/config/cache fields. The materialized config is read-only and
its exact bytes/hash, plus full Git provenance, are checked before and after
every step. Windows commands are created suspended, assigned to a
kill-on-close Job Object, and resumed only after containment. On every exit,
including success, the job is terminated and queried until it proves zero
active descendants. Cleanup uncertainty is a failed evaluation. A shared
external content-addressed cache makes T2 warm without placing mutable cache
state in candidate worktrees.

Stdout/stderr are continuously drained into bounded tails. Any truncation is
an evidence-integrity failure, so a valid-looking JSON suffix can never be
parsed or promoted after earlier output was discarded.

Every new or resumed worktree must also be an exact lexical checkout of the
Git tree. Only tracked files/directories and its validated `.git` worktree
indirection may exist. Ignored modules, weights, caches, extra files, empty
directories, and symlink/junction aliases left by a killed prior process are
rejected before another candidate command can start.

The warm-cache workflow is directly constructible: use one persistent external
worktree root, for example `C:\Users\John\aigp-worktrees`. Every evaluation
child then receives
`AIGP_CACHE_ROOT=C:\Users\John\aigp-worktrees\.artifact-cache`. The first T2
evaluation for an exact source/input/dependency key materializes verified
prepared artifacts; a later candidate with the same relevant key reports a
cache hit and reuses them. T3/T4 worker processes share the root but lock per
content key. Source, resolved configuration, schema, numerical dependency, or
seed changes produce new keys. Corrupt/partial payloads are rejected and
rebuilt atomically. Use separate worktree roots when cache sharing is unwanted.

## Private replay capture and corpus

Capture requires both organizer approval and the explicit operator flag:

```powershell
.\.venv\Scripts\python.exe scripts\aigp_vq2_run.py `
  --stage preflight --record --replay-bundle --recording-approved
```

Bundles are private, immutable, content-addressed session directories. The
bounded callback snapshots every image (including read-only views with a
writable base alias) but never serializes or performs disk I/O on the
vision/control thread. Queue loss,
invalid callback types, count mismatch, writer/finalizer error, or timeout
leaves incomplete or permanently invalid evidence. Verification checks exact
manifest/record types, sequence and frame-token uniqueness, non-symlink
in-bundle blobs, exact references, HxWx3 `uint8` metadata/pixels, all hashes,
and the dataset identity.

Bundle construction rolls back its newly owned leaf on setup failure. Once an
initial incomplete manifest exists, every close/precommit failure exhausts
handle cleanup and terminal-incomplete publication before returning; it never
leaves a nominally open writer that an async finalizer can mistake for sealed
evidence.

Reader and writer ceilings bound manifest, records, JSONL line/count,
annotations, corpus/policy files, frame count, encoded blob, decoded
dimensions, and whole-session blob bytes. NPY headers are checked against the
HxWx3 `uint8` allocation ceiling before `numpy.load`. Outcome and start/finish
provenance are part of the dataset identity. Annotation schema deliberately
keys labels by `(session_id, generation, frame_id)`; `sim_time_ns` remains the
pixel/order integrity token, while writer, recorder, and reader require the
label pair `(generation, frame_id)` to be unique within each frame stream.

A corpus manifest has schema `aigp-vq2-replay-corpus/1` and exact members:

```json
{
  "schema": "aigp-vq2-replay-corpus/1",
  "sessions": [
    {
      "session_id": "approved-flight-001",
      "bundle": "approved-flight-001.vq2replay",
      "annotations": "approved-flight-001.labels.jsonl",
      "policy": "approved-flight-001.policy.json"
    }
  ]
}
```

Train/validation assignment hashes whole `(session_id, dataset_hash)` groups,
rejects duplicate/coerced identities, and guarantees nonempty train and
validation groups when at least two sessions exist.

## T1 full-stack replay contract

The candidate is not a detector-only callback. For every decoded image it
receives a deep-immutable causal context with schema
`aigp-vq2-full-stack-context/1`. Candidates run strictly in decoded publication
order and see only allowlisted sensor records whose sequence is no later than
that decoded frame. A later recorded processed-frame sequence, existence, and
state are not exposed; recorded detector/tracker/estimator/command outputs are
never candidate inputs. Temporal, reacquisition, miss-streak, and command
pairing scores likewise use exact publication sequence; simulator timestamps
are evidence fields and cannot reorder controller history. The callable must
return exactly:

```text
detections, tracker, estimator, generated_command
```

All four outputs have strict finite schemas. The isolated worker receives only
the image and sanitized context over a bounded, request-ID-matched line
protocol. Request write plus response read share one deadline; responses have
a size limit, while stderr is drained to a bounded tail. The reviewed wrapper
must attest that every descendant is killed
when it exits and that the candidate cannot inspect the trusted host process,
its memory, handles, or command line. An insufficient wrapper is rejected
before launch. Python `random` and
NumPy are seeded before candidate import/execution from the frozen exact seed;
`PYTHONHASHSEED` uses the same seed reduced to its unsigned 32-bit domain.
Candidates using Torch or another framework remain responsible for their own
deterministic framework settings.

The line-protocol worker is securely resolved to an absolute path in the exact
pristine candidate worktree and launched as `python -I <absolute-worker>`, so
an installed or `PYTHONPATH` `scripts` package cannot select a different
worker. Only after its stdlib/NumPy dependencies load does it insert the
derived candidate root. The host passes the exact local processor source path;
the worker rejects an imported module with a different origin. Dotted
processor modules require every local parent to have a secure `__init__.py`,
preventing namespace-package fallback. Candidate worker/processor provenance
is therefore bound by `processor_code_sha256`; the worker is not claimed as a
base-evaluator source. The base evaluator identity exactly binds its executed
package initializer, utility, ledger, promotion, replay, and CLI sources. The
trusted host still validates every request ID, schema, output bound, deadline,
and returned control/perception value.

The measured IPC plus full-stack wall time is reported as
`full_stack_latency_ms`, never mislabeled detector latency. A recorded detector
latency remains informational only for recorded-stack scores. Processor
provenance is the full Git worktree code hash, not merely the entry module.
Promotion-eligible isolated runs additionally require the candidate worktree
to match the exact pristine Git tree, rejecting ignored modules, extra files,
and symlinked package parents. Non-isolated in-process processor scoring is
explicitly test-only and cannot satisfy the promotion isolation gate.

Score output separates deterministic `evaluation_input_hash` (bundle/corpus,
labels, policy, full code, evaluator config including seed, and isolation
wrapper) from timing/result-dependent `evaluation_result_hash`. Corpus scoring
applies every session policy, requires one seed/processor/isolation identity,
and reports aggregate plus worst-session metrics.

Replay can validate perception, causal estimation, and open-loop command
generation, but cannot validate observations changed by those commands.
Closed-loop candidates must still pass T2-T4 and a finalist needs separately
authorized T5.

The T1 wrapper is host/process containment, not Byzantine integrity for Python
running inside the candidate process. T1 therefore assumes the candidate entry
point is operator-reviewed, protocol-conforming, and non-Byzantine. Evaluator
correctness still depends on the separately hash-reviewed trusted
host/bootstrap and exact corpus/policy; do not treat wrapper attestation alone
as proof that arbitrary in-process test or plugin code reported honestly.

## Powered campaign is fail-closed

`scripts/aigp_campaign.py` only freezes a plan; it never starts power. Its
input must have exact schema `aigp-live-campaign-plan-input/1`, one baseline,
unique trial IDs, one shared reviewed powered stage for baseline and
candidates, exact SHA-256 candidate provenance, and a reviewed backend
declaration. The
declaration includes build, offline/non-interactive flags, candidate mode,
maximum duration, and watchdog metadata with an implementation SHA-256 and
`hard_stop_before_return=true`. Those values are frozen into the plan hash but
are not self-authenticating proof. No repository code invokes the backend.
Even with the exact authorization phrase, `WarmCampaign.run()` raises
`powered campaign execution is unavailable` before any live lease or T5 child
is created. A future executor must be a separately reviewed, hash-pinned,
out-of-process supervisor that itself owns and proves hard-stop process-tree
containment; it cannot be enabled merely by changing the declaration.

## Historical evidence

`benchmark_history.jsonl`, `.loop/`, and `.research_loop/` remain preserved but
untrusted. The one-time importer marks historical rows non-comparable and
ineligible. `.ignore` keeps those archives, private replays, and ledger state
out of normal agent search.
