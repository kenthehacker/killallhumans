# Offline training resilience

These tools are offline legacy-control or detector experiments. They are not
the build-3385 VQ2 production path and are never part of a generic test or
powered simulator command.

## Tracker residual

Dataset collection uses the content-addressed prepared-course API. Its default
horizon is `max(45 seconds, planned trajectory time + 15 seconds)` and it
publishes only safe, valid, completed track sessions:

```powershell
.\.venv\Scripts\python.exe scripts\collect_residual_dataset.py `
  --out control\residual_dataset.npz
```

`--allow-prefix` is diagnostic only. It accepts an incomplete session only
when completion is literally false and termination is exactly `time_limit`;
crash, DQ, invalid, contradictory, and missing evidence fail closed. The
schema-v4 NPZ stores whole-session/track identifiers and non-pickle name tables
so the trainer can make grouped holdouts. Its manifest also binds every track
config digest, the collector source, evaluator version, and numerical
dependency fingerprint. All seven current courses are eligible; none is
silently excluded.

Training prepares every course once, computes one baseline matrix, and reuses
those objects for every candidate evaluation. Completion scoring is the
default; a short prefix comparison must be requested and labeled explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\train_tracker_residual.py `
  --dataset control\residual_dataset.npz `
  --out control\residual_weights.npz `
  --checkpoint .artifacts\residual-training-state.npz
```

The checkpoint is atomically replaced after every epoch and every completed
expensive evaluation. It contains current and best models, all Adam buffers,
RNG state, completed epoch, history, grouped split, normalization, baseline
results, prepared artifact identities, and early-stop/best-candidate state.
Dataset, trainer/dependency source, configuration, Python, or NumPy drift makes
resume fail closed. The signature also hashes the closed-loop rollout sources
and evaluator version, so benchmark/sequencer/controller drift cannot silently
reuse an old checkpoint. Each track's `max_total_time_s` threshold is resolved
to the same completion horizon used for that rollout (including horizons over
30 seconds):

```powershell
.\.venv\Scripts\python.exe scripts\train_tracker_residual.py `
  --dataset control\residual_dataset.npz `
  --out control\residual_weights.npz `
  --checkpoint .artifacts\residual-training-state.npz `
  --resume
```

Use `--restart` to explicitly replace an existing checkpoint. A candidate that
fails safety, plan validity, completion, progress, or tracking-regression hard
gates is never published as runtime weights. If no candidate clears every
closed-loop gate, training preserves its resumable checkpoint and fails closed
without writing `--out`. Only the explicit `--skip-closed-loop` diagnostic mode
may publish a validation-loss-only model. Prefix results never count as
race-completion evidence.

## Historical YOLO pose output

See `gate_detection/training/README.md`. The only executable added for the
historical run is a read-only audit:

```powershell
.\.venv\Scripts\python.exe scripts\audit_yolo_experiment.py --smoke
```

It hashes and CRC-checks the archives without deserializing model pickle. It
does not train, export, alter weights, or claim the missing dataset/pipeline is
reproducible.

The audit exposes basic file/package prerequisites separately from actual
campaign readiness. Actual readiness additionally requires the strict,
content-bound `gate_detection/training/campaign_contract.json` described by
`campaign-contract.schema.json`, including immutable pipeline/config and a
fully transitive hashed lock, bounded smoke/successive-halving budgets, complete
checkpoint/resume/drift state, grouped holdout evidence, classical versus
ONNX/TensorRT comparisons, and VQ2 runtime/replay evidence. That contract is
absent today, so the historical run remains fail-closed even if someone later
adds only the missing filenames or dataset directory.
