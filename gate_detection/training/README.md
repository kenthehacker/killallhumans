# Learned gate-pose experiment status

The checked-in `runs/gate_pose_v1/` directory is a historical output bundle,
not a reproducible or runtime-integrated detector. Do not resume it or spend a
long autonomous campaign on it yet.

What is verifiable today:

- `args.yaml` points to an absent absolute macOS dataset path.
- The extraction, train, validation, and export scripts formerly named in
  `ARCH.md` are not present.
- The local dataset and its provenance are absent.
- `best.pt` and `last.pt` are historical PyTorch archives. They have the same
  byte size but different SHA-256 digests, so they are not duplicate files.
  Neither is loaded by the build-3385 VQ2 runtime.
- `results.csv` records 100 epochs and about 62,630 seconds of elapsed training.

Run the read-only audit (it never deserializes model pickle and never trains):

```powershell
.\.venv\Scripts\python.exe scripts\audit_yolo_experiment.py --smoke
```

The report deliberately separates `basic_prerequisites_present` from
`ready_for_training`. File presence, a valid dataset manifest, direct package
versions, and historical metrics are useful diagnostics, but they can never
make a campaign ready on their own. `ready_for_training` remains false unless
an exact `campaign_contract.json` conforming to
`campaign-contract.schema.json` is present and every referenced artifact hash
is verified. The checked-in experiment has no such contract.

Before learned detection can be reconsidered, all of these gates must exist:

1. A human-reviewed dataset manifest conforming to
   `dataset-manifest.schema.json`, including source/permission, a content hash,
   label order, and disjoint session/track groups.
2. Versioned extraction, training, validation, and export code with locked
   optional-training dependencies and immutable resolved configuration.
3. A bounded one- or two-epoch smoke tier, followed by successive-halving
   budgets (for example 2, 10, then 25 epochs) instead of another blind
   100-epoch run.
4. Atomic periodic checkpoints containing optimizer, scheduler, scaler, RNG,
   epoch, history, split manifest hash, and best-candidate state. Resume must
   reject dataset/config/code drift.
5. A grouped held-out evaluation and ONNX/TensorRT accuracy/latency comparison
   against the existing sub-millisecond classical detector.
6. Explicit VQ2 runtime integration and replay evidence before any live use.

The campaign contract binds the dataset-manifest digest; the four pipeline
source files; immutable resolved configuration; a self-contained, fully
transitive `==` lock with SHA-256 distribution hashes; smoke and
successive-halving budget evidence; atomic checkpoint/resume/drift evidence;
grouped held-out results; classical, ONNX, and TensorRT comparison reports; and
VQ2 integration/replay artifacts. Paths must be canonical repository-relative
regular files with no symlink component. Missing fields, unknown fields,
duplicate JSON keys, non-finite numbers, path escapes, stale hashes, an
unhashed/unpinned lock, or an integration artifact not detected in the runtime
all fail closed. The manifest is an auditable gate, not permission to launch a
training job; training remains an explicit human decision.

The schema is scaffolding only; it intentionally does not invent the missing
dataset's provenance or labels. `last.pt` is evidence that a checkpoint file
exists, not evidence that the absent pipeline can be resumed safely.

`content_sha256` is verified against the complete sibling `dataset/` tree. The
audit hashes a version tag followed by every regular file's UTF-8 relative
path, byte length, and bytes in sorted path order; empty trees and symlinks are
rejected. Groups sharing a `session_or_track` may not cross holdout partitions.
The historical `args.yaml` data path must resolve to that verified `dataset/`
tree (or its `data.yaml`); a different existing path remains blocked. Relative
paths resolve against the immutable run directory. A schema-valid manifest
without the matching content remains blocked.
