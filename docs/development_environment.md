# Development environments and dependency disclosure

The active competition-facing profile is `requirements/runtime-vq2.txt`:
FlightSim build 3385 vision, `HIGHRES_IMU`, race status, and raw MAVLink. The
root `requirements.txt` remains a conventional pip entry point for that
profile. If the competition publishes a stricter submission-file format,
adapt the packaging step without merging test, simulation, or training tools
into the flight runtime.

Use the exact Windows development lock from a fresh CPython 3.12 environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\development-test.lock.txt
.\scripts\dev.cmd test-vq2
```

On Bash hosts, `scripts/setup_venv.sh` now defaults to the same exact
development/test lock and normal `.venv`:

```bash
bash scripts/setup_venv.sh
```

The setup script reserves its exact profile before creating the environment or
running `pip`, so an interrupted install can only resume with the same profile.
It uses the venv's bundled `pip` to install the selected requirements directly;
there is no unbounded bootstrap upgrade. The development lock itself pins
`pip`, `setuptools`, and `wheel`, and the legacy profile includes that lock.
It deliberately refuses to adopt a populated environment that predates the
profile marker because its dependency provenance cannot be proved. In that
case, select a fresh `VENV_DIR` (or recreate the old environment manually)
instead of layering the managed profile over it.

Legacy simulation must use an explicitly distinct environment. The setup
script resolves path aliases, refuses to install `legacy-simulation.txt` into
normal `.venv`, and records the selected requirements profile so a later run
cannot silently mix profiles:

```bash
VENV_DIR="$PWD/.venv-legacy" \
REQUIREMENTS_FILE="$PWD/requirements/legacy-simulation.txt" \
bash scripts/setup_venv.sh
```

Use `dev.cmd` as the canonical Windows entry point. It launches the reviewed
`dev.ps1` with a process-scoped `ExecutionPolicy Bypass`, so a restrictive
machine policy does not require a persistent user/system policy change.

The profiles are deliberately separate:

- `runtime-vq2.txt`: the minimal build-3385 flight runtime.
- `development-test.txt`: reviewed direct inputs for resolving a new test
  environment.
- `development-test.lock.txt`: exact versions used by normal Windows
  development and CI-style validation.
- `legacy-simulation.txt`: synthetic/PyBullet and visualization tools. The Git
  dependency is pinned to an immutable commit rather than a moving branch.
- `optional-training.txt`: Torch/Ultralytics/ONNX tools, excluded from runtime.
  A long training campaign still needs a fully transitive platform lock plus
  dataset provenance before it is reproducible. See
  `docs/training_resilience.md` for the offline residual checkpoint contract
  and the read-only historical YOLO audit.

To update the development lock, resolve `development-test.txt` in a clean
CPython 3.12 environment, review every version change, write exact resolved
versions to `development-test.lock.txt`, and pass both `test-fast` and
`test-vq2`. Do not refresh a lock opportunistically inside a performance
experiment.

## Dependency and tool disclosure

Generate a local CycloneDX dependency inventory with:

```powershell
.\scripts\dev.cmd sbom
```

The default output is `.artifacts/dependency-inventory.cdx.json`, which is
ignored by Git. It records installed Python distributions and the repository
revision; it does not prove license compatibility or constitute legal advice.
Review package license metadata and the competition's current disclosure form
before submission. Use `docs/disclosures/ai-and-tools-template.md` to record
generative-AI and other material development tools. Keep simulator captures,
credentials, and private competition material out of Git.
