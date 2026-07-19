# AI Grand Prix engineering instructions

## Current target

The active target is FlightSim **build 3385** in **Training** mode. Treat
`docs/aigp/2026-07-18-vq2-handoff.md` as authoritative for live flight status,
the verified interface, and the safety contract.

Public VQ1 material is useful historical context, but it does not match every
empirical build-3385 behavior. In VQ2 there is no usable pose or gate-map
stream. The production path is:

```text
UDP JPEG vision + HIGHRES_IMU + race status
                  -> target tracking and IMU attitude estimation
                  -> safety-gated body-rate/thrust commands
```

Do not invent qualifier dates or assume an older public interface overrides a
verified build-specific finding. Flag any conflict for human review.

## Default development loop

Use the canonical Windows task surface from the repository root:

```powershell
.\scripts\dev.cmd test-target tests/test_aigp_vq2_runner.py
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
.\scripts\dev.cmd test-vq2
```

Run directly affected tests after each edit. Run `test-vq2` for each accepted
candidate; it is the fast, dedicated non-live safety suite, and its collected
test count is expected to grow with the stack. The default pytest policy
excludes `slow`, `benchmark`, and `live`, enforces strict markers, and applies
a hard wall timeout.

Synthetic and PyBullet matrices are module-specific or pre-merge evidence, not
the universal optimization objective. Invoke them explicitly with
`test-benchmark`; use `test-full-non-live` only at a promotion boundary. Do not
append to `benchmark_history.jsonl`, whose historical records are multiline
objects and whose skipped PyBullet tier made old `overall_passed` values
misleading.

The explicit isolated and bounded legacy tiers are
`.\scripts\dev.cmd test-unit` and `.\scripts\dev.cmd test-slow`. A `live` marker is
run directly with `python -m pytest -m live` only after explicit authorization;
there is intentionally no generic powered-test task.

## Simulator boundary

`preflight` is passive and sends no arm or flight targets:

```powershell
.\scripts\dev.cmd preflight
```

Any powered FlightSim stage requires the user's explicit authorization. Never
put `sign-id`, `hover`, `gate0`, `gate0-observe`, or a future powered stage in a
generic test command. Powered work must preserve the proved reset epoch,
countdown/GO, fresh-stream checks, watchdogs, command bounds, disarm/reset, and
cleanup confirmation in the VQ2 handoff. A cleanup failure is a failed stage.

The parameterized Windows launcher is:

```powershell
$env:AIGP_FLIGHTSIM_PATH = 'C:\path\to\AIGP_3385\FlightSim.exe'
.\scripts\dev.cmd launch-sim
```

It discovers the active interactive desktop and refuses to double-launch.
Training-mode GUI selection can still require a human desktop action.

## Data, dependencies, and disclosure

Use `requirements/development-test.lock.txt` for normal development. Keep the
VQ2 runtime, test, legacy simulation, and optional training environments
separate as documented in `docs/development_environment.md`.

Keep new/private full captures, credentials, simulator data, and generated
dependency inventories out of Git; preserve existing tracked historical
capture evidence. Before a submission, generate an inventory with
`.\scripts\dev.cmd sbom` and complete
`docs/disclosures/ai-and-tools-template.md` through human review. The generated
metadata is evidence, not a guarantee of license or rules compliance.

Historical `.loop` and `.research_loop` files are evidence. Preserve them, but
exclude them from routine searches unless the task explicitly needs history.
