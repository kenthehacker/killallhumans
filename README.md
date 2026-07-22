# killallhumans

Autonomy software for the AI Grand Prix. The active target is FlightSim build
3385 in Training mode. VQ2 provides vision, `HIGHRES_IMU`, race status,
actuator status, and collisions; it does not provide usable pose or gate-map
data.

The current runner safely passes gate 0 and observes gate 1, but it is not a
full-race controller. See the
[build-3385 VQ2 handoff](docs/aigp/2026-07-18-vq2-handoff.md) before changing
flight behavior.

## Windows development

Create the exact development/test environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\development-test.lock.txt
```

Use one command surface from the repository root:

```powershell
.\scripts\dev.cmd test-target tests/test_aigp_vq2_runner.py
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
.\scripts\dev.cmd test-vq2
.\scripts\dev.cmd test-slow
.\scripts\dev.cmd test-benchmark
.\scripts\dev.cmd test-promotion
```

Normal pytest selection excludes `slow`, `benchmark`, and `live`, rejects
unknown markers, and enforces a hard per-test wall timeout. The benchmark and
full non-live tiers are explicit promotion gates, not the per-edit loop.
`test-promotion` typically takes 10-13 minutes, streams per-test progress, and
uses external commit-keyed state so another caller attaches to the same run
instead of repeating it after an output-channel timeout. The historical
`test-full-non-live` spelling is a command-name-only compatibility alias; both
names require a clean exact commit and may reuse its result. State defaults to
`%LOCALAPPDATA%\AIGP\promotion-tests\v1\<repository-scope>` and can be moved to
another external local directory with `AIGP_PROMOTION_STATE_ROOT`. A hard
15-minute supervisor ceiling prevents an unexpectedly wedged suite from
running indefinitely. Run it from a fresh exact-commit worktree: ignored files
are rejected because they could otherwise change behavior without changing the
durable key.
Tests marked `live` are invoked directly with `python -m pytest -m live` only
after explicit authorization; there is intentionally no generic powered task.

Passive simulator health check:

```powershell
.\scripts\dev.cmd preflight
```

Preflight does not arm or send flight targets. Every powered FlightSim stage
requires explicit authorization and must retain the reset/countdown/watchdog/
cleanup contract in the handoff.

The simulator launcher accepts `AIGP_FLIGHTSIM_PATH` or an explicit path,
discovers the active interactive Windows session, and refuses to double-launch:

```powershell
$env:AIGP_FLIGHTSIM_PATH = 'C:\path\to\AIGP_3385\FlightSim.exe'
.\scripts\dev.cmd launch-sim
```

Training-mode selection may still require an interactive desktop action.

Environment profiles, lock updates, dependency inventory generation, and the
AI/tool disclosure template are documented in
[development environments](docs/development_environment.md). Keep generated
inventories and new/private full captures out of Git; preserve existing
tracked historical capture evidence.
