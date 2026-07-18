# killallhumans
bytedancers 

# Directories:
* gate_detection (houses logic to enable existence of all gates)
* gate_sequencing (logic for determining which gate to go through next)

## AIGP VQ2 training runner

Start FlightSim in **Training**. It may be windowed and unfocused, but keep it
unminimized and unpaused. From this repository:

```powershell
.\.venv\Scripts\python.exe -m scripts.aigp_vq2_run --stage preflight --record
```

The bounded powered stages are `sign-id`, `hover`, and `gate0`. Each stage
proves a fresh simulator reset, waits for GO, checks every required stream,
and confirms disarm/reset on exit:

```powershell
.\.venv\Scripts\python.exe -m scripts.aigp_vq2_run --stage sign-id --record
```

`gate0` has completed a simulator-credited, collision-free first-gate pass. It
is still a bounded development stage, not a full race runner. Captures are
written under `captures/` and ignored by Git.

See the [VQ2 build 3385 handoff](docs/aigp/2026-07-18-vq2-handoff.md) for the
current live-test results, safety invariants, simulator paths, and next step.

