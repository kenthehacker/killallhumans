# AI Grand Prix agent guide

Follow `Agents.md` for the canonical workflow and
`docs/aigp/2026-07-18-vq2-handoff.md` for the authoritative build-3385 flight
state and safety contract.

## Operating model

- Active simulator target: FlightSim build 3385, Training mode.
- Available production inputs: UDP JPEG vision, `HIGHRES_IMU`, heartbeat, race
  status, actuator status, and collision events.
- Unavailable/unusable in VQ2: a trustworthy pose stream and gate-map geometry.
- Output: bounded attitude-rate plus thrust commands. Current yaw command is
  zero until authority/sign are separately calibrated.
- Per edit: run directly affected tests.
- Per accepted candidate: run `.\scripts\dev.cmd test-vq2`; treat the command's
  current collected count as authoritative because the safety suite grows.
- Synthetic/PyBullet evaluation: run only for affected planning/control work or
  pre-merge promotion, never as the automatic objective for every change.

The active VQ2 architecture is intentionally narrower than the legacy VQ1
pose/map stack:

```text
Camera -> red-gate detector/tracker ----+
HIGHRES_IMU -> attitude estimator ------+-> safety-gated VQ2 runner
Race status/collisions/watchdogs -------+-> body rates + thrust
```

## Commands

```powershell
.\scripts\dev.cmd test-target <paths-or-node-ids>
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
.\scripts\dev.cmd test-vq2
.\scripts\dev.cmd test-slow
.\scripts\dev.cmd test-benchmark
.\scripts\dev.cmd test-full-non-live
.\scripts\dev.cmd preflight
```

`preflight` is passive. Powered stages are never routine tests and require
explicit user authorization. They must retain fresh reset proof,
countdown/GO, stream and attitude/rate/collision watchdogs, command pacing and
bounds, and proved cleanup. Do not weaken a flight invariant to accelerate a
development loop.

Do not treat historical `overall_passed` benchmark records as validated VQ2
results, append new objects to `benchmark_history.jsonl`, hardcode qualifier
dates, or silently resolve a conflict between public VQ1 documentation and
empirical build-3385 behavior. Keep new/private full captures ignored and
preserve existing tracked historical capture evidence.

Dependency profiles, the exact development lock, inventory generation, and the
AI/tool disclosure template are documented in `docs/development_environment.md`.
