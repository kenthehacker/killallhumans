# iter-026 sim-stack unification plan (synthesis)

## Background

User reported visual_demo race_01 = 4/12 gates while matrix = 12/12. Root
cause established: matrix bench (1 kg / 20 N / kinematic accel-clamped at
15 m/s²) and visual_demo (27 g CF2X / gym-pybullet-drones DSLPIDControl
/ full rigid-body) run fundamentally different drones. AIGP spec
(VADR-TS-002) specifies 280×280×160 mm chassis but explicitly does
NOT specify mass or thrust (must be SITL-calibrated).

3-agent planning swarm fired (Opus 4.7 max-thinking, GPT-5.5 xhigh,
Composer-2). Outputs in `.loop/research/sim_stack_unification_*.md`.

## Convergent diagnosis

All three: matrix wins are overfit to a permissive kinematic drone,
visual_demo's CF2X is too small to follow those trajectories. Solution
is NOT to chase CF2X or PyBullet matrix — solution is to swap
visual_demo to a 1-kg-class PyBullet drone that matches the bench.

## Synthesis decision (chosen approach)

Adopt **Opus's plan** with minor sharpening:

1. **`sim_pybullet/drone.QuadrotorDrone` is already at 1 kg / 20 N** (iter-021
   deliberately kept conservative attitude caps). Already coincident with
   the matrix bench's drone envelope. Swap visual_demo from `GPDDrone` (CF2X)
   to `QuadrotorDrone`.

2. **Keep kinematic matrix as fast oracle.** Don't replace with PyBullet
   wholesale — kinematic checks autonomy logic ~100× faster (5 s/track
   vs 30+ s/track). Two-tier model:
     - `kinematic_oracle` = fast smoke tier
     - `pybullet_aigp_quad` = truth tier (binding)

3. **Freeze `competition/drone_spec.py` values** — 1 kg / 20 N / 15 m/s² /
   15 m/s. Don't take GPT-5.5's 0.8 kg / 24 N rebaseline in this iter.
   Re-tuning everything baked at 1/20 invalidates iter-009i racing-line
   basin, race_01 ILC schedule, and auto-velocity ceiling. Land structure
   first; rebaseline numbers AFTER SITL calibration.

4. **Add `tests/test_sim_stack_parity.py`** — the binding gate. Runs
   `run_sim_benchmark(race_01)` and headless `visual_demo`; asserts
   `gates_passed_bench == gates_passed_demo` (both ≥ 12). This would
   have caught the present 12/12 vs 4/12 split.

5. **Keep `GPDDrone` (CF2X) behind `backend="cf2x"` opt-in.** Don't
   uninstall gym-pybullet-drones — legacy comparison may matter.

## Disagreements + resolution

- **GPT-5.5 wanted to rebaseline to 0.8 kg / 24 N.** Reject for iter-026:
  changes the envelope and invalidates current tunings without SITL data
  to justify the specific number. Re-baseline as iter-027+ AFTER SITL.

- **Composer wanted `PlantProfile` named-profile abstraction.** More
  structural but slower to ship. Adopt Composer's *naming discipline*
  (rename drone_spec docstring to "AIGP proxy v1") without the full
  PlantProfile refactor.

## Code-change order (Opus's 10 steps, lightly edited)

1. `competition/drone_spec.py` — add `DEFAULT_BODY_SIZE_M`,
   `DEFAULT_ARM_LENGTH_M`, `DEFAULT_LINEAR_DAMPING`,
   `DEFAULT_ANGULAR_DAMPING`. Rename docstring to "AIGP proxy v1".
2. `sim_pybullet/drone.py` `DroneConfig` — source mass/thrust/gravity
   /arm/body_size from `drone_spec` via `field(default_factory=...)`.
3. `sim_pybullet/drone.py` `QuadrotorDrone` — add `step_reference(...)`
   that wraps an internal `GeometricTracker(TrackerConfig())` and maps
   `AttitudeCommand` → normalized `apply_command(...)`. Estimated
   ~25-40 lines including thrust-N → normalized-throttle math
   (thrust ÷ max_thrust_n) and roll/pitch-rad → normalized [-1, 1] via
   max_tilt_rad.
4. `sim_pybullet/env.py` — add `backend: Literal["aigp_quad","cf2x"] =
   "aigp_quad"` kwarg. Branch construction.
5. `scripts/visual_demo.py` — drop the CF2X TrackerConfig override
   (mass=0.027, max_thrust_n=0.6); use `env.drone.step_reference(...)`
   (or attitude-command path).
6. `scripts/benchmark.py:run_sim_benchmark` — switch to
   `backend="aigp_quad"`; thread `target_acc` from trajectory point.
7. `sim_pybullet/configs/race_01.json` — remove CF2X residue
   (plan_max_speed_mps=4.0, cmd_max_speed_mps=4.0).
8. `competition/pybullet_adapter.py` — audit `send_attitude` post-swap.
9. Tests — extend `test_drone_spec_contract.py`; add
   `test_sim_stack_parity.py`, `test_backend_selection.py`.
10. CLAUDE.md — promote PyBullet to truth tier.

Each step independently revertable.

## Risks

- **race_01 12/12 will likely regress** when CF2X 4 m/s ceiling drops
  and the QuadrotorDrone takes over. Expected — current 12/12 is on a
  permissive kinematic bench. Snapshot new baseline.
- **QuadrotorDrone attitude PD** is hand-tuned (kp=12, kd=4 in iter-021)
  — may need re-tuning at the higher commanded speeds.
- **Mass/thrust still incorrect vs real AIGP** (likely 0.6-0.9 kg / 30-40 N
  for 280 mm class). Land structural unification first; SITL calibration
  iter-027+ rebaselines.

## Estimated effort

- Step 3 alone (step_reference) is ~1 hr of careful coding + testing.
- Full plan: 3-5 iters of disciplined commits.
- Parity test (step 9) is the highest-priority deliverable — if shipped
  alone it converts "matrix passes / demo crashes" from invisible drift
  into a visible CI failure.

## Recommendation for iter-026 today

Ship steps 1, 2, 3 (drone_spec extension + QuadrotorDrone config wiring
+ step_reference method). Defer steps 4-7 (the actual visual_demo /
benchmark swap) to iter-027 — those need empirical validation against
race_01 and may surface attitude PD instability.

Iter-026's parity test deliverable (step 9 partial) should be the
"will-fail-on-main" gate that ratchets future changes toward unified
behaviour.
