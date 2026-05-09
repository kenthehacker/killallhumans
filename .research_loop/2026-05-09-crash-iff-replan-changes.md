# 2026-05-09 — Crash + iff-highlighted + dynamic-replan changes

Snapshot of the diff for the parallel cross-validation run. Subagents
should treat this as the authoritative description of what changed.

## Goals

1. Tests determine whether a drone crashed into a gate.
2. Tests report a drone passed through a gate **iff** that gate was the
   highlighted (current target) at the time of crossing.
3. The drone can dynamically alter its racing line when something
   changes — collision, deviation, or a missed highlighted gate.

## Files changed

- `gate_sequencing/sequencer.py`
  - `GateSpec`: added `border_width: float = 0.15`,
    `outer_width`, `outer_height` properties.
  - `GateSequencer`:
    - state: `_crashes`, `_misses`, `_last_event`.
    - `update()` now classifies any plane-crossing of the *highlighted*
      gate that didn't credit a pass into either `'crash'` (crossing
      inside outer frame, outside opening) or `'miss'` (outside outer
      frame).
    - new accessors: `crashed_gate_ids`, `last_crash`, `missed_gate_ids`,
      `last_event`, `passed_gate_ids`.
    - new method: `mark_collision(gate_id, position=None)` —
      authoritative crash mark for physics-driven sources.
    - new helpers: `_plane_was_crossed`, `_compute_crossing`,
      `_point_in_outer_frame`. `_check_pass_through` now delegates to
      `_compute_crossing`.
    - `reset()` clears crash/miss state.
- `sim_pybullet/sequencer.py`
  - Same crash/miss surface as the platform-agnostic sequencer.
  - `__init__` accepts `pass_through_margin: float = 1.5` (was hardcoded).
  - `update()` classifies `'pass'`/`'crash'`/`'miss'`.
  - new properties: `crashed_gate_ids`, `last_crash`, `missed_gate_ids`,
    `last_event`, `state` (lightweight RaceState shim), `gates_passed`.
  - new methods: `mark_collision`, `_compute_crossing`,
    `_point_in_outer_frame`.
- `planning/dynamic_replanner.py` (new)
  - `ReplanConfig` (cooldown_seconds, lateral_error_threshold_m,
    sustained_frames).
  - `ReplanTrigger` (gate_collision, missed_gate, off_track,
    sustained_lateral_error, crashed_gate_id; `triggered`, `reasons`).
  - `DynamicReplanner` — stateful. Methods: `evaluate`,
    `should_replan`, `waypoints_for_replan`, `mark_replanned`, `reset`.
    Properties: `replan_count`, `last_trigger`.
  - `_gate_centre()` helper that accepts both GateSpec (`.position`)
    and Gate (`.pose.x/y/z`) shapes.
- `sim_pybullet/runner.py`
  - Imports `DynamicReplanner`, `ReplanConfig`.
  - `RaceRunner.__init__` constructs the replanner, tracks
    `_replan_count`, `_last_replan_reasons`, `_last_contact_gate_id`.
  - Main loop now polls `env.gate_contact()` each tick and dedupes
    sustained PyBullet contact manifolds — first tick of a new contact
    calls `sequencer.mark_collision`.
  - New method `_maybe_replan(sim_time, drone_state)` runs every tick:
    evaluates the trigger, and on `should_replan` rebuilds the
    `RacingLine` from `[drone_position, ...remaining_gate_centres]`,
    resets target-altitude slew bookkeeping, marks replanned, redraws.
  - New property `RaceRunner.crashed_into_gate` returns
    `sequencer.last_crash[0]` or None.
  - `_reset()` resets the replanner.

## Tests added

- `gate_sequencing/tests/test_sequencer.py` — `TestCrashIntoGate` (8),
  `TestPassIfAndOnlyIfHighlighted` (5).
- `planning/tests/test_dynamic_replanner.py` (new) — 14 tests:
  triggers, cooldown, waypoints, counters, reset.
- `sim_pybullet/tests/test_sequencer_crash_miss.py` (new) — 14 tests:
  crash detection, iff-highlighted, non-pass non-advance.
- `sim_pybullet/tests/test_runner_replan_integration.py` (new) —
  7 tests: gate_contact → mark_collision wiring, replanner trigger,
  replanned line starts at drone position, replan count persistence.

## Test results

`python3 -m pytest --ignore=gate_detection -q` → **236 passed**.

## Design notes

- The 1.5× pass-through margin in the production `sim_pybullet`
  sequencer makes the geometric crash zone empty for typical border
  widths. `mark_collision` (driven by PyBullet's contact manifold) is
  the authoritative crash signal in production. The tests use
  `pass_through_margin=1.0` to exercise geometric classification.
- Replanner trigger logic is purely pure-function over the sequencer
  surface — testable without PyBullet.
- The replan cooldown (default 0.5 s) prevents replanning storms when a
  single perturbation persists across multiple ticks.
- A replan resets the sustained-lateral-error counter — error against
  the *old* line is meaningless once the line has been rebuilt.
