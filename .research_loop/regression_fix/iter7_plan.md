# Iteration 7 Fix Plan (revised after codex red-team)

## Original plan (REJECTED after codex P0)
Tangent-aligned entry/exit for off-angle gates. **Rejected** because:
1. Tangent-aligned waypoints would cause polynomial to skim the gate plane — `_check_pass_through()` in `gate_sequencing/sequencer.py:199-223` requires a plane sign change. With gate-10 margin already at 0.08m, tangent-skimming risks breaking gate completion (codex P0).
2. Re-defining the reference to track a skimmier line might improve the reported tracking error metric without actually fixing controller feasibility.
3. Hard threshold at `alignment=0.5` is a cliff; racing-line cache doesn't key on planner behaviour (codex P1).

## Revised plan (codex P2 — slow the helix)

Controller is the bottleneck. Telemetry shows actual drone speed **mean=4.15m/s, max=8.68m/s** vs reference mean=3.51m/s, max=4.00m/s — drone overshoots the reference during helix turns and oscillates. Slowing the reference gives the controller time to track without oscillation.

### Code changes (minimal, targeted)
1. `scripts/visual_demo.py` — reduce `PLAN_MAX_SPEED` and the safety `MAX_CMD_SPEED`:
   - `PLAN_MAX_SPEED: 4.0 → 3.0` (caps SpeedProfiler + `DroneConstraints.max_velocity`)
   - `MAX_CMD_SPEED: 4.0 → 3.0` (caps velocity command to drone.step)
2. `planning/trajectory_optimizer.py` — strengthen helix compression floor:
   - `max_compression_helix: 0.72 → 0.80` (helix retains more inflation ⇒ slower through helix)

### Rationale
- **Research**: ILMPC (Zhao 2025) — adaptive cost gives helix/tracking priority over time. TACO (Sanghvi 2025) — adapt trajectory to controller capability. Per-segment compression matches TOPP-RA practice.
- Leaves normal-crossing entry/exit geometry untouched, so gate-10 0.08m margin is preserved at minimum, likely improves (miss distance is dominated by controller lag, not trajectory geometry).
- Slalom gates were already tracking well (mean 0.585m) — the bottleneck is helix (mean 1.004m). Helix-specific compression bump avoids unnecessarily slowing slalom.
- Race time est: 29.21s → 38-42s (well under 120s limit). No PRD race-time requirement.

### Expected impact
- `avg_tracking_error: 1.52m → ~0.7-0.9m` (approaching 0.5m target)
- Gate-10 miss: 1.12m → ~0.6-0.8m (healthier margin)
- All 12 gates: still pass (preserves normal-crossing geometry)

### Validation
- Run `visual_demo --no-render` 3× consecutively
- Accept if: 12/12 gates × 3 AND avg_tracking_error shows large reduction (target <0.5m; if not reached, progress vs. 1.52m is still valuable)
- Red-team diff with codex post-implementation

### Rollback
Trivial: reset 3 numeric parameters back to their previous values (all comments show the previous value).

### Follow-up (iter 8+ if still above 0.5m)
- Enable ILC offsets in `visual_demo` (exists in `benchmark.py` with `compute_ilc_offset_table`)
- Tune `max_compression_helix` further (up to 0.85) OR push `PLAN_MAX_SPEED` to 2.5
- Investigate `_select_by_sim` using hardcoded `max_velocity=15.0` (codex iter 4 P1)
