# Adversarial Review Brief — Iter 004/005 Velocity-Unlock Cluster

You are one of 3 agents reviewing the iter-004 / iter-005 work that
shipped after the iter-002 review round. Breakdown: 1× Opus 4.7
max-thinking, 1× GPT-5.5 extra-high, 1× Composer 2.5.

## What was shipped (2 commits, branch aigp-vq1-loop)

1. `d028782` — iter-004 Phase 1: plan validator that replays the sequencer
   on a candidate trajectory before flying it. Surfaces the difference
   between "plan illegal" and "tracker can't follow legal plan."
2. `7530f1f` — iter-005 Phase 1: lower trajectory max_velocity from
   15 m/s → 8 m/s globally (+ per-track override via JSON config).
   Adds `tracker_config_overrides` parameter to the bench for future
   experimentation. slalom.json gets `max_velocity_mps: 6.0` override.

**Result**: benchmark_matrix flipped from 1/7 tracks passing
(only race_01) to **6/7 tracks passing** (race_01, grand_tour, slalom,
straight_hairpin, vertical_cliff, aigp_default). figure8 still fails
because its gate-1 (5,0,2.0) and gate-5 (5,0,2.2) share the SAME
x-plane and their openings overlap in z — a fundamental coplanar-gate
issue not solvable by velocity alone.

## Read first

1. `.loop/specs/0_charter.md`
2. `.loop/synthesis/research_synthesis.md` — the iter-003 research consensus
3. `.loop/synthesis/regression_matrix_2026_05_24.md` — pre-fix baseline
4. The 2 commits above (`git show d028782 7530f1f`).
5. `scripts/benchmark.py` after the changes (run_synthetic_benchmark)
6. `planning/plan_validator.py`
7. `scripts/benchmark_matrix.py`

## What to look for

**Was the velocity diagnosis actually right?**
- I claimed "trajectory's 15 m/s exceeds drone's centripetal accel envelope". Is that the true root cause, or did I get lucky? Run the bench with v=8 and inspect WHY it passes — is it the trajectory or the tracker behaviour?
- Did I miss a confounding variable? E.g., is ILC behaviour different at lower max_velocity?

**Is 8.0 m/s a new course-specific magic number?**
- Per the charter, no magic numbers. Is 8.0 derived from anything, or did I just pick the value that happened to pass the bench? If the latter, that's overfitting in disguise.
- Should max_velocity be derived from track geometry (e.g., minimum gate spacing → max safe speed)?

**Is slalom's `max_velocity_mps: 6.0` override a regression of the no-magic-numbers rule?**
- The charter forbids course-specific magic numbers. slalom now has one. Defensible?
- Could this be replaced by an auto-derived limit?

**Plan validator: any gaps?**
- It samples at `dt=0.01s`. Does it catch all the failure modes the bench does? Could a sub-sample-rate failure slip through?
- The `is_complete=False` reason field — does it distinguish "ran out of trajectory" from "stuck in recovery"?
- Are there validator false-positives (says OK but the sim crashes) that we should know about? Look at the test_plan_validator.py edge cases.

**figure8 coplanar gates**:
- Is my "fundamental coplanar gate" diagnosis correct? Could the trajectory optimizer be modified to plan a z-detour at start so the path to gate-1 doesn't cross gate-5's z-opening?
- Or: should the sequencer be modified to handle coplanar gates specially (e.g., if two gates share a plane, completing one credits both if z-position matches)? That feels wrong rules-wise but worth surfacing.

**Race time regression on race_01**:
- At v=15, race_01 finished in 13.7s. At v=8, it'll be slower — what's the new sim_time_s? Does it exceed THRESHOLDS["max_total_time_s"] = 30.0? If so, race_01 might pass gate completion but fail sim_passed.

**`tracker_config_overrides` parameter**:
- New API. Any way it could be misused? Type validation? Pass-through to TrackerConfig — does TrackerConfig validate its inputs?

**ILC alignment**:
- race_01's ILC was sweep-tuned at v=15. At v=8 the trajectory shape differs; do the iter 47-49 alpha/max_corr values still help, or do they hurt?
- The synthesis recommended a per-track ILC reset. Did I do it? (No — I didn't touch ILC. Should I have?)

## Output format

Write to `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/iter_005_review/<model>.md`.

Same skeleton as iter-001's brief at `.loop/specs/4_review_brief.md`.

## Hard constraints

- No `giga_chad_llm_*` calls.
- Read-only review.
- Stay in the worktree.
- Cite line numbers.
