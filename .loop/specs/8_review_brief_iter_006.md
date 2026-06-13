# Adversarial Review Brief — Iter 006 (auto-velocity + airspace + matrix-strict)

You are one of 3 reviewers on the iter-006 cluster. Breakdown: 1× Opus
4.7 max-thinking, 1× GPT-5.5 extra-high, 1× Composer 2.5.

## What was shipped (2 commits, branch aigp-vq1-loop)

1. `06ba521` — iter-006 F3 (consensus MAJOR from iter-005 review):
   removed the 8.0 / per-track 6.0 magic numbers; replaced with a
   centripetal-acceleration-limited `derive_safe_max_velocity()` based
   on minimum bend radius across the gate triplets.
2. `4b6cef7` — iter-006b F5 + F11: validator now flags ground/ceiling
   airspace exits (matching the bench's z<0.05 / z>20 checks);
   benchmark_matrix marks a track as regressed when sim_passed=True
   but plan_validation.ok=False.

Multi-track benchmark matrix after this iter:
- race_01: 12/12 PASS (18.1s, slower than v=15 baseline 13.7s)
- grand_tour: 14/14 PASS
- slalom: 8/8 PASS
- straight_hairpin: 6/6 PASS
- vertical_cliff: 4/4 PASS
- aigp_default: 6/6 PASS
- figure8: 1/8 FAIL (coplanar-gate edge case, deferred)

## Read first

1. `.loop/specs/0_charter.md` (no-magic-numbers rule)
2. `.loop/synthesis/iter_002_review.md` + `.loop/synthesis/research_synthesis.md`
3. `planning/auto_velocity.py` (new)
4. `tests/test_auto_velocity.py`
5. `planning/plan_validator.py` (airspace bounds added)
6. `scripts/benchmark.py` — the new derive_safe_max_velocity call site
7. `scripts/benchmark_matrix.py` — the plan_validation regression check

## What to specifically hunt for

**The centripetal physics**:
- Is the formula `r ≈ chord / (2·sin(β/2))` correct as an inscribed-arc radius? Check against a known case (90° turn at 1m spacing → r=?).
- `safety_factor=0.8` — is this just a new magic number? Argue both sides.
- `absolute_cap_mps=15.0` — also a magic number. Justified by what?
- For race_01 the derived value is ~10.8 m/s but the new race time is 18s vs 13.7s at v=15. Did the matrix actually use the auto-derived value, or is something else throttling? Maybe planner's `plan_max_speed_mps` default still wins.

**Edge cases in derive_safe_max_velocity**:
- What if all bend angles are below `_MIN_BEND_RAD = 5°` but still measurable curvature? Returns absolute cap; is that right or should we use the smallest measurable bend?
- Hairpin turn (β near π): `sin(β/2)` near 1, so r → chord/2 (smallest). Does the formula behave at extremes?
- What if gates 1 and 3 are coincident (back-to-back hairpins where ABC are collinear with B at the apex)? Look for divide-by-zero or NaN paths.

**Plan validator airspace bounds**:
- z=-2 (NED) vs z=2 (z-up) — the test stubs in `tests/test_plan_validator.py` were changed from -2 to +2 to match the bench's z-up convention. But `gate_sequencing/tests/test_sequencer_adversarial.py` still uses position z=-2. Is the codebase consistent on convention? If a competition track config uses NED (z down → negative), would the validator's z<0.05 ground check fire spuriously?
- `ground_z_threshold=0.05` / `ceiling_z_threshold=20.0` — these are bench-convention magic numbers. Are they overridable from CallSite? Are they baked into validate_trajectory's signature?

**Benchmark matrix strict-on-plan**:
- The new check is `if pv.get("ok") is False`. What about None (no plan_validation field)? Is that treated as OK or treated as missing-data?
- For the placeholder track (aigp_default), is the new strict check applied or skipped?

**Cross-iteration drift**:
- The race_01 time regression from 13.7s → 18.1s: is the CLAUDE.md "<14s aspirational" target still hit anywhere? The Opus iter-005 review F2 flagged it. Did we just accept the regression or address it?
- ILC was tuned at v=15. At auto-derived v~10.8 the sweep-tuned alpha/max_corr values may now be misaligned. Check if iter-006 made things worse for race_01.

**figure8**:
- Still failing. Was that worked-on at all in iter-006? Document the deferral or fix.

## Output format

Write to `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/iter_006_review/<model>.md`.

Same skeleton as the iter-001 brief (`.loop/specs/4_review_brief.md`).

## Hard constraints

- No `giga_chad_llm_*` calls.
- Read-only review.
- Stay in the worktree.
- Cite line numbers.
