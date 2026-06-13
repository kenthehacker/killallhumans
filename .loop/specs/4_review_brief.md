# Adversarial Review Brief — Iter 001

You are one of EIGHT agents tasked with **adversarially reviewing** the
iter-001 implementation of the AI Grand Prix VQ1 stack at
`/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/`. The breakdown
is: 1× Opus 4.7 max-thinking, 2× GPT-5.5 extra-high, 5× Composer 2.5.

Your one job: **find what's broken, missed, over-engineered, or fragile.**
The synthesiser will read all 8 reviews and prioritise fixes for iter-002.

## What was shipped in iter-001 (6 commits on branch aigp-vq1-loop)
1. `1979dd2` — A1-A6: adversarial tests + sequencer in-order DQ + AIGP geometry defaults
2. `2077688` — A7: bench reads sequencer crash + DQ signals; honesty surface exposed
3. `4a782c5` — A8+A9: curvature-derived ILC sections; magic numbers gone
4. `ca8fc45` — A14+A15: ML tracker residual (10→64→3 MLP, hard-clamped, off-by-default)
5. `cd49d89` — A10+A11: vision UDP receiver on :5600 (24B header + JPEG chunk reassembly)
6. `05188a1` — A13+A18: SITL calibration stub + AIGP placeholder track config

Deferred (NOT in scope for this review, but flag if you spot something egregious):
- A12 MAVLink command tests (needs MAVSDK mocks; deferred to iter-002)
- A16 pipeline-pybullet smoke (parallel scaffold for runner.py collapse)
- A17 benchmark_matrix (needs additional track configs)

## Read first
1. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/specs/0_charter.md` — mission + hard constraints
2. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/specs/1_aigp_spec_distill.md` — VADR-TS-002 distilled
3. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/specs/2_known_issues.md` — pre-iter-001 audit (the bar)
4. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/synthesis/iter_001.md` — what was promised
5. The 6 commits above (`git show <hash>`).

## What to look for (NOT exhaustive — find what the others miss)

**Correctness bugs** —
- Sequencer DQ logic: does it actually catch every U-turn / skip pattern? Or does it leak when, say, drone exits and re-enters the same future-gate's plane?
- Crash zone math after the AIGP-geometry default change: race_01 now uses
  border_width 0.18 m; the strut-hit annulus is [opening, opening + 2×border] —
  does the test_crash_does_not_advance_target case still cover the right region?
- ILC section partition: are integer step-conversions monotonic when point timestamps
  are non-uniform? Does the partition handle a trajectory whose first acceleration
  sample is already above the quantile threshold (i.e., no leading "low" section)?
- Vision UDP receiver: what happens if `payload_size` is 0? Or `jpeg_size`
  doesn't equal `sum(chunk payload sizes)`?
- Tracker residual hard clamp: does the clamp compose correctly with the
  pre-existing `max_tilt_rad` clamp? Is there an order-of-clamping bug
  where a 0.85+0.05 = 0.9 rad command could leak through?

**Spec drift** —
- VADR-TS-002 says "fx = fy = 320". We have `CameraIntrinsics.from_fov(...)`
  setting fx via horizontal FoV. Are the two reconciled correctly when both
  paths are used?
- The spec says camera tilt is upward by 20°. Our `pitch_offset_rad =
  math.radians(20.0)` is positive. Does the sign actually push the horizon
  DOWN (image-frame y > cy) per `test_horizon_projects_below_image_center_with_upward_tilt`?
- The spec says 8-min max run. Where is that enforced in our bench /
  pipeline? Is there a missing timeout?

**Over-engineering / over-tight tests** —
- Are any of the new tests testing implementation details that lock us into a
  specific design? Suggest a tighter / looser bar.
- Is the iter-001 honesty contract too strict for race_01 (e.g. will the new
  enforce_in_order=True default DQ a legitimate replanner-driven recovery)?
- Did we introduce a per-course magic number in disguise (e.g. via
  config/ilc_defaults.json defaults that only make sense for the existing
  race_01 helix)?

**Safety / robustness** —
- Tracker residual: is the hard clamp the LAST gate before AttitudeCommand
  emits? Could a future refactor accidentally swap the order?
- Vision UDP: the unbounded `_delivered_ids` list — is the cap actually sized
  appropriately? Memory leak on a long-running connection?
- SITL calibration: are there pathological inputs (e.g. all-zero thrust)
  that produce a NaN in `lstsq`?

**Test coverage gaps** —
- Anything in iter-001 that has NO adversarial test for it?
- Tests that should exist but were deferred to iter-002 (e.g. multi-track
  ILC regression) — are any of them critical?

**Architectural concerns** —
- The synthesis deferred the `runner.py` ↔ `race_pipeline.py` collapse.
  Does the current half-state make iter-002 worse? Should we accelerate
  the collapse, or stay split?
- Does the calibration stub make sense as-is, or does it need to land an
  actual mass-thrust separation method (e.g. measuring at zero thrust)?

## Output format (strict)
Write your review to:
`/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/iter_001_review/<your_model_slug>.md`

The synthesiser will parse `## Findings` sections, so use this skeleton:

```markdown
# Iter 001 Adversarial Review — <model name>

## Summary
<1-3 sentences on what you found. Be honest if iter-001 looks clean — but
look hard before claiming so. The bench is suspect; so is this code.>

## Findings (ordered by severity)

### F1. <Short title> — [BLOCKER | MAJOR | MINOR | NIT]
- **File(s)**: `path/to/file.py:LINE_NUMBER`
- **Issue**: <concrete: what's wrong / missing / fragile>
- **Repro**: <how to trigger / how a future change would surface it>
- **Fix sketch**: <what iter-002 should do — be specific>
- **Confidence**: <high | medium | low — why you might be wrong>

### F2. <…>
…

## Things iter-001 got right
<2-5 bullets. Briefly. The synthesiser uses this to calibrate consensus.>

## What I did NOT review
<Files you didn't open / things you skipped. Honesty here lets the
synthesiser avoid double-counting your review with someone else's.>
```

## Hard constraints
- **No** `giga_chad_llm_investigate` / `giga_chad_llm_code` / `giga_chad_llm_plan` / `giga_chad_llm_scout` / `giga_chad_llm_spawn*` calls.
- **Read-only review**: do NOT edit any source files.
- Stay inside the worktree at `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/`.
- Be specific. "Tests look thin" is useless. "test_X at file:line covers case
  A but not case B, where case B can fire when …" is useful.
- Cite line numbers; the synthesiser cross-references them against the
  current code state.
