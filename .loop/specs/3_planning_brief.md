# Planning Brief — Iteration 001 (give to each cursor agent)

You are one of three agents asked to **propose the next concrete actions** for the
AI Grand Prix VQ1 stack at `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/`.
The main loop will synthesise your plan with the other two agents into a single
plan; you do NOT need to coordinate with the others.

## Read first (in this order)
1. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/specs/0_charter.md`
2. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/specs/1_aigp_spec_distill.md`
3. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/specs/2_known_issues.md`
4. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/ARCH.md`
5. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/PLAN.md`
6. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/race_pipeline.py`
7. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/gate_sequencing/sequencer.py`
8. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/scripts/benchmark.py`
9. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/sim_pybullet/runner.py`
10. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/competition/mavlink_bridge.py`
11. `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/competition/adapter.py`
12. Sample at least one of `planning/trajectory_optimizer.py`, `planning/racing_line.py`, `estimation/ekf.py`, `gate_detection/src/` to ground your plan in real code.

## Your job
Produce a **prioritised, executable action plan** for iteration 001 that:

1. **Fixes the testbench honesty bugs** (I-1, I-2) so a future PASS actually means PASS.
2. **Removes course-specific magic numbers** (I-3, I-4) without losing performance on race_01.
3. **Removes hard-coded gate / drone / camera mismatches** (I-5, I-6, I-8) so the stack reads the AIGP geometry as ground truth.
4. **Collapses runner.py onto race_pipeline.py** (I-7) — only if the cost is low; otherwise leave it for a later iter and say so.
5. **Lands the MAVLink2 + vision-UDP plumbing** (I-9) so we can actually talk to the DCL sim.
6. **Adds ONE lightweight ML model** (I-10) that materially helps — pick the cheapest of: tiny CNN gate-corner regressor, learned residual on the tracker, learned residual on the EKF, drone-dynamics regression for sim-to-real calibration.
7. **Adds adversarial tests FIRST** (I-11) so each fix above clears a concrete bar.

## Output format (strict — the main loop parses your file)
Write to `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/iter_001_plan/<your_model_slug>.md`.

Use **exactly** this markdown skeleton:

```markdown
# Iter 001 Plan — <model name>

## Summary
<1-3 sentences on your overall approach.>

## Actions (ordered)
### A1. <Short title>
- **File(s)**: `path/to/file.py:LINE`
- **Change**: <concrete: add function X, modify Y to do Z, delete W>
- **Rationale**: <which I-N issue this resolves, plus any paper / pattern>
- **Test**: <name of the adversarial test that must pass after the change>
- **Risk**: <one sentence — what could break>
- **Effort**: S | M | L

### A2. <…>
…

## ML choice
<Which ML option from item 6 you picked and why. Sketch the data path and
the training/eval strategy in ≤ 10 lines.>

## What NOT to do this iter
<List 2-3 tempting changes you explicitly defer, with a one-line reason.>

## Open questions for the synthesiser
<Anything where you'd want the synthesiser to decide between two roads.>
```

Do not edit any source files yet. Plan only. Keep the whole file ≤ 1200 lines.

## Hard constraints you must respect
- **No** `giga_chad_llm_investigate` / `giga_chad_llm_code` / `giga_chad_llm_plan` / `giga_chad_llm_scout` / `giga_chad_llm_spawn*` calls. Read-only RAG / code-graph / widget tools are OK if you really need them.
- **No** new course-specific magic numbers.
- **No** "just rerun the benchmark and see" — the bench is suspect; assume PASS means nothing until your adversarial test passes too.
- Stay inside the worktree at `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/`.
