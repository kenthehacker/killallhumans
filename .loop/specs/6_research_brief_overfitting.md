# Research Swarm Brief — Multi-Track Generalization Stall

You are one of 4 research agents (1× Opus 4.7 max-thinking, 1× GPT-5.5
extra-high, 1× Gemini 3.1 Pro, 1× Composer 2.5) asked to surface
research-proven techniques for drone racing controller generalization.

## The stall

After iter-001 → iter-003 (17 commits, 346 tests green, honest bench
infrastructure), the synthetic benchmark matrix
(`scripts/benchmark_matrix.py`) shows:

| Track             | Gates    | Outcome                          |
|-------------------|----------|----------------------------------|
| race_01 (tuned)   | 12/12    | sim_passed=True (race_complete)  |
| figure8           | 0/8      | DQ out_of_order:gate-2, 1.7s     |
| grand_tour        | 1/14     | DQ out_of_order:gate-3, 2.7s     |
| slalom            | 0/8      | CRASH gate-8 strut, 4.8s         |
| straight_hairpin  | 1/6      | DQ out_of_order:gate-3, 2.9s     |
| vertical_cliff    | 0/4      | DQ out_of_order:gate-2, 2.4s     |
| aigp_default      | 0/6      | CRASH gate-1 strut, 1.2s         |

Only race_01 completes. Every other track fails 0-17%, most via
out-of-order DQ in <3 seconds. **The bench is honest** — tests are
green, the sequencer correctly catches these failure modes, and
iter-002/003 closed all the BLOCKER/MAJOR bugs from two adversarial
review rounds. **The controller is overfit to race_01.**

## What's in the stack right now

- **Trajectory**: time-optimal polynomial trajectory through gate waypoints
  (`planning/trajectory_optimizer.py`), with offline per-section ILC
  cross-track corrections (`planning/ilc_sections.py`, hyperparameters
  in `config/ilc_defaults.json`).
- **Racing line**: Catmull-Rom through gate centers
  (`planning/racing_line.py`), speed-profiled per curvature.
- **Tracker**: geometric tracker (Lee 2010 SE(3) style),
  PD on position error + acceleration FF + optional ML residual
  (`control/mpc_tracker.py`). No MPC.
- **Sequencer**: strict in-order DQ on any future-gate opening crossing
  (`gate_sequencing/sequencer.py`). This DQ logic is correct per the
  AIGP competition rules.
- **Bench**: kinematic 2nd-order drone with drag,
  `pass_through_margin=1.0`, AIGP geometry defaults (1.5 m gates).

## Why it overfits

The trajectory optimizer's first segment, on tightly-packed tracks
(slalom, figure8, vertical_cliff), produces a path whose initial
acceleration phase overshoots gate-1 enough that it crosses gate-2's
opening before gate-1 is credited. The kinematic drone follows that
trajectory and trips the DQ.

The ILC default hyperparameters (alpha=0.4, max_corr=0.15) work for
race_01's wide spacing but produce cross-track corrections that push
the trajectory through future gates' planes on tighter geometries.

## Your job: surface research-proven techniques

Look at the academic literature, recent papers, robotics conferences,
and well-known systems. **Find concrete techniques that solve this
class of problem.** For each candidate, give:

1. The paper / system (with citation)
2. What it does, in 1-3 sentences
3. Why it would unblock our specific stall (gates-in-order without
   overshoot on tight geometries)
4. Implementation cost (S / M / L) in our existing Python stack
5. Expected gain (qualitative: small / medium / large)
6. Hard constraints we should worry about (data dependency,
   compute budget, real-time guarantees)

**Cover both classical control and learning-based approaches.** Don't
only give RL or only give classical — diversity matters.

Candidate areas to consider (NOT exhaustive — find what we miss):
- Time-optimal gate-traversing planners with corridor constraints
  (e.g., TOGT, FAST-Racing)
- Model Predictive Contouring Control (MPCC++) with safety tunnels
- Perception-aware planning (Bauersfeld 2021, Foehn 2022)
- Learned residual policies on top of geometric controllers
  (NGTC Pries 2025, On-Your-Own Romero 2025)
- Differentiable physics for trajectory optimization
- Cross-validation harnesses that fit a policy across multiple tracks
  (multi-task RL, domain randomization)
- Spatial / arc-length parametrised trajectories
- ILC variants robust to track topology change
  (Bristow & Alleyne, Zhang 2024)

## Output format

Write to `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/research/<your_model>.md`.

Skeleton:

```markdown
# Multi-Track Generalization Research — <model>

## Summary (3-5 sentences)
<Your overall read on the stall + top recommendation.>

## Top 3 candidate techniques (ranked)

### C1. <Technique name>
- **Citation**: <paper / system reference>
- **What it does**: <1-3 sentences>
- **Why it unblocks our stall**: <specific link to the overshoot /
  out-of-order DQ pattern>
- **Cost (S/M/L)**: <effort>
- **Expected gain (small/medium/large)**: <effect>
- **Risks/constraints**: <data, compute, real-time>

### C2. <…>
### C3. <…>

## Other candidates (don't pick, but flag)
<2-5 more, briefer.>

## What NOT to do
<Approaches that look attractive but won't work for our setting —
e.g., heavy RL when we have no GPU budget.>

## My #1 pick if I had to ship one in iter-004
<Concrete recommendation + sketch implementation plan.>
```

## Hard constraints
- **No** `giga_chad_llm_*` tool calls.
- Stay inside the worktree at `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/`.
- Read-only research — do NOT edit source files.
- Cite real papers with year + venue. Don't invent.
- If you must search the web, do it via `WebFetch` or your model's own browsing.
