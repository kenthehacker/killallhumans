# Iteration 30 — Research Synthesis: Inflation Reduction Round 2

## Thesis
Iteration 29 demonstrated that reducing post-optimization inflation by 1-3% per parameter is safe and recovers race time (14.03→13.80s) while ILC absorbs the accuracy regression (0.175→0.185m). Round 2 applies the same methodology for another 1-2% per parameter, targeting race time ~13.6s with avg error <0.20m.

## Research Base (cumulative from 84 papers)

### Papers supporting this iteration's direction

1. **ILMPC Drone Racing (arXiv:2508.01103)** — Iterative performance improvement through progressive constraint relaxation. Achieved 60.85% improvement through iterative lap time optimization on simulated trajectories. Confirms: iterative, incremental speed improvement is the right paradigm.

2. **Spatial ILC Virtual Tube (Lv 2023)** — Time-optimal ILC within safety tube. As ILC converges, the tube can be tightened (which translates to: margins can be reduced when tracking improves). Core theoretical backing for our approach.

3. **SPIRAL Self-Play Racing (2025)** — Progressive speed improvement through incremental complexity. Start conservative, gradually push speed. Iteration 29's success (1-3% per parameter) and iteration 29's failure (>3% causes basin switching) are perfectly aligned with this paradigm.

4. **Track-centric Iterative Learning (arXiv:2601.21027, 2026)** — Global trajectory optimization using wavelet transforms + Bayesian optimization within an iterative learning framework. Achieved 20.7% lap time improvement. Key insight: iteratively learn and optimize the full trajectory based on execution data.

5. **IteraOptiRacing (arXiv:2507.09714, 2025)** — Unified planning-control framework using historical lap data for iterative improvement. Confirms: using execution history to progressively optimize trajectory timing.

6. **FBGA (Piazza 2025)** — Forward-backward velocity profiling matches optimal control within 0.36%. Compression floors are the binding constraint for race time.

7. **COP Pareto Planning (Tzoumanikas 2022)** — Pareto-aware accuracy-speed tradeoff. Our current operating point (0.185m, 13.80s) has clear room to move along the Pareto frontier.

### Consensus
- **7/7 papers** support progressive, incremental speed optimization after tracking accuracy improves
- **No contradictions** in the literature — the only question is rate of reduction
- Iteration 29 empirically validated: 1-3% per parameter is safe, >3% causes basin switching

### Safety analysis
- Current avg error: 0.185m, threshold: 0.25m → **0.065m headroom**
- Expected regression from round 2: +0.01m → 0.195m, still 0.055m from threshold
- Worst gate (gate-7): 0.282m — not affected by inflation changes (helix geometry issue)
- Most sensitive gates (gate-5, gate-8): currently at 0.167m and 0.224m — even +20% would keep them under 0.27m

## Recommended Changes

### S-turn inflation (`_inflate_sharp_turns`)
| Parameter | Current | Proposed | Delta |
|-----------|---------|----------|-------|
| junction inflate | 1.10 | 1.08 | -2% |
| standard inflate | 1.08 | 1.06 | -2% |
| approach decel | 1.02 | 1.01 | -1% |
| departure (pure) | 1.03 | 1.02 | -1% |
| departure (junction) | 1.01 | 1.005 | -0.5% |

### TOPP compression (`_topp_retime`)
| Parameter | Current | Proposed | Delta |
|-----------|---------|----------|-------|
| protected floor | 0.66 | 0.64 | -2% |
| easy floor | 0.60 | 0.58 | -2% |

### End speed (optional, low risk)
| Parameter | Current | Proposed | Delta |
|-----------|---------|----------|-------|
| end speed factor | 0.65 | 0.70 | +5% |

End speed only affects the last 1-2 segments and cannot cause basin switching (it's in the backward pass only).

## Risk Mitigation
1. **Test TOPP changes first** (independent of S-turn inflation)
2. **Test S-turn changes second** (add on top)
3. **If combined >3% total effective change on any single segment, split into two sub-steps**
4. **Racing line basin switching**: All individual parameter changes ≤2%, should be safe per iter 29 lesson
5. **End speed change is independent** — only affects backward pass terminal condition
