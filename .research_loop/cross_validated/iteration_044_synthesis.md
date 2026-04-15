# Iteration 44 — Research Synthesis: Speed Recovery via Inflation Reduction

## Current Problem
Race time stuck at 14.08s with avg tracking error 0.138m. The 0.25m threshold gives 0.112m headroom (81% utilization of error budget). The system is leaving 2+ seconds of race time on the table by over-protecting turn segments.

## Research Consensus (from 118 analyzed papers)

### 1. Post-optimization inflation is the primary speed bottleneck
- **TOGT Planner (Qin 2024)**: Time-optimal trajectories should respect actual dynamic constraints, not conservative proxies. Our post-optimization inflation adds 7-25% time padding on top of what the L-BFGS optimizer produces — this is NOT research-backed beyond initial stabilization.
- **CPC (Foehn 2021)**: Complementary Progress Constraints allow trading path accuracy for speed. The key insight: when tracking error is below threshold, speed can be increased.
- **TACO (Sanghvi 2025)**: Trajectory-aware controller optimization shows that controller capability should dictate trajectory aggressiveness, not fixed inflation factors. Our ILC proves the controller CAN track faster trajectories.
- **MonoRace (2026)**: A2RL competition winner uses NO post-processing FOV relaxation or turn inflation. The trajectory optimizer handles everything.

### 2. ILC compensates for increased speed
- **Schoellig 2012**: ILC reduces systematic tracking error by 87% in 3-5 iterations. Our ILC currently reduces error from ~0.25m (no ILC) to 0.138m — a 45% reduction.
- **Spatial ILC (Lv 2023)**: Time-optimal spatial ILC progressively increases speed while maintaining tracking within a virtual tube. This is EXACTLY our approach: reduce inflation → ILC re-converges on faster trajectory.
- **Track-Centric ILC (2026)**: Section-specific ILC naturally adapts to different section speeds.

### 3. Inflation reduction is safer than TOPP floor changes
- **Failed approach iter 29-32**: TOPP floor changes > 1% trigger racing line basin switching. But inflation changes DON'T affect TOPP floors — they modify the input times to TOPP. Since the racing line is cached (iter 33), inflation changes only affect the trajectory timing.
- **FBGA (Piazza 2025)**: Forward-backward velocity profiling naturally handles variable segment times. Reducing inflation just means TOPP receives shorter input times and compresses less.

### 4. Per-gate headroom analysis supports selective reduction
| Gate | Avg Error | Headroom to 0.25m | Can Speed Up? |
|------|-----------|-------------------|---------------|
| gate-1 | 0.113m | 0.137m | Yes |
| gate-2 | 0.214m | 0.036m | Minimal |
| gate-3 | 0.191m | 0.059m | Small |
| gate-4 | 0.142m | 0.108m | Yes |
| gate-5 | 0.139m | 0.111m | Yes |
| gate-6 | 0.074m | 0.176m | Lots |
| gate-7 | 0.164m | 0.086m | Moderate |
| gate-8 | 0.128m | 0.122m | Yes |
| gate-9 | 0.109m | 0.141m | Yes |
| gate-10 | 0.144m | 0.106m | Yes |
| gate-11 | 0.116m | 0.134m | Yes |
| gate-12 | 0.141m | 0.109m | Yes |

Gate-2 and gate-3 have least headroom. The inflation factors around these gates should be reduced less aggressively.

## Proposed Direction
**Reduce all turn inflation factors by ~40-50%**, leaving TOPP floors unchanged. This avoids the known basin-switching failure mode (iters 29-32, 39) while achieving significant race time reduction. The ILC will partially compensate for increased tracking error.

## Contradictions & Risks
- **Risk**: Inflation reduction could push gate-2/3 error over 0.25m threshold if ILC doesn't compensate sufficiently. Mitigation: Keep S-turn inflation reduction more conservative (~25%).
- **Contradiction**: Iter 14 showed uniform compression fails because turn segments are at dynamic limits. But we're NOT doing uniform compression — we're reducing the PROTECTIVE inflation that was added AFTER L-BFGS optimization.
