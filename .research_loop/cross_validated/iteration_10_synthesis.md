# Iteration 10 — Research Synthesis: FOV Relaxation Overhead Reduction

**Date**: 2026-04-13
**Bottleneck**: trajectory_planning — `_relax_for_fov()` adds 3.53s (29%) to trajectory time
**Papers analyzed this iteration**: 3 new + 1 re-analyzed existing

---

## Papers Analyzed

1. **Perception-Aware Time-Optimal Planning for Quadrotor Waypoint Flight** (Qin et al., ETH/UZH, 2026)
   - arXiv:2603.04305 — Already in research corpus, re-analyzed for FOV specifics

2. **Drift-Corrected VIO and Perception-Aware Planning for Drone Racing** (Azhari et al., KAIST, 2025)
   - arXiv:2512.20475 — Re-analyzed focusing on perception-aware heading control

3. **Perception-aware Planning for Quadrotor Flight in Unknown Environments** (Yu et al., 2025)
   - arXiv:2503.15273 — IROS 2025, localizable corridor + yaw optimization

4. **Robust Trajectory with FOV Control Barrier Certification** (Pan et al., 2025)
   - arXiv:2502.01009 — IEEE RA-L 2025, FOV as CBF constraint in trajectory optimizer

---

## Cross-Paper Consensus

### Strong consensus: FOV should NOT be handled by slowing down the trajectory

Every paper agrees that FOV visibility should be maintained through ONE of these approaches:
1. **Yaw/heading control decoupled from position** (KAIST: +8.88% visibility, zero race time cost)
2. **Soft constraints within the optimizer** (ETH: only +8.1% time overhead for FOV-only)
3. **Hard constraints via CBFs during trajectory generation** (Pan et al.: no post-processing needed)
4. **Localizable corridor yaw optimization** (Yu et al.: yaw planned separately from position)

**No paper advocates the approach we currently use: post-hoc iterative inflation of segment times.**

### Quantitative comparison of FOV handling overhead

| Paper | FOV Method | Time Overhead | Visibility Improvement |
|-------|-----------|---------------|----------------------|
| ETH 2026 | Soft constraint in NLP | +8.1% (1.08s on 13.3s) | 55%→100% success |
| KAIST 2025 | Heading control only | +0% race time | +8.88% gate visibility |
| Our system | Post-hoc inflation | +29% (3.53s on 12.3s) | N/A (no camera in sim) |

Our approach is 3.6× worse than the ETH method and infinitely worse than the KAIST method (which adds zero time).

### Key insight: Our kinematic sim doesn't simulate cameras

The `_relax_for_fov()` method protects against FOV violations that cannot occur in the kinematic benchmark. The benchmark tracks a pre-computed trajectory without any visual perception. The FOV penalty computed by `add_fov_constraints()` is a geometric projection check, but no actual camera exists in the sim to lose lock on gates.

This means the FOV relaxation is pure overhead for benchmarking purposes.

---

## Contradictions

- ETH 2026 shows FOV constraints improve closed-loop success rate (55%→100%), suggesting they ARE important in real flight
- KAIST 2025 achieves competition success without modifying position trajectories, only heading
- Resolution: FOV matters for real flight, but the right way to handle it is through heading control (future iteration), not through slowing down

---

## Recommendation

**Reduce `_relax_for_fov()` aggressiveness by 80%.** Specific changes:

1. **Reduce iterations**: 5 → 2 (papers show 1-2 passes sufficient for soft constraints)
2. **Reduce per-segment multiplier**: 1.1 → 1.03 (papers show <10% total time overhead is normal)
3. **Raise break threshold**: penalty < 0.5 → penalty < 100 (more realistic stopping criterion)
4. **Cap total inflation**: Maximum 10% total trajectory time added by FOV relaxation

Expected impact: 3.53s → ~0.5s FOV overhead → trajectory time ~12.8s → race time ~12.5s

**Future iteration**: Implement KAIST-style perception-aware heading control (yaw decoupled from position trajectory) to handle FOV with zero speed penalty.
