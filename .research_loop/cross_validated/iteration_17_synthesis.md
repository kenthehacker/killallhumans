# Iteration 17 — Research Synthesis: FOV Relaxation Removal

## Bottleneck
trajectory_planning — Remove the `_relax_for_fov` post-processing stage to recover ~0.5s race time.

## Papers Analyzed (6 total, 3 new + 3 previously analyzed)

### New papers (this iteration)
1. **MonoRace** (Bahnam et al., 2026) — A2RL competition winner, monocular pipeline
2. **PA-MPPI** (Zhai et al., 2025) — Perception-aware MPPI from UZH RPG
3. **Mastering Diverse Tracks** (Yu et al., 2025) — RL-based vision racing

### Previously analyzed (directly relevant)
4. **Perception-Aware Time-Optimal Planning** (Qin et al., ETH/UZH 2026) — FOV soft constraints
5. **FOV CBF Quadrotor** (Pan et al., 2025) — Control barrier function for FOV
6. **Drift-Corrected VIO** (2025) — Heading-based FOV control

---

## Consensus Across Papers

### 1. FOV awareness should be INTEGRATED, not post-processed

All six papers agree on one fundamental principle: **perception constraints belong inside the planning/control loop, not as a separate post-processing stage.**

- **ETH 2026**: Uses FOV as soft constraints inside the NLP optimizer (FOV-only adds +8.1% to trajectory time)
- **FOV CBF 2025**: Embeds FOV as HOCBF hard constraints in MPC — explicitly advocates eliminating post-processing
- **PA-MPPI 2025**: Perception cost is embedded directly in the MPPI sampling cost function
- **MonoRace 2026**: Uses adaptive image cropping guided by EKF state predictions — perception handled in the pipeline, not post-hoc
- **Mastering Diverse Tracks 2025**: Perception reward (r_perception = 0.1 weight) baked directly into RL reward
- **Drift-Corrected VIO 2025**: Heading-based FOV control adds +0% to race time

**Conclusion**: Our `_relax_for_fov` post-processing stage is architecturally backwards. The L-BFGS optimizer already has a FOV penalty (weight=10) that provides the "integrated" approach. The post-processing step is redundant overhead.

### 2. Post-processing FOV relaxation adds time with minimal perceptual benefit

- **ETH 2026 data**: FOV-only constraint adds +8.1% when done properly inside the optimizer. Our post-processing cap is also 8%. This means we're paying the time cost TWICE — once in the L-BFGS optimizer (which slows segments to reduce FOV penalty) and again in the post-processing step.
- **Drift-Corrected VIO**: Heading control (not position slowing) is sufficient for FOV — position trajectory doesn't need slowing for perception.
- **MonoRace**: Achieved champion-level performance (16.56s lap, 28.23 m/s peak) with NO post-processing FOV relaxation at all. Their perception robustness comes from adaptive cropping + EKF prediction + multi-gate PnP.

### 3. For our kinematic simulation benchmark, FOV is not measured

Critical point: Our benchmark measures tracking error, gate pass rate, race time, and EKF uncertainty. **FOV visibility is not measured in the benchmark.** The `_relax_for_fov` stage adds time to improve a metric that is not evaluated. This is pure overhead in the current optimization loop.

For the real competition, FOV matters — but it should be addressed through the L-BFGS integrated penalty (already present) and potentially through yaw optimization (future work), not through trajectory time inflation.

---

## Contradictions

### Minor: How much overhead does FOV awareness add?
- **ETH 2026**: +8.1% for FOV-only, +17.4% for full perception awareness
- **Drift-Corrected VIO**: +0% (heading-based, not position-based)
- **PA-MPPI**: ~5-10% overhead but via sampling, not optimization

**Resolution**: The overhead depends on HOW FOV is implemented. Position-based FOV (slow down turns) costs 8-17%. Heading-based FOV (adjust yaw) costs ~0%. Our L-BFGS penalty is position-based (tilt estimate → time allocation). The post-processing step adds more position-based overhead on top.

### No contradictions on the core question
No paper advocates for post-processing FOV relaxation as a separate stage. All advocate integration.

---

## Ranked Actionable Takeaways

1. **Remove `_relax_for_fov` call** (Priority: IMMEDIATE)
   - The L-BFGS FOV penalty (weight=10) already provides integrated FOV awareness
   - Post-processing adds ~0.5s of redundant time inflation
   - Expected: race time ~13.5s (from 13.95s), tracking error unchanged
   - Risk: LOW — L-BFGS penalty is the primary mechanism; post-processing is documented as "safety net"

2. **Verify L-BFGS FOV penalty adequacy** (Priority: VERIFICATION)
   - After removing `_relax_for_fov`, check if `add_fov_constraints()` penalty increases significantly
   - If penalty remains < 100 (current threshold for triggering relaxation), the L-BFGS penalty is sufficient

3. **Future: Add yaw optimization for FOV** (Priority: LATER)
   - MonoRace and Mastering Diverse Tracks both use heading/yaw alignment with gates
   - This would add FOV awareness with +0% time overhead (per Drift-Corrected VIO finding)
   - Not for this iteration — would be a separate bottleneck

---

## Proposed Implementation Direction

**Remove the `_relax_for_fov` call from the `generate()` method.** This is a minimal, low-risk change:
- Delete lines 276-284 in trajectory_optimizer.py (the FOV penalty check and relaxation call)
- Keep the `_relax_for_fov` method and `add_fov_constraints` method in the code for future use
- The pipeline becomes: L-BFGS → inflate_sharp_turns → generate → TOPP retime → final generate

**Research evidence**: 6/6 papers support integrated over post-processed FOV. The L-BFGS optimizer already provides the integrated approach. The competition winner (MonoRace) uses no post-processing FOV relaxation.
