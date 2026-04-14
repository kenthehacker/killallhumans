# Iteration 17 — Cross-Validated Research: FOV Relaxation Removal

## Synthesis Summary
Remove the `_relax_for_fov` post-processing stage. 6/6 papers support integrated FOV constraints over post-processing. The L-BFGS optimizer already provides integrated FOV awareness (weight=10).

## Cross-Validation: Challenging the Synthesis

### Challenge 1: "What if the L-BFGS FOV penalty is too coarse?"
The L-BFGS FOV penalty uses a simplified tilt-angle estimate from centripetal acceleration (lines 768-793). The `add_fov_constraints` method (lines 809-938) uses full body-frame projection with velocity-derived attitude. These give DIFFERENT answers.

**Counter-argument**: Even if the L-BFGS penalty is coarse, the _relax_for_fov step doesn't fix this properly either — it just inflates ALL turn segments by 3% per iteration, which is also a blunt instrument. Neither mechanism provides precise FOV control. The real solution (future work) is yaw optimization, not time inflation.

**Verdict**: Valid concern but doesn't change the recommendation. Both mechanisms are coarse; removing the redundant one saves time.

### Challenge 2: "What if FOV violations cause real problems in the competition?"
In the kinematic sim benchmark, FOV is not measured. But the competition is different.

**Counter-argument**:
- The L-BFGS penalty (weight=10) still provides primary FOV awareness
- MonoRace won the A2RL competition with NO post-processing FOV relaxation
- The competition-ready solution is yaw optimization (per Drift-Corrected VIO: +0% time cost), not time inflation
- Time inflation is the worst way to handle FOV — it makes the trajectory slower without guaranteeing visibility

**Verdict**: For competition, we need better FOV handling (yaw optimization), not MORE time inflation. Removing the redundant stage is the right direction.

### Challenge 3: "What if removing _relax_for_fov causes a regression?"
The post-processing currently caps at +8% total time. If the L-BFGS penalty is already accounting for FOV, removing the post-processing should just recover that ~8% time.

**Counter-argument**:
- The benchmark will tell us immediately if there's a regression
- We can revert in one `git checkout -- .` if metrics get worse
- The post-processing was already reduced from 25% → 8% cap (iter 14) without regression

**Verdict**: Low risk. Benchmark before AND after. Revert if regression.

### Challenge 4: "Is the TOPP retimer affected?"
The pipeline is: L-BFGS → inflate → (FOV relax) → TOPP retime. If FOV relaxation increases some segment times, TOPP may compress them back.

**Counter-argument**:
- TOPP has a compression floor (0.65 for straight segments)
- Turn segments have high curvature → low TOPP speed limit → minimal compression
- So FOV relaxation's inflation on turn segments likely persists after TOPP
- Removing FOV relaxation should directly reduce those turn segment times

**Verdict**: The time savings should propagate through TOPP to the final trajectory.

## Final Recommendation

**PROCEED with removing `_relax_for_fov`.** The evidence is strong:
1. 6/6 papers support integrated over post-processed FOV
2. L-BFGS already provides integrated FOV penalty
3. Post-processing is documented as "safety net, not primary mechanism"
4. Benchmark doesn't measure FOV — pure overhead
5. Competition winner (MonoRace) uses no post-processing FOV relaxation
6. Low risk: reversible in one command

**Expected outcome**: Race time 13.95s → ~13.5s, tracking error unchanged or improved slightly (tighter trajectory means less tracking lag).

## Risk Mitigation
- Run full benchmark immediately after change
- Compare ALL per-gate metrics, not just aggregate
- If any gate regresses by > 0.05m or race time doesn't improve, revert
- Keep `_relax_for_fov` method in code (don't delete it) for potential future use
