# Iteration 9 — Cross-Validated Research (Self-Review)

## Synthesis Validated

The core approach (drag compensation → higher feedforward → reduced inflation → faster race) is backed by:
- Faessler 2018: analytical proof that drag-compensated flatness enables accurate tracking
- Zhang 2024: drag-unaware planning causes 83% more tracking error
- Teissing 2024: drag modeling in planner reduces tracking error at high speeds
- Wu 2025: ℒ₁ compensating drag gives 5x tracking improvement

## Critical Review Points

1. **Mathematical certainty of drag compensation**: In our sim, `actual_accel = commanded_accel - 0.5 * vel`. Adding `0.5 * vel` to `commanded_accel` exactly cancels drag. This is NOT an approximation — it's exact cancellation for the kinematic sim's model.

2. **Why not ℒ₁ or learned tracking penalty**: Both are designed for UNKNOWN disturbances. Our drag is known (coefficient=0.5, linear model). Adding complexity (state predictor, filter tuning, neural network) for a known single-parameter problem is over-engineering.

3. **Risk of inflation reduction**: Inflation was added in iters 6-7 when feedforward was BROKEN (ff=0 due to bug). With ff working AND drag-compensated, the controller is fundamentally more capable at turns. Reducing inflation is safe but should be done conservatively.

4. **Two-phase approach reduces risk**: First, add drag compensation and find optimal ff weight (improves accuracy). Then, reduce inflation (trades some accuracy for speed). If step 2 regresses, we can revert just the inflation change and still keep the drag compensation gains.

## Authoritative Recommendation

Implement drag compensation + feedforward sweep first. If ff=0.8+ works, reduce inflation parameters. Target: race time <15.5s with avg error <0.25m.
