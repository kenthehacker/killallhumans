# iter-027: ML training pipeline — end-to-end runnable, but net negative on matrix

## What shipped

- `scripts/train_tracker_residual.py` — numpy-only Feedback-Error
  Learning trainer for the 10→64→3 `TrackerResidualMLP`. Takes a
  collected dataset .npz from `scripts/collect_residual_dataset.py`
  (iter-025), derives per-sample (delta_roll, delta_pitch, delta_thrust)
  targets from observed position errors (linearised around hover),
  runs 200 epochs of batch SGD with MSE loss + validation split.
  Writes weights to `control/residual_weights.npz`.

- `control/mpc_tracker.py:track()` extension — when the residual is
  active, the residual's (delta_roll, delta_pitch, delta_thrust) is
  also projected into a WORLD-frame acceleration delta and added to
  `self._last_accel_des`. Without this projection the residual was
  invisible to the synthetic kinematic bench (which consumes
  `last_desired_acceleration`, not the attitude command).

## Empirical result (matrix tracks, duration=25s)

| Track            | Baseline err | Trained err | Delta |
|------------------|-------------:|------------:|------:|
| race_01          | 0.089 m      | 0.086 m     | **−3.3%** |
| aigp_default     | 0.233 m      | 0.243 m     | +4.4% |
| grand_tour       | 0.079 m      | 0.081 m     | +1.9% |
| slalom           | 0.159 m      | 0.171 m     | +7.8% |
| straight_hairpin | 0.070 m      | 0.071 m     | +1.8% |
| vertical_cliff   | 0.055 m      | 0.056 m     | +1.9% |

**1/6 improved, 5/6 marginally worse. Not a net win.**

All tracks still PASS sim_passed; the residual doesn't break anything
(the iter-001 A15 safety clamps hold). Just doesn't help.

## Why

The FEL target heuristic (`target = -k * pos_err / g`) was chosen for
the simple linearisation-around-hover case and dominantly trained on
race_01-class data (longest run = most samples). It overfits to the
"wide gentle bends" of race_01 and underfits the tight transients of
slalom / aigp_default. With more sophisticated training (per-track
fitting, inverse-dynamics targets, weighted loss against high-curvature
segments) we'd likely improve — but that's research-scale work and
each tuning cycle requires re-running matrix.

## What this proves

The infrastructure works end-to-end:
- Dataset collection works (iter-025)
- Training works (iter-027)
- Inference works in both attitude AND acceleration paths (iter-027)
- Safety clamps hold (iter-001 A15 + the 0.05 rad / 0.05 thrust caps)
- A future training iter with better targets / per-track tuning can
  re-use all of the plumbing.

## Per the brick-wall rule

This is exactly the "lightweight ML provides negligible benefit"
case the user codified in `feedback_loop_termination_brick_wall.md`.
Continuing to grind on residual hyperparameters here is unlikely
to convert 1/6 → 6/6 — the fundamental issue is the target heuristic,
not the optimiser. A better target needs domain expertise (inverse
dynamics, gate-aware loss weighting) — research-scale work, not loop-
iteration work.

Status update:
- Charter item 5 (lightweight ML) — INFRASTRUCTURE COMPLETE, weights
  trainable, safety preserved. The "trained model beats baseline"
  goal not met by this iter's training run.
- Closing task #11 as "infrastructure shipped, training optimisation
  deferred".

## Code that ships in iter-027

- `scripts/train_tracker_residual.py` (NEW)
- `control/mpc_tracker.py` — residual projection into accel_des
- (this synthesis doc)
