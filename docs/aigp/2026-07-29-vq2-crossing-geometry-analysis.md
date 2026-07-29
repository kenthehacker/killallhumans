# VQ2 crossing-geometry analysis — why gate 1 is never passed

Date: 2026-07-29. Author: automated evidence analysis (read-only against
`C:\Users\John\aigp-evidence\fast-flight-cycles`).

Sample: all **111 credited visual-course runs from 2026-07-28/29** (gate-0→1
`authoritative_transitions` non-empty in `result.json`); **85 runs** had a
usable tick trace plus a `visual_gate_transition_promoted` event and form the
primary sample. Track centers are normalized image coords (x right, y down,
top edge = -1.0; vertical half-FOV = 30 deg, horizontal = 45 deg per
`flight_control/adapter.py`). "Credit" = the authoritative race-status
transition 0→1 (`visual_gate_transition_promoted`).

Representative evidence runs cited below: `20260729T040518Z-visual-course-797413fd`,
`20260729T034617Z-visual-course-c6407bd0`, `20260729T033632Z-visual-course-5d8d8587`.

## Q1 — Crossing geometry at gate-0 credit (n = 85)

Vehicle state at the credit tick:

| quantity | median | p10 | p90 |
|---|---|---|---|
| pitch (rad) | -0.069 | -0.136 | -0.016 |
| roll (rad) | -0.015 | -0.234 | +0.048 |
| body pitch rate (rad/s) | +0.121 | — | — |
| body roll rate (rad/s) | -0.081 | — | — |
| body yaw rate (rad/s) | **-0.330** | — | — |
| commanded thrust | 0.295 | (advance_thrust, above support ~0.278) | |
| world up specific force (m/s²) | +9.98 | +9.61 (p25) | +10.45 (p75) |

Net vertical accel at credit ≈ +0.17 m/s² (slight climb; only ~2% of runs
descending faster than -0.5 m/s² at credit). The vehicle crosses gate 0
mildly nose-down, climbing gently, but already yawing at ≈ -0.33 rad/s.

Gate-1 first observed position:

- Gate 1 (the promoted track) is **already visible in 85/85 runs before
  credit** (median last pre-credit y = -0.689, p90 = -0.482).
- After credit it is re-acquired on the **very next frame** (median 1 frame,
  ~20 ms; 84/85 runs).
- First post-credit appearance: **median y = -0.692** (p10 -0.759, p90 -0.494),
  median x = +0.381.
- **20.2% of runs: gate 1 is already TOP-clipped and center-censored on its
  first post-credit appearance** (18.8% already top-clipped pre-credit).

Verdict: the crossing itself is gentle, but gate 1 sits ~0.69 of the way to
the top edge (only ~0.31 norm ≈ 9 deg of margin) at the moment control
authority switches to it, and 1 in 5 crossings is already censored at credit.

## Q2 — Vertical sign, empirically

Pre-credit gate-0 approach segments (target cleanly visible, n = 5,566
consecutive-tick pairs across 85 runs):

- corr(commanded thrust, world-up acceleration) = **+0.913**. Binned net
  vertical accel vs thrust: -0.27 / +0.36 / +0.78 / +1.48 m/s² for thrust
  quartiles around support ≈ 0.278. **Increasing collective ascends; the
  thrust→vertical channel is conventional.**
- corr(thrust, d(image_y)/dt) = **+0.163**. After removing the pitch-rate
  component (fit d(image_y)/dt ≈ 0.93 * pitch_rate, i.e. pure camera
  rotation), the residual correlation is still **+0.130**, monotone across
  thrust quartiles (-0.163, -0.249, -0.241, -0.152 norm/s).

So: more collective → vehicle climbs → gate target moves **DOWN** in the
image (image_y, down-positive, increases). This is the standard pinhole
geometry; the measured plant contradicts the derived Rx(pi) camera flip used
to justify the gates-1+ `support + K*e` law. Gate 0's `BASE - K*e` is the
stabilizing sign; `support + K*e` is positive feedback in the vertical axis.

Post-credit gate-1 segments (n = 1,059 visible, uncensored pairs):
corr(thrust, d(image_y)/dt) = **-0.122** — opposite sign, but this subset is
badly confounded: gate 1 sits at the top edge where pitch rotation (the
recovery law slews pitch from ≈ -0.31 to ≈ 0 rad) and closure perspective
dominate the measured image motion. We do not treat the post-credit
correlation as plant evidence; the pre-credit approach (target near center,
small rates) is the clean identification condition.

## Q3 — Post-credit trajectory (n = 85)

- Credit→failure: median **1.64 s** (p25 1.18, p75 2.39); **66% die within
  2 s** of credit. (Matches the cohort-level "84% within 2 s" when the
  all-credit population, including earlier dates, is counted.)
- Failure reasons (all 111 runs): collision 25, visual-authority refused 18,
  visibility-gap horizon expired 16, visibility-gap guidance refused 11,
  post-credit recovery timeout 7, others <7 each. Directly or indirectly,
  ~all are loss-of-gate-1-observability failures.
- Post-credit thrust: median 0.294 → 0.278 (first→last tick), but runs that
  enter the top-censored recovery law drop to **0.21 (subsupport,
  `subsupport_collective_authorized: true`)** — a commanded descent.
- Gate-1 vertical error while still measurable (aperture events, n = 1,496):
  median **-0.594** (p10 -0.739, p90 -0.294) and drifting negative.
- In the censored-recovery regime (n = 30 runs with ≥5-sample
  `stable_center_norm` series): the target's estimated y drifts **away from
  center (upward) in 62% of runs**, median drift -0.21 norm/s (p25 -0.52);
  **median per-run monotonic-away fraction is 1.00** (53% of runs are ≥90%
  monotonic-away). Example (`...040518-797413fd`): y = -0.93 → -2.29 in
  ~0.75 s while thrust = 0.21 and yaw pinned at -0.150.
- Per-run corr(thrust, stable_y): positive in only 8/17 measurable runs —
  but thrust is nearly constant (0.21) inside recovery, so the descent→
  upward-drift coupling is visible as *level* (subsupport → drift away)
  rather than as within-run correlation.

Verdict: post-credit, collective is cut to subsupport, the vehicle descends,
the below-gate closure geometry raises gate 1's elevation angle, and the
target walks off the top edge monotonically in the median run. The vertical
error never recovers; failure follows ~1.2-1.6 s after credit.

## Q4 — Counterfactual: would a pre-crossing climb / vertical recentering help?

Rough estimate from the measured geometry (assumptions stated):

- At credit, gate-1 y median = -0.69 → elevation ≈ 0.69 × 30° ≈ 20.7° above
  the camera axis (~17° in world given pitch ≈ -4°). Margin to the top edge:
  0.31 norm ≈ 9.3°.
- Recovery-event telemetry gives `time_to_contact` ≈ 2.4–2.8 s at credit and
  a nonrotational upward center rate of ≈ -0.9 norm/s initially (measured
  `residual_rate_norm_s`), decaying to a median -0.21 norm/s over the window.
- With ~0.3 norm of margin and a -0.5…-0.9 norm/s initial drift, gate 1
  leaves the censored-safe zone in **0.3–0.6 s**, versus the ~2.5 s needed to
  reach it. Even at the median drift (-0.21) the window is only ~1.5 s.
- Geometric scaling: both the initial elevation angle and its closure-driven
  growth rate are proportional to the below-gate height offset Δh
  (θ ≈ atan(Δh/d)). Halving Δh — crossing ≈ 1–1.2 m higher, or equivalently
  recentering so gate 1 first appears at y ≈ -0.35 — roughly **doubles** the
  observability window (to ~1–3 s) and halves the upward drift the vertical
  loop must fight. Getting gate 1 to y ≈ -0.2 at credit would approximately
  cover the closure time if the drift halves with it.

Verdict: **yes, warranted.** The margin math is one-sided: as flown, the
crossing hands gate 1 to the controller already ~9° from the top edge with
~2.5 s of closure to fly. A pre-crossing climb (or crossing with gate 0
lower in frame / the camera less nose-down) is the single largest available
observability improvement, but it must be paired with a correct-sign vertical
loop after credit, or the subsupport descent re-opens the same divergence.

## Q5 — Yaw/roll lateral response (n = 353 post-credit recovery pairs)

- **All** post-credit recovery pairs have gate 1 at x > +0.1 (median
  x = +0.568); the course geometry always hands off gate 1 to the right.
- With the target right of center, the commanded yaw is **-0.142 rad/s mean
  (pinned near the -0.150 cap)** and the target's image-x moves **away from
  center (d x/dt > 0) in 75% of pairs**, mean +0.91 norm/s. Example
  (`...040518-797413fd`): x = 0.35 → 0.58 over ~0.5 s at yaw = -0.150.
- corr(yaw_cmd, d x/dt) = -0.14; corr(roll_cmd, d x/dt) = +0.18 — both weak
  because yaw is saturated constant in this regime.

Sign reading: with a down-positive/right-positive image and a standard
yaw convention, a **negative** yaw rate rotates the camera left, shifting
image features **right** — i.e. the commanded yaw pushes a right-side target
further from center. Recentering x > 0 requires a positive yaw. As measured,
the lateral command has the destabilizing sign (or is a saturated artifact of
one). Confound: pure forward closure makes any off-axis feature drift
radially outward (expansion), which also produces d x/dt > 0; the recovery
telemetry's own nonrotational rate confirms expansion contributes. But the
*rotational* contribution of yaw = -0.15 is independently away-from-center,
so the closed loop cannot have converged even without expansion.

## Implications for the clean controller

1. **Vertical feedback sign: use `BASE - K*e` (the gate-0 sign) for all
   gates.** The measured plant is unambiguous in the clean identification
   condition (pre-credit approach): more collective → climb (corr 0.91 with
   up-acceleration) → target moves down in image (residual corr +0.13 after
   derotation). The Rx(pi) flip behind `support + K*e` is not supported by
   build-3385 data; that law is positive feedback and is a primary suspect
   for the universal gate-1 failure. Also drop the subsupport (0.21)
   recovery collective — a commanded descent while the target is escaping
   upward is backwards; hold support or climb when vertical error is
   censored-negative.
2. **Pre-crossing climb / vertical recentering: warranted.** Cross gate 0
   higher (target: gate-1 first-observed y ≈ -0.3 or lower, vs measured
   -0.69) to double the post-credit observability window. Cheapest
   mechanism: raise collective to ~support+Δ through the final gate-0
   approach and/or reduce the advance nose-down pitch so the camera axis is
   less depressed at credit.
3. **Lateral sign: correct it.** With gate 1 right of center the loop
   commands yaw = -0.15 and x diverges; recentering requires positive yaw
   for positive x-error. Verify the sign convention of the yaw error term
   against this measurement before the next flight; treat the expansion
   confound as secondary.
4. Caveats: image-y motion mixes rotation, closure perspective, and
   translation; the post-credit vertical correlation is not plant evidence
   (censored-axis, saturated commands). Command-to-plant lag (one tick,
   ~20 ms) is small versus the observed drifts and does not change any sign
   conclusion. All numbers above are from the 2026-07-28/29 credited cohort
   (n = 85) unless stated.
