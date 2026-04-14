# Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag
- **URL**: https://arxiv.org/abs/1712.02402
- **Authors**: Matthias Faessler, Antonio Franchi, Davide Scaramuzza
- **Year**: 2018
- **Venue**: IEEE Robotics & Automation Letters, Vol. 3(2), pp. 620–626

---

## Key Contribution

This paper makes a single, tightly-scoped but high-impact contribution: it proves that the dynamical model of a quadrotor subject to **linear rotor drag effects** is **differentially flat** with flat outputs chosen as position `p` and heading `ψ`. This extends the earlier Mellinger & Kumar (2011) result, which established flatness only for the drag-free model, by showing that flatness is preserved even when the body-frame velocity-dependent drag force `−RDRᵀv` (where `D = diag(dx, dy, dz)` is a diagonal drag matrix) is included.

The payoff of this proof is fully practical: every state variable and every control input of the quadrotor — thrust `c`, orientation `R`, body rates `ω`, and angular accelerations `ω̇` — can be written as **exact algebraic functions of the reference trajectory and its time-derivatives up to snap** (4th derivative of position). These algebraic expressions are used as feedforward terms in a cascaded nonlinear controller, replacing the prior approach of treating drag as an unknown disturbance rejected by feedback alone.

The secondary contribution is a **gradient-free (Nelder-Mead) drag coefficient identification** procedure that identifies D purely from flight data, requiring no IMU differentiation or rotor speed measurement.

---

## Technical Approach

### Dynamics Model

The paper adopts the lumped-parameter drag model from Kai et al. (2017):

```
ṗ = v
v̇ = −g·zW + c·zB − R·D·Rᵀ·v
Ṙ = R·ω̂
ω̇ = J⁻¹(τ − ω×Jω − τg − A·Rᵀ·v − B·ω)
```

The key term is `−R·D·Rᵀ·v`: the drag force is proportional to world-frame velocity, mapped into the body frame by `Rᵀ`, scaled per axis by D, then mapped back to world frame. This is a linear velocity effect and represents the dominant aerodynamic disturbance at racing speeds. An additional thrust model `c = ccmd + kh·(v·(xB+yB))²` accounts for horizontal velocity effects on collective thrust.

### Flatness Proof — Orientation and Thrust

The key insight is to re-arrange the translational dynamics (eq. 2) and project onto body axes. Define:

```
α = a + g·zW + dx·v
β = a + g·zW + dy·v
```

Orthogonality constraints on `xB` and `yB` with respect to `α` and `β`, combined with the heading constraint (xB projected into xW-yW plane is collinear with xC), yield the body frame axes:

```
xB = (yC × α) / ‖yC × α‖
yB = (β × xB) / ‖β × xB‖
zB = xB × yB
```

The collective thrust is recovered by projecting equation (2) onto `zB`:
```
c = zBᵀ(a + g·zW + dz·v)
```

All of these are computable from reference position, velocity, acceleration, and the identified drag coefficients — no feedback error enters.

### Feedforward Body Rates

Taking the time derivative of the translational dynamics (eq. 16) and projecting onto `xB` and `yB` gives two linear equations in `(ωx, ωy, ωz)`. A third equation comes from differentiating the heading constraint. The body rates are the solution of a 3×3 linear system:

```
ωy(c − (dz−dx)(zBᵀv)) − ωz(dx−dy)(yBᵀv) = xBᵀj + dx·xBᵀa
ωx(c + (dy−dz)(zBᵀv)) + ωz(dx−dy)(xBᵀv) = −yBᵀj − dy·yBᵀa
ωz = (1/‖yC×zB‖)(ψ̇·xCᵀxB + ωy·yCᵀzB)
```

These require trajectory position, velocity, acceleration, and **jerk** (3rd derivative).

### Feedforward Angular Accelerations

Differentiating the body-rate equations once more yields a similar 3×3 linear system for `ω̇`, requiring trajectory **snap** (4th derivative). The term `ξ` in equations (26-28) depends on `R, ω, D, v, a, j` and is fully computable from known quantities.

With `ω̇ref` known, the torque input `τ` is recovered from equation (4).

### Control Law Structure

The full controller is cascaded:

1. **Position controller** computes desired acceleration:
   ```
   ades = afb + aref − ard + g·zW
   ```
   where `afb = −Kpos(p−pref) − Kvel(v−vref)` is PD feedback and `ard = −Rref·D·Rᵀref·vref` is the drag-compensation feedforward from the reference.

2. From `ades` and heading, compute desired orientation `Rdes` via eqs. (33-35).

3. Compute thrust: `ccmd = adesᵀzB − kh(v·(xB+yB))²`

4. **Body-rate controller** adds feedback on attitude error to the reference body rates:
   ```
   ωdes = ωfb + ωref
   ω̇des = ω̇ref
   ```

The cascade has the property that, in the absence of any state error, the feedforward terms alone reproduce the exact reference trajectory including drag compensation.

### Drag Coefficient Identification

Nelder-Mead optimization minimizes the RMSE tracking error `Ea` over repeated flights of a known trajectory. The optimization searches over `(dx, dy, dz, kh)`. It typically converges in ~70 iterations (~30 min with battery swaps). No IMU differentiation or rotor speed measurement is needed. Key finding: `dz` and `kh` have only minor effects on tracking; `dx` and `dy` dominate.

---

## Results

All experiments used a 610 g FPV racing quadrotor with 6-inch propellers, a thrust-to-weight ratio of 4, and OptiTrack ground truth at 200 Hz. The high-level controller ran at 55 Hz on an offboard laptop.

### Drag Coefficients Identified

| Trajectory | dx (s⁻¹) | dy (s⁻¹) |
|------------|-----------|-----------|
| Circle | 0.544 | 0.386 |
| Lemniscate | 0.491 | 0.236 |

dx > dy in both cases, consistent with the vehicle being wider than long.

### Tracking Error Comparison (10 loops, 4 m/s)

**Circle trajectory (radius 1.8 m, 4 m/s):**

| Method | max ‖Ep‖ (cm) | σ ‖Ep‖ (cm) | Ea (cm) |
|--------|--------------|-------------|---------|
| No drag compensation | 21.08 | 2.11 | 17.53 |
| Drag ID on circle | 14.54 | 2.63 | 6.54 |
| Drag ID on lemniscate | 12.39 | 2.53 | 8.16 |

**Lemniscate trajectory (max 4 m/s):**

| Method | max ‖Ep‖ (cm) | σ ‖Ep‖ (cm) | Ea (cm) |
|--------|--------------|-------------|---------|
| No drag compensation | 16.79 | 3.19 | 11.27 |
| Drag ID on circle | 10.25 | 2.30 | 5.56 |
| Drag ID on lemniscate | 10.02 | 2.23 | 5.51 |

**Summary**: The proposed method reduces RMSE tracking error by approximately **50%** across both trajectories. The absolute tracking error (Ea) drops from ~17.5 cm to ~6.5 cm on the circle and from ~11.3 cm to ~5.5 cm on the lemniscate. Drag compensation begins to help at speeds above ~0.5 m/s and the benefit grows monotonically with speed up to the tested 5 m/s.

Remaining error after compensation correlates with collective thrust variation, which violates the model's assumption that drag is thrust-independent.

---

## Relevance to Our System

Our system (`mpc_tracker.py`, `trajectory_optimizer.py`) currently uses a geometric SE(3) tracker (Lee et al.) that treats drag as a disturbance rejected by feedback. At racing speeds (3-8 m/s, with segments up to 10+ m/s after the iteration-5 speed optimization), this is exactly the regime where drag-induced tracking error becomes significant.

Specific connections:

1. **Per-gate tracking error**: The paper shows ~50% error reduction from feedforward drag compensation at 4 m/s. Our current average tracking error is ~0.285 m (28.5 cm) and the target is <0.25 m. Even a 20-30% improvement from partial feedforward implementation would close the gap.

2. **Cascade structure matches our architecture**: Our `mpc_tracker.py` already uses a cascaded position → attitude → body rate structure. The Faessler feedforward terms slot directly into the position controller's desired acceleration computation and the body-rate reference.

3. **Reference acceleration feedforward exists**: The current tracker already adds `aref` (reference acceleration from the min-snap trajectory). What is missing is the drag compensation term `ard = −Rref·D·Rᵀref·vref`.

4. **Sharp turns are the worst case**: Gates 3 and 7 (the two gates with highest per-gate errors in our diagnostics) involve aggressive direction changes. At high speed with tight curvature, `v` is large and the drag force `RDRᵀv` creates a body-frame braking force that the current controller under-compensates as speed increases.

5. **Min-snap provides all required derivatives**: Our `trajectory_optimizer.py` already generates snap-continuous polynomial trajectories. Position, velocity, acceleration, jerk, and snap are all available analytically. This means the full feedforward chain (orientation → thrust → body rates → angular accelerations) can be computed without numerical differentiation.

---

## Actionable Takeaways

1. **Add drag compensation to position controller** (highest priority, lowest effort): In `mpc_tracker.py`, compute `ard = −Rref·D·Rᵀref·vref` and subtract it from the desired acceleration before computing desired orientation and thrust. This is the single most impactful change — it corresponds to the dominant ~50% error reduction in the paper. Requires only dx, dy as tunable parameters.

2. **Add body-rate feedforward**: Currently `ωdes` likely only has `ωfb`. Add `ωref` computed from the linear system (eqs. 17-18, 25), which requires jerk from the trajectory. The `trajectory_optimizer.py` already returns polynomial coefficients so jerk evaluation is free.

3. **Add angular acceleration feedforward**: Add `ω̇ref` (from snap, eq. 26-28) as feedforward to the body-rate controller. This is the most involved change but completes the full flatness-based feedforward chain.

4. **Identify drag coefficients**: Fly two loops of our existing circle/gate trajectory, minimize RMSE tracking error over `(dx, dy)` using scipy.optimize.minimize with Nelder-Mead. Expected values: dx ≈ 0.4-0.55 s⁻¹, dy ≈ 0.24-0.39 s⁻¹ based on the paper's FPV platform (similar size class to competition drones). Start with dx = dy = 0.35 s⁻¹ as a first approximation.

5. **Use cross-validated coefficients**: The paper shows that coefficients identified on a high-excitation trajectory (circle) generalize well to other trajectories. Identify on the highest-speed circuit segment and apply globally.

6. **Set dz = 0 initially**: The paper found `dz` has negligible effect. Only optimize `dx` and `dy` first.

---

## Limitations & Caveats

1. **Drag assumed thrust-independent**: The model assumes D is constant, independent of thrust level. In reality drag scales with rotor speed (≈ √thrust). This is cited as the cause of remaining errors when thrust varies significantly (e.g., 10-18 m/s² on the circle). At racing speeds with aggressive thrust variations (gate pull-ups, steep dives), this model will be imperfect.

2. **Low control rate (55 Hz)**: The paper's experiments used 55 Hz for the high-level position controller, which is below our target 100+ Hz. The feedforward approach compensates for some of the bandwidth limitations but not all.

3. **Drag coefficients are trajectory-dependent**: Different trajectories at different speeds yield different identified coefficients, because the model's thrust-independence assumption is violated to different degrees. This means identified drag coefficients from one trajectory segment may not fully generalize to a qualitatively different segment.

4. **Identified on FPV racer, not our platform**: The paper's drone (610 g, 6-inch props) is a specific size class. Our platform's drag coefficients could differ by ±50%. The Nelder-Mead identification procedure must be re-run for our specific vehicle.

5. **No vertical drag in practice**: The paper found `dz = 0` in all experiments, even for vertical trajectories at 2.5 m/s. This simplifies implementation but means the model does not capture aerodynamic drag in the vertical axis.

6. **Motion capture ground truth**: All experiments used OptiTrack at 200 Hz. Our system uses an EKF with visual odometry. EKF state noise means `vref` vs. actual `v` have estimation error, which enters the drag compensation term. If EKF velocity uncertainty is high, the drag feedforward can inject noise rather than correct error.

7. **No obstacle avoidance or gate-position uncertainty**: The paper tracks pre-computed smooth trajectories on an open platform. Our system must handle gate position uncertainty, which means the reference trajectory itself may be imperfect. Feedforward based on the nominal reference will be "wrong" in a consistent direction — however, it will still be better than no feedforward.

---

## Key Parameters / Constants

| Parameter | Symbol | Value (paper) | Notes |
|-----------|--------|---------------|-------|
| Drag coeff. (x-axis) | dx | 0.491–0.544 s⁻¹ | Higher due to wider frame |
| Drag coeff. (y-axis) | dy | 0.236–0.386 s⁻¹ | Lower due to narrower fore-aft |
| Drag coeff. (z-axis) | dz | ≈ 0 | Negligible in practice |
| Horizontal thrust factor | kh | 0.009 m⁻¹ | Order of magnitude less impactful |
| Vehicle mass | m | 610 g | FPV racing quadrotor |
| Thrust-to-weight ratio | TWR | 4 | Typical FPV racing spec |
| Control loop rate | — | 55 Hz (high-level) | 4 kHz body-rate inner loop |
| Latency compensated | — | 32 ms | Full pipeline latency |
| Identification convergence | — | ~70 iterations | ~30 min with battery swaps |
| Speed where drag matters | — | > 0.5 m/s | Below this, no improvement |
| Tracking error reduction | — | ~50% RMSE | At 4 m/s on tested trajectories |
| Circle radius (test) | — | 1.8 m | At 4 m/s |
| Max tested speed | — | 5 m/s | Speed ramp experiments |
| Required trajectory derivative | — | up to snap (4th) | For full ω̇ feedforward |
| Required trajectory derivative | — | up to jerk (3rd) | For ω feedforward only |
| Required trajectory derivative | — | up to acceleration (2nd) | For orientation+thrust only |

Sources:
- [Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag (arXiv:1712.02402)](https://arxiv.org/abs/1712.02402)
- [Full PDF (RPG Lab UZH)](https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf)
- [IEEE RA-L DOI: 10.1109/LRA.2017.2776353](https://doi.org/10.1109/LRA.2017.2776353)
- [Code: uzh-rpg/rpg_quadrotor_control](https://github.com/uzh-rpg/rpg_quadrotor_control)
