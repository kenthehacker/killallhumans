# Iteration 43 Research Synthesis: ILC Acceleration Feedforward Correction

## Current State
- ILC computes position offsets (Δp) and velocity offsets (dΔp/dt) with per-section scaling
- Avg tracking error: 0.138m, max: 0.727m, race time: 14.08s
- The feedforward acceleration signal is still from the ORIGINAL trajectory, creating an inconsistency

## Research Basis

### 1. Schoellig et al. 2012 — Full feedforward correction
The seminal ILC paper explicitly states that for full reference consistency, ALL feedforward inputs should be corrected. When position is shifted by Δp(t), the velocity reference should be dΔp/dt, and the acceleration reference should be d²Δp/dt². Quote: "The feedforward input for the next trial is updated via convex optimization." The paper corrects position, velocity, AND acceleration feedforward in its full formulation.

**Key insight**: Our current implementation only corrects position and velocity. The acceleration feedforward still comes from the original trajectory's polynomial. This means the controller is told "be at position p+Δp, move at velocity v+dΔp/dt, but accelerate as if Δp didn't exist." This is inconsistent.

### 2. Tal & Karaman 2018 — Differential flatness feedforward
The INDI paper demonstrates that higher-order derivative feedforward (jerk → angular rate, snap → angular acceleration) significantly reduces tracking error. The key finding: "jerk feedforward reduces response overshoot" by anticipating attitude changes before positional error accumulates.

**Key insight for acceleration correction**: The acceleration feedforward tells the controller what force is needed to follow the trajectory's curvature. If the path is shifted by ILC offsets, the curvature changes, and the required acceleration changes. Not correcting acceleration is like telling the controller "the road curves here" when actually the road curves slightly differently.

### 3. Signal processing consideration — Second derivative noise
The position offsets are already Butterworth-filtered (0.35 Hz, 4th order). First derivative (velocity) is clean because:
- The position signal is very smooth (Butterworth filtered)
- np.gradient central differences are well-conditioned for smooth signals

The second derivative (acceleration) amplifies noise quadratically. However:
- Our position offsets are deterministic (no stochastic noise)
- The Butterworth filter already removed high-frequency content
- We can apply an additional Butterworth filter to the acceleration offset

### 4. Per-section scaling strategy
Following Bristow & Alleyne 2007 (time-varying Q-filter), the acceleration correction should use per-section scaling similar to velocity:
- **Pre-inflection (0-200 steps)**: 0.0 — same as velocity, protect gate-2
- **Inflection (200-440 steps)**: 0.2-0.3 — very conservative, S-turn is sensitive
- **Post-inflection (440-740 steps)**: 0.3-0.4 — moderate
- **Helix (740+)**: 0.5-0.7 — aggressive, where offsets are largest and benefit highest

## Consensus
All research agrees: full feedforward consistency (pos + vel + accel) is strictly better than partial consistency. The risk is noise amplification in the second derivative, which is managed by Butterworth filtering and conservative per-section scaling.

## Proposed Implementation
1. After converging ILC position offsets, compute acceleration offset = d²Δp/dt²
2. Apply Butterworth low-pass filter (0.5-1.0 Hz) to the acceleration offset to suppress differentiation noise
3. Use per-section acceleration scaling (7th element of section_boundaries)
4. Apply in both ILC inner sim and benchmark execution
5. Start with conservative scaling (0.0 / 0.2 / 0.3 / 0.5) and sweep if successful

## Expected Impact
- 2-5% avg error reduction (the acceleration correction fills the remaining reference inconsistency gap)
- Primary benefit in helix section where ILC offsets are largest
- Risk: noise in second derivative could cause regression if filtering is insufficient
