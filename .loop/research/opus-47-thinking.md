# Multi-Track Generalization Research — Opus 4.7 (max-thinking)

## Summary (3-5 sentences)

The stall is *geometric*, not perceptual or learning-based: the polynomial
trajectory's first segment overshoots gate-1's normal and the kinematic
drone follows it across gate-2's opening before the sequencer credits
gate-1, tripping a correct in-order DQ. The fix that maps most directly
to a Python-stack edit is **encoding gate-aware safety constraints
into the existing polynomial trajectory optimizer** — a *safety tunnel*
between consecutive gates (MPCC++ / Safe Flight Corridors style) plus
*gate-polygon traversal* constraints (TOGT style). Both are deterministic
QP-friendly additions to what we already have. A second, principled but
heavier rewrite is to swap the *time-allocation* layer for **Complementary
Progress Constraints (CPC)**, which provably forbid progress on gate
*N+1* until the trajectory is within proximity of gate *N* — exactly the
property the in-order DQ is enforcing at runtime. We should defer
RL-based generalization (Swift, On-Your-Own, DreamerV3) for iter-005+:
no GPU budget, and the stall is solvable in optimization-land.

## Top 3 candidate techniques (ranked)

### C1. Safety-tunnel trajectory constraint (MPCC++ prismatic tunnel + SFC polyhedra)

- **Citation**: Krinner, Romero, Bauersfeld, Zeilinger, Carron, Scaramuzza.
  "MPCC++: Model Predictive Contouring Control for Time-Optimal Flight
  with Safety Constraints." *Robotics: Science and Systems (RSS) 2024*
  (paper 109). Also: Liu, Watterson, Mohta, Sun, Bhattacharya, Taylor,
  Kumar. "Planning Dynamically Feasible Trajectories for Quadrotors
  Using Safe Flight Corridors in 3-D Complex Environments." *IEEE
  Robotics and Automation Letters* 2(3):1688–1695, 2017.
- **What it does**: Defines a *prismatic Frenet-frame tunnel* (W(θ), H(θ))
  around the centerline that joins gate inner corners; the trajectory
  is constrained to stay inside that tunnel via linear inequalities,
  while time-optimality lives in the cost. MPCC++ proves recursive
  feasibility via a terminal set and demonstrates **100% gate-crash-free
  success at >80 km/h** in real flight. SFC builds the same family of
  constraints as convex polyhedra for polynomial QPs.
- **Why it unblocks our stall**: The failure mode is *cross-track
  overshoot pushing the path through future gate openings*. A tunnel
  that connects gate-1's inner corners to gate-2's inner corners
  *geometrically excludes* gate-3's plane from gate-1→2's reachable
  set; the optimizer cannot emit a trajectory that crosses gate-2 before
  gate-1 because gate-2's opening is not in the gate-0→1 tunnel. The
  same constraint prevents the slalom strut crash (gate-8 frame hit at
  4.8 s) and the aigp_default crash (gate-1 strut at 1.2 s).
- **Cost (S/M/L)**: **M**. We already have polynomial trajectories
  (`planning/trajectory_optimizer.py`) and a Catmull-Rom centerline
  (`planning/racing_line.py`). Add: (a) per-segment Frenet tunnel
  parameterization keyed off the centerline, (b) linear inequality
  constraints on the polynomial sample points (or on the control
  polygon for Bernstein-form bounds), (c) corner-tightening near gates.
  Solver stays QP/SOCP — no new dependencies.
- **Expected gain (small/medium/large)**: **Large**. Directly attacks
  the geometric failure mode; expected to flip all 5 out-of-order DQ
  tracks into "passes gate-1, passes gate-2" within iter-004.
- **Risks/constraints**: Tunnel may infeasibilise tight slalom if
  width is too small; need *gate-section* widening rule (taper to gate
  half-width at θ_k, expand to free-space corridor in mid-segment).
  Compute: one QP per replan; well within the 100 Hz loop budget if we
  precompute offline. No data dependency.

### C2. Time-Optimal Gate-Traversing constraints (TOGT, polygon-shaped gates)

- **Citation**: Qin, Michet, Chen, Liu. "Time-Optimal Gate-Traversing
  Planner for Autonomous Drone Racing." *2024 IEEE International
  Conference on Robotics and Automation (ICRA)*, Yokohama, Japan,
  pp. 8693–8699, 2024.
- **What it does**: Models each gate as a *2-D convex polygon at a
  specified pose* (not a point waypoint) and enforces the trajectory
  pass through the gate's polygon *and* in the direction of the gate
  normal, while keeping full single-rotor-thrust constraints. Solved as
  a polynomial-based optimization in seconds for tracks with dozens of
  gates. Open implementation at `FSC-Lab/TOGT-Planner` (C++; the
  Python wrapper `Run-TOGT-Planner` exists).
- **Why it unblocks our stall**: Two of our 7 tracks DQ via *crash on a
  gate strut* (slalom, aigp_default). Waypoint-passing planners aim
  at the gate *center* and ignore the gate's finite opening, which on
  a tight initial-acceleration segment can clip a frame. A polygon
  traversal constraint (a) forces the path through the gate's interior
  (no strut clipping) and (b) gives the optimizer freedom to use the
  whole opening — naturally smoothing the gate-1 → gate-2 cross-track
  curvature so the first segment doesn't pre-cross gate-2.
- **Cost (S/M/L)**: **M** (port the polygon-traversal constraint into
  our QP; we don't need the C++ TOGT solver — we just need the
  *constraint formulation*: position at θ_k inside gate polygon, velocity
  at θ_k parallel to gate normal within a cone). **L** if we adopt the
  full FSC-Lab solver as a Python subprocess.
- **Expected gain (small/medium/large)**: **Medium-to-Large**. Fixes the
  two crashes outright (gate-clipping) and tightens the per-segment
  overshoot via the heading-cone constraint. Expected to combine
  multiplicatively with C1.
- **Risks/constraints**: AIGP gate geometry is a round opening; treat
  it as an inscribed polygon (8-sided is plenty) or use the disk
  half-plane formulation. No data dependency. Real-time: offline
  precompute fits the existing model.

### C3. Complementary Progress Constraints (CPC) on time allocation

- **Citation**: Foehn, Romero, Scaramuzza. "Time-optimal planning for
  quadrotor waypoint flight." *Science Robotics* 6(56):eabh1221, 2021.
  (Preprint: Foehn & Scaramuzza, "CPC: Complementary Progress Constraint
  for Time-Optimal Quadrotor Trajectories," arXiv:2007.06255, 2020.)
- **What it does**: Augments each waypoint with a *progress variable*
  λ_k(t) ∈ [0,1] whose derivative is constrained by a *complementarity
  relation* with proximity to waypoint k: λ_k can only increase when
  the trajectory is within a tolerance ball of waypoint k. The QP/NLP
  then jointly optimizes the trajectory *and* the unknown waypoint
  time allocation, guaranteeing waypoints are completed *in order*
  before later progress variables can advance.
- **Why it unblocks our stall**: This is *the same invariant the AIGP
  sequencer enforces* (in-order pass-through), encoded *inside the
  planner*. A CPC-constrained optimizer is structurally incapable of
  producing a trajectory whose first segment "crosses" gate-2 before
  gate-1 has been completed — the progress complementarity blocks it
  at solve-time, not run-time. Eliminates a whole class of overfit
  failures.
- **Cost (S/M/L)**: **L**. Requires rewriting the time-allocation /
  segment-timing layer of `planning/trajectory_optimizer.py` from
  fixed per-segment durations to a CPC formulation. Needs an NLP
  solver (CasADi + IPOPT) — we'd add `casadi` as a dependency. The
  reference Science Robotics CPC implementation is documented but
  not Python-first.
- **Expected gain (small/medium/large)**: **Large** for correctness
  (provable ordering), **Medium** for race-time improvement over
  polynomial baseline once C1/C2 are landed. Best long-term primitive.
- **Risks/constraints**: NLP solve time (seconds, not milliseconds) —
  must be offline-precompute. CPC NLPs are non-convex with multiple
  local minima; needs warm-starting from the polynomial trajectory.
  Real-time: not real-time, but the existing stack precomputes
  trajectories offline anyway.

## Other candidates (don't pick, but flag)

- **MPCC base controller (Romero, Sun, Foehn, Scaramuzza,
  "Model Predictive Contouring Control for Time-Optimal Quadrotor
  Flight," IEEE T-RO 38(6):3340–3356, 2022)** — the contouring-control
  predecessor to MPCC++. If we ever move from a PD geometric tracker
  to receding-horizon control, this is the right entry point. Defer:
  MPC adds a real-time NLP at 100 Hz, which is a much bigger
  engineering lift than C1's offline tunnel encoding.

- **FAST-Racing (Han, Wang, Xu, Cao, Gao. "FAST-Racing: An
  Open-Source Strong Baseline for SE(3) Planning in Autonomous
  Drone Racing." IEEE RA-L 6(4):8631–8638, 2021)** — SE(3) planning
  with corridor + minimum-jerk. Useful reference for an open-source
  baseline if we want to validate our C1 implementation against a
  known good. Skip as primary.

- **Iterative Learning Control survey (Bristow, Tharayil, Alleyne. "A
  survey of iterative learning control." IEEE Control Systems
  Magazine 26(3):96–114, 2006)** — Q-filter / L-filter design for
  ILC robustness to model/track variation. Relevant because our
  per-section ILC defaults (α=0.4, max_corr=0.15) are the *secondary*
  failure mechanism per the regression matrix. A small fix:
  per-track ILC reset (don't reuse race_01's section schedule) and
  a low-pass Q-filter on cross-track corrections so they don't push
  the path off-axis on tight geometries. Cheap (S), but won't fix
  the geometric overshoot — pair with C1.

- **Domain randomization for racing (Ferede, Blaha, Lucassen,
  De Wagter, de Croon. "One Net to Rule Them All: Domain
  Randomization in Quadcopter Racing Across Different Platforms."
  *ICRA 2025*)** — trains *one* state-only NN controller across
  randomized dynamics. Demonstrated sim-to-real generalization across
  3-inch and 5-inch quadcopters. We don't have RL training infra and
  the network outputs motor commands (we run at body-rate level via
  Lee tracker), but the *principle* — randomize tracks during the
  ML residual's training set, not just race_01 — is directly portable
  to `control/learned_residual.py`. **S effort, S–M gain.**

- **Adaptive environment-shaping for RL drone racing (Wang et al.,
  "Adaptive Environment-Shaping for Generalizable Reinforcement
  Learning in Drone Racing," ICRA 2025; rpg.ifi.uzh.ch preprint).**
  Curriculum + environment shaping for cross-track RL generalization.
  Same RL-budget caveat; cite as "the right paper if we ever get a
  GPU."

- **On-Your-Own pro-level racing (Bosello, Romero, Scaramuzza, et al.
  "On Your Own: Pro-level Autonomous Drone Racing in Uninstrumented
  Arenas." IEEE RA-L 11(3):2674–2681, March 2026 — arXiv:2510.13644).**
  Vision + state-estimation + MPC stack without ground-truth
  fine-tuning — the architectural reference for our perception →
  EKF → tracker pipeline. Not a direct fix for the geometric stall.

## What NOT to do

- **End-to-end RL (Kaufmann et al., "Champion-level drone racing using
  deep reinforcement learning," *Nature* 620:982–987, 2023 — Swift;
  Loquercio et al., "Learning high-speed flight in the wild,"
  *Science Robotics* 6(59):eabg5810, 2021; Romero et al. "Dream to
  Fly," arXiv:2501.14377, 2025 — DreamerV3 for racing).** Requires
  GPU training infrastructure, a competitive sim env (gym-pybullet
  is fine but not racing-tuned), and tens of thousands of episodes
  per track. The stall is geometric — the trajectory crosses future
  gate planes — and is *fully* addressable in optimization. RL is
  the right hammer for sim-to-real *physics* gaps, not for in-sim
  in-order failures we can directly constrain. Re-evaluate when we
  have aerodynamic residuals or perception noise to learn against.

- **Tightening only race_01-style hyperparameters (ILC α, max_corr,
  PD gains).** The regression matrix shows max-tracking-error 0.66 m
  on race_01 vs 1.44 m on slalom — the bench is honest that ILC
  defaults *worsen* performance on tighter tracks. Tuning against
  race_01 in isolation will pump these numbers further into the red
  on the other 6 tracks. The optimization-stall protocol in
  `.loop/specs/0_charter.md` explicitly forbids this.

- **Softening the in-order DQ.** The strict in-order DQ is *AIGP
  competition rules* per the brief — it's correct. The fix lives
  in the planner that emits trajectories the DQ would flag, not in
  the DQ logic.

- **Adopting Romero T-RO 2022 MPCC as the primary controller in
  iter-004.** MPCC requires a real-time NLP solver in the 100 Hz
  inner loop. That's a multi-week swap, not an iter-004 hop. Defer
  to iter-005 *after* C1's offline-encoded tunnel proves the
  generalization hypothesis.

## My #1 pick if I had to ship one in iter-004

**Ship C1 (Safety-tunnel trajectory constraint, MPCC++ / SFC style)
inside the existing `planning/trajectory_optimizer.py` polynomial
QP. Pair with a per-track ILC reset (don't inherit race_01's section
schedule).** This is the surgical fix that maps to the smallest patch
in the current Python stack.

Sketch implementation (≤ 1 iter):

1. **Centerline + Frenet frame**: along the existing Catmull-Rom
   `planning/racing_line.py` curve, compute (t̂, n̂, b̂) at each
   sample. We already have this implicitly via the curvature-aware
   speed profile.

2. **Tunnel parameterization W(θ), H(θ)**:
   - Near a gate (|θ − θ_k| < ε_gate, e.g., 0.5 s of flight): clamp
     W, H to the gate inner half-dimensions minus a safety margin
     (e.g., gate radius − 0.2 m for AIGP's 1.5 m gates).
   - Mid-segment: expand W, H to the *minimum lateral free space* of
     the two adjacent gate-plane projections (rough heuristic: half
     the inter-gate spacing perpendicular to the centerline).
   - Smooth between the two regimes with a Hermite spline of θ.

3. **Constraint encoding in the polynomial QP**:
   - For each polynomial sample point p(t_i), express the deviation
     d_i = R_frenet(θ_i)·(p(t_i) − c(θ_i)) where c is the centerline.
     Add linear inequalities |d_i,n̂| ≤ W(θ_i)/2 and
     |d_i,b̂| ≤ H(θ_i)/2.
   - Bernstein-coefficient bounds are tighter (no sampling artifacts)
     if we already use Bernstein form; otherwise sample at
     ≥ 20 nodes per segment.

4. **Per-track ILC reset**: in `planning/ilc_sections.py`, gate the
   "carry-over from race_01 hyperparameters" behind a `track_id`
   check; start `alpha=0, max_corr=0` for any track other than
   race_01, accumulate corrections from the matrix benchmark itself.

5. **Verify in `scripts/benchmark_matrix.py`**: the success criterion
   is *not* "race_01 still passes." It is **"≥ 4 of the 7 tracks
   produce a gate-1 credit before any DQ"** — i.e., the geometric
   overshoot failure mode is gone. Reasonable iter-004 stretch:
   ≥ 3 tracks complete (≥ N − 1 gates).

6. **Adversarial bench gate**: add a unit test in
   `tests/planning/test_corridor_constraint.py` that synthesizes a
   tight 3-gate track (90° turn between gate-1 and gate-2, spacing
   ≤ 4 m) and asserts the optimizer's output trajectory *never*
   crosses gate-2's plane before gate-1's plane.

If C1 lands and the regression matrix flips 4+ tracks to gate-1
credited, queue C2 (gate-polygon traversal) for iter-005 to clean up
the remaining 2 crash failures. Defer C3 (CPC) to iter-006 only if C1
+ C2 plateau on the higher-gate-count tracks (grand_tour, slalom).
