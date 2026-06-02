# Research Analysis: Velocity-Robust Racing Line Selection (Iter-009 F9)

## Diagnosis
The `RacingLineOptimizer` selects optimal lateral offsets by running a multi-start L-BFGS algorithm and scoring the resulting candidates using a kinematic simulation (`_select_by_sim`). This scoring mechanism is currently tightly coupled to the trajectory's `max_velocity_mps`. 

When the velocity is lowered from the default 15.0 m/s to an auto-derived lower value (e.g., 5-6 m/s for tight courses like `aigp_default`), the simulated tracking error landscape changes dramatically. At lower velocities, the centripetal acceleration required to take sharp corners drops, allowing the kinematic simulator to easily track aggressive, corner-cutting paths without saturating acceleration limits. As a result, the optimizer favors the absolute shortest path over a smoother path. 

However, when these aggressively sharp paths are flown by the actual drone dynamics (which include unmodeled aerodynamic drag, latency, and tracking overshoot), the drone crashes at sharp transitions (e.g., gate-1). The core issue is the **coupling of spatial trajectory optimization with velocity-dependent tracking simulation**, leading to velocity-biased local minima that are dynamically unsafe in the real environment.

## Top Techniques for Velocity-Robust Optimization

### 1. Decoupled Geometry and Velocity Profile (Minimum Curvature)
**Citations**: 
- *Heilmeier et al. (2019) "Minimum Curvature Trajectory Planning and Control for an Autonomous Race Car"*
- *Kapania & Gerdes (2016) "Design of a feedback-feedforward steering controller for accurate path tracking and stability at the limits of handling"*

**Concept**: 
The TUM (Technical University of Munich) approach separates the problem into two distinct stages: spatial optimization and temporal (velocity) profiling. The spatial path (the racing line) is optimized purely for geometry—typically minimizing curvature or a weighted sum of path length and curvature squared—entirely independent of the vehicle's velocity. Once the optimal geometric line is established, a velocity profile is calculated based on physical dynamic constraints (e.g., centripetal acceleration limits). This ensures the racing line remains geometrically consistent, smooth, and robust regardless of the specific speed at which it is flown.

### 2. Multi-Fidelity BO / Velocity-Coupled Caching
**Citations**:
- *Jain et al. (2020) "Computing the Racing Line Using Bayesian Optimization"*
- *Chen et al. (2026) "AERO-MPPI"*

**Concept**: 
If trajectory evaluation must remain velocity-coupled to account for specific nonlinear dynamic effects, the optimization landscape should be treated as multi-modal across different velocity regimes. Candidates can be evaluated using a multi-fidelity approach where cheap, low-velocity simulations filter out catastrophic geometric failures, and high-velocity simulations refine the selection. Furthermore, the caching mechanism (`_compute_cache_key`) can be extended to include velocity bands, ensuring that the optimizer remembers different optimal lines for fundamentally different speed profiles.

## Concrete Code Change

To implement the **Decoupled Geometry** approach within the current architecture, we can lock the candidate evaluation simulation to a purely geometric ideal by forcing a high nominal velocity. This ensures that the simulation penalizes sharp, unsmooth turns (which exceed acceleration limits at high speeds), forcing the optimizer to select the smoothest geometric line. The actual flight trajectory is later planned using the correct `auto_velocity`.

```python:planning/racing_line.py
    def _select_by_sim(
        self,
        gates: List[GateWaypoint],
        all_results: list,
        start_position: Tuple[float, float, float],
    ) -> int:
        from .trajectory_optimizer import (
            DroneConstraints, TrajectoryOptimizer, TrajectoryPoint,
        )

        # FIX (Iter-010): Decouple racing line geometry evaluation from auto-velocity.
        # Evaluate geometric candidates at a high nominal speed (e.g., 15.0 m/s)
        # to ensure the selected path prioritizes smoothness and penalizes sharp bends.
        # The actual flight trajectory will be planned subsequently using the true velocity.
        evaluation_velocity = max(self.config.max_velocity_mps, 15.0)

        traj_opt = TrajectoryOptimizer(
            constraints=DroneConstraints(max_velocity=evaluation_velocity),
            dt_sample=0.02,
        )
        
        # ... (rest of the candidate evaluation remains unchanged) ...
```

## Risks and Tradeoffs

- **Decoupled Geometry (High-Velocity Evaluation Constraint)**:
  - *Tradeoff*: A purely geometric racing line (or one evaluated strictly at high velocities) may not exploit low-velocity dynamic capabilities. At very low speeds, a drone can physically take a sharper turn to minimize distance. Forcing a smooth, wide turn results in a suboptimal path length, slightly increasing lap times on tight, slow tracks.
  - *Risk*: The assumption that the smoothest line is universally the fastest line breaks down if the drone is significantly over-actuated relative to the track's target velocity.

- **Velocity-Banded Caching / Multi-Fidelity**:
  - *Tradeoff*: Significantly increases the computational cost of the offline optimization since multiple trajectories must be built and simulated for different velocity conditions before caching.
  - *Risk*: Cache fragmentation and potential boundary discontinuities. If a track's velocity estimate dances on the edge of a cache band, the drone could oscillate between two fundamentally different racing lines across runs, hurting determinism.