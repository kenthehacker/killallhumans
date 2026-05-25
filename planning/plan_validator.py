"""
Plan validator (iter-004 research-swarm Phase 1).

Origin: GPT-5.5 extra-high's #1 pick in the iter-003 research swarm
(`.loop/research/gpt-55-xhigh.md` C1) — "before accepting a trajectory,
sample at high resolution and replay through a fresh GateSequencer.
Reject if it would DQ or crash."

This is a cheap diagnostic gate that reuses our existing honesty
infrastructure. It does NOT fix the underlying overfitting (that's
Phase 2 SFC corridor work) but it does:
  - flag plans that would DQ at runtime (so the bench surfaces them
    BEFORE the flight starts, not 2.4s in)
  - give us a metric (`validator_passed`) that iter-005+ corridor
    work can use as a success bar
  - catch regressions where a future planner change produces
    legal-looking-but-actually-illegal trajectories

The validator is deliberately stateless and dependency-free: it
takes a trajectory and a list of GateSpecs, replays a fresh sequencer
against samples, and returns a structured result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from gate_sequencing.sequencer import (
    GateSequencer,
    GateSpec,
    SequencerConfig,
)


@dataclass
class ValidationResult:
    """Outcome of replaying the sequencer on a candidate trajectory."""
    ok: bool                            # True iff the trajectory would
                                        # legally complete the course
    reason: str                         # Human-readable summary
    gates_passed: int                   # Gates credited during replay
    total_gates: int
    crashed: bool                       # Sequencer recorded a frame strike
    disqualified: bool                  # Sequencer DQ'd (out-of-order, etc.)
    dq_reason: Optional[str]            # If disqualified, the reason
    last_crash_gate: Optional[str]      # If crashed, the gate hit
    samples_evaluated: int              # How many trajectory points were sampled
    first_failure_time_s: Optional[float] = None  # Sim time of DQ/crash


def validate_trajectory(
    trajectory,
    gates: List[GateSpec],
    dt: float = 0.01,
    enforce_in_order: bool = True,
    proximity_pass_distance: float = 1.0,
    ground_z_threshold: float = 0.05,
    ceiling_z_threshold: float = 20.0,
) -> ValidationResult:
    """Replay a fresh GateSequencer against samples of `trajectory`.

    Iter-005b (Opus F1 / composer F1 / gpt-55-xhigh F3 MAJOR consensus):
    `proximity_pass_distance` defaults to 1.0 to match the synthetic
    bench's SequencerConfig (was 0.0, which produced false-negatives
    on plans that legitimately use proximity-credit at gate close-pass).
    Callers wanting the strict no-proximity check can pass 0.0.

    Iter-006 (Opus F5 MAJOR): airspace bounds. The bench terminates a
    run with `crash_ground` / `crash_ceiling` when the kinematic drone
    exits the z envelope; the validator now flags trajectories that
    would do the same. Without this the validator could say "ok" on a
    plan that the bench would terminate at the first ground clip.

    Args:
        trajectory: any object with `.total_time` and `.sample(t)` returning
            a TrajectoryPoint with `.position`. The repo's `RaceTrajectory`
            qualifies; so does any duck-typed stub.
        gates: GateSpec list — the same gates the runtime sequencer will
            check against. Order matters; the sequencer's `_current_idx`
            advances through them in sequence_index order.
        dt: sample step in seconds. 0.01 matches the bench default.
        enforce_in_order: if True (default), the sequencer's strict
            in-order DQ logic is on — same as the runtime.
        proximity_pass_distance: forward to SequencerConfig. Default 1.0
            matches the synthetic bench's SequencerConfig.
        ground_z_threshold: if any sample has z < this, fail as
            `crash_ground` (matches bench at scripts/benchmark.py:445).
        ceiling_z_threshold: if any sample has z > this, fail as
            `crash_ceiling` (matches bench at scripts/benchmark.py:449).

    Returns:
        ValidationResult. `ok` is True iff the trajectory completes the
        course without crash, DQ, or airspace exit. Otherwise carries the
        diagnostic fields needed to localise the failure.
    """
    total_time = float(getattr(trajectory, "total_time", 0.0))
    if total_time <= 0 or not gates:
        return ValidationResult(
            ok=False,
            reason="empty trajectory or zero gates",
            gates_passed=0,
            total_gates=len(gates),
            crashed=False,
            disqualified=False,
            dq_reason=None,
            last_crash_gate=None,
            samples_evaluated=0,
        )

    cfg = SequencerConfig(
        enforce_in_order=enforce_in_order,
        proximity_pass_distance=proximity_pass_distance,
    )
    seq = GateSequencer(gates, cfg)
    seq.start()

    samples = int(total_time / dt) + 1
    first_failure_time: Optional[float] = None
    airspace_violation: Optional[str] = None
    # Iter-008 (Opus F4 dead-code cleanup): track samples evaluated
    # explicitly instead of using a brittle `"step" in dir()` guard
    # that doesn't work if the for-loop body never executes.
    samples_processed = 0

    for step in range(samples):
        samples_processed = step + 1
        t = step * dt
        if t > total_time:
            break
        ref = trajectory.sample(t)
        pos = tuple(ref.position)
        # Iter-006 F5: airspace bounds match the kinematic bench's
        # ground/ceiling checks. A plan that the bench would terminate
        # at z<0.05 / z>20 must NOT validate ok.
        if pos[2] < ground_z_threshold:
            airspace_violation = "crash_ground"
            first_failure_time = float(t)
            break
        if pos[2] > ceiling_z_threshold:
            airspace_violation = "crash_ceiling"
            first_failure_time = float(t)
            break
        seq.update(pos)

        if first_failure_time is None and (
            seq.is_disqualified or seq.last_crash is not None
        ):
            first_failure_time = float(t)
            break  # don't keep sampling past a terminal event

    crashed = seq.last_crash is not None
    disqualified = seq.is_disqualified
    ok = (
        seq.is_complete
        and not crashed
        and not disqualified
        and airspace_violation is None
    )

    if ok:
        reason = f"trajectory passes all {seq.total_gates} gates cleanly"
    elif airspace_violation:
        reason = (
            f"airspace exit at t={first_failure_time:.2f}s: "
            f"{airspace_violation}"
        )
    elif disqualified:
        reason = f"DQ at t={first_failure_time:.2f}s: {seq.dq_reason}"
    elif crashed:
        reason = (
            f"crash at t={first_failure_time:.2f}s: "
            f"gate-strut hit on {seq.last_crash[0]}"
        )
    elif not seq.is_complete:
        reason = (
            f"incomplete: {seq.gates_passed}/{seq.total_gates} gates "
            "passed during full-trajectory replay"
        )
    else:
        reason = "unknown failure"

    return ValidationResult(
        ok=ok,
        reason=reason,
        gates_passed=seq.gates_passed,
        total_gates=seq.total_gates,
        crashed=crashed,
        disqualified=disqualified,
        dq_reason=seq.dq_reason,
        last_crash_gate=seq.last_crash[0] if seq.last_crash else None,
        samples_evaluated=samples_processed,
        first_failure_time_s=first_failure_time,
    )
