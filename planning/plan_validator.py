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

from dataclasses import dataclass, field
from typing import List, Optional

from gate_sequencing.sequencer import (
    GateSequencer,
    GateSpec,
    RaceState,
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
    extras: dict = field(default_factory=dict)    # Per-event metadata


def validate_trajectory(
    trajectory,
    gates: List[GateSpec],
    dt: float = 0.01,
    enforce_in_order: bool = True,
    proximity_pass_distance: float = 0.0,
) -> ValidationResult:
    """Replay a fresh GateSequencer against samples of `trajectory`.

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
        proximity_pass_distance: forward to SequencerConfig.

    Returns:
        ValidationResult. `ok` is True iff the trajectory completes the
        course without crash or DQ. Otherwise carries the diagnostic
        fields needed to localise the failure.
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

    for step in range(samples):
        t = step * dt
        if t > total_time:
            break
        ref = trajectory.sample(t)
        seq.update(tuple(ref.position))

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
    )

    if ok:
        reason = f"trajectory passes all {seq.total_gates} gates cleanly"
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
        samples_evaluated=step + 1 if "step" in dir() else samples,
        first_failure_time_s=first_failure_time,
    )
