"""End-to-end closed-loop flight regression for the VQ1 course.

Runs the real RacePipeline control stack (geometric tracker + gate sequencer
+ dynamic replanner) against point-mass NED dynamics via
``scripts.sim_closed_loop.run`` and asserts the drone actually flies the
course: completes, passes every gate, and does so without the failure
signatures this loop has been fixing (spinning, circling, divergence).

This is the capstone guard: a regression in tracking, sequencing, off-track
detection, or replanning that lets the drone spin / circle / stall will fail
here even though the unit suites stay green (the exact gap that let the
original "drone spins in circles" bug ship).
"""

import math

import pytest

from scripts.sim_closed_loop import run


@pytest.fixture(scope="module")
def flight():
    # Default competition speed; start facing +X so the drone must first
    # yaw ~180 deg to face gate-0 (the historical spin trigger).
    telem_log, summary = run(max_speed=8.0, start_yaw=0.0, max_sim_s=60.0)
    return telem_log, summary


def test_race_completes_all_gates(flight):
    _, summary = flight
    assert summary["termination_reason"] == "race_complete"
    assert summary["gates_passed"] == summary["total_gates"] == 6


def test_no_spin_or_circling(flight):
    telem_log, _ = flight
    # Net displacement should be most of the path length: a circling/spinning
    # drone piles up path length without net progress.
    xs = [r["pos"][0] for r in telem_log]
    ys = [r["pos"][1] for r in telem_log]
    zs = [r["pos"][2] for r in telem_log]
    pathlen = sum(
        math.dist((xs[i], ys[i], zs[i]), (xs[i - 1], ys[i - 1], zs[i - 1]))
        for i in range(1, len(telem_log))
    )
    net = math.dist((xs[-1], ys[-1], zs[-1]), (xs[0], ys[0], zs[0]))
    assert net > 150.0, f"insufficient progress (net {net:.1f} m)"
    assert pathlen / net < 1.5, f"circling: path/net = {pathlen/net:.2f}"


def test_reference_was_tracked(flight):
    telem_log, _ = flight
    # The recorder fix must populate ref_pos for (essentially) every tick.
    have_ref = sum(1 for r in telem_log if r.get("ref_pos") is not None)
    assert have_ref / len(telem_log) > 0.99


def test_no_replan_storm_on_clean_course(flight):
    _, summary = flight
    # A clean run of a feasible course should not need to replan at all.
    assert summary["replans"] == 0, (
        f"unexpected replans ({summary['replans']}) on a clean course"
    )


def test_recovers_from_midcourse_disturbance():
    """Resilience: a strong lateral gust mid-course must be detected (the
    replanner fires) and the drone must still finish the course. Guards the
    recovery stack fixed in iters 3 (replanner cooldown trigger-loss) and 6
    (cross-track off-track detection) working together end-to-end."""
    _, summary = run(
        max_speed=8.0, start_yaw=0.0, max_sim_s=45.0,
        perturb=(5.0, (0.0, 15.0, 0.0)),  # +15 m/s lateral kick at t=5 s
    )
    assert summary["termination_reason"] == "race_complete"
    assert summary["gates_passed"] == 6
    # The disturbance must actually have been noticed and acted on, not
    # silently ignored — at least one replan (or a RECOVERY excursion).
    assert summary["replans"] >= 1 or summary["entered_recovery"], (
        "disturbance was not detected by the recovery stack"
    )
