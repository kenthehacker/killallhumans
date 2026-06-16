"""Tests for the INDI roll-crux read-out tool (scripts/analyze_indi_run.py).

These synthesize small captures in the SAME per-tick record schema the real
recorder emits (scripts/aigp_vq1_run.py's iter-59 ``_recording_callback`` /
``_write_telem_log``) so the verdict classifier is exercised through the real
parse path. To prove the synthetic rows match the live INDI snapshot shape, the
INDI dicts are produced by the actual ``IndiDebug.debug_dict()`` (control
module), not hand-rolled.

The four cases the task requires:
  * RECOVERED         — g_roll converges; achieved roll ~= commanded;
  * BANDWIDTH-LIMITED — g_roll converges; achieved roll ~= 0.53x commanded,
                        roll often saturated;
  * NO INDI           — a run flown without --indi: exits non-zero with the
                        "re-run with --indi" message;
  * INCONCLUSIVE      — an under-excited / no-fast-turn run.
"""
from __future__ import annotations

import gzip
import json
import math
import os

import pytest

from competition.adapter import Quaternion
from control.indi_inner_loop import IndiDebug
from scripts.analyze_indi_run import (
    DEFICIT_RATIO,
    RECOVERED_RATIO,
    analyze,
    classify,
    g_convergence,
    load_rows,
    roll_discriminator,
)


# ---------------------------------------------------------------------------
# Synthetic capture builders — emit the REAL recorder record schema.
# ---------------------------------------------------------------------------
def _indi_snapshot(*, ghat_roll, u_roll, saturated_roll, q_err_roll):
    """A per-tick INDI debug dict via the real IndiDebug.debug_dict() (so the
    test row shape == the live ``adapter.indi_debug`` shape: per-axis length-3
    lists for roll/pitch/yaw + scalar dt)."""
    dbg = IndiDebug(
        alpha_des=(0.0, 0.0, 0.0),
        alpha_meas=(0.0, 0.0, 0.0),
        ghat=(ghat_roll, 2.1, 2.1),
        saturated=(saturated_roll, False, False),
        g_updated=(True, False, False),
        u=(u_roll, 0.0, 0.0),
        q_err_vec=(q_err_roll, 0.0, 0.0),
        dt=0.01,
    )
    # debug_dict() is an instance method on IndiInnerLoop, but the field layout
    # is exactly IndiDebug's; build the dict the same way the loop does.
    return {
        "alpha_des": list(dbg.alpha_des),
        "alpha_meas": list(dbg.alpha_meas),
        "ghat": list(dbg.ghat),
        "saturated": list(dbg.saturated),
        "g_updated": list(dbg.g_updated),
        "u": list(dbg.u),
        "q_err_vec": list(dbg.q_err_vec),
        "dt": dbg.dt,
    }


def _row(t, *, cmd_roll, achieved_roll, roll_rate, indi=None, include_cmd=True):
    """One recorder ``entry`` dict (subset of the real schema sufficient for the
    analyzer): t_wall/t_us, pos/vel, the achieved roll ANGLE + gyro, the
    commanded roll ANGLE (cmd_roll), and the INDI snapshot."""
    q = Quaternion.from_euler(achieved_roll, 0.0, 0.0)
    entry = {
        "t_wall": float(t),
        "t_us": int(t * 1e6),
        "pos": [0.0, 0.0, 0.0],
        "vel": [0.0, 0.0, 0.0],
        "yaw": 0.0,
        "roll": float(achieved_roll),
        "pitch": 0.0,
        "gyro": [float(roll_rate), 0.0, 0.0],
        "gates_passed": 0,
        "target_gate": 0,
        # also carry the achieved orientation quaternion so the q_err fallback
        # path can be exercised when cmd_roll is dropped.
        "orientation": {"w": q.w, "x": q.x, "y": q.y, "z": q.z},
    }
    if include_cmd:
        entry["cmd_roll"] = round(float(cmd_roll), 4)
        entry["cmd_pitch"] = 0.0
        entry["cmd_yaw"] = 0.0
        entry["cmd_thrust"] = 0.4
    if indi is not None:
        entry["indi"] = indi
    return entry


def _converging_g(k, n, final=1.9, seed=1.0):
    """A ghat[0] trajectory that rises from the seed and settles flat well before
    the end (so the analyzer's tail-spread test sees a converged g)."""
    # Settle by ~40% of the run, then hold flat.
    f = min(1.0, k / (0.4 * n))
    return seed + (final - seed) * f


def _slalom_cmd(k, n, amp=0.45):
    """A fast slalom roll command (alternating large bank) — high-demand on roll."""
    return amp * math.sin(2 * math.pi * (k / n) * 6.0)


def _write(tmp_path, name, rows, gz=True):
    path = os.path.join(str(tmp_path), name)
    opener = gzip.open if gz else open
    with opener(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _build_recovered(n=400):
    """g converges AND achieved roll ~= commanded (ratio ~1.0)."""
    rows = []
    for k in range(n):
        cmd = _slalom_cmd(k, n)
        ach = 0.98 * cmd  # achieved tracks commanded -> ratio ~0.98
        indi = _indi_snapshot(ghat_roll=_converging_g(k, n), u_roll=cmd * 1.9,
                              saturated_roll=False, q_err_roll=0.5 * (cmd - ach))
        rows.append(_row(k * 0.01, cmd_roll=cmd, achieved_roll=ach,
                         roll_rate=(0.5 * cmd), indi=indi))
    return rows


def _build_bandwidth_limited(n=400):
    """g converges but achieved roll ~= 0.53x commanded and roll is often
    saturated (the clamp is pinned on the fast turns)."""
    rows = []
    for k in range(n):
        cmd = _slalom_cmd(k, n)
        ach = 0.53 * cmd  # the documented deficit
        big = abs(cmd) > 0.3  # the fast part of each slalom swing saturates
        indi = _indi_snapshot(ghat_roll=_converging_g(k, n), u_roll=cmd * 1.9,
                              saturated_roll=big, q_err_roll=0.5 * (cmd - ach))
        rows.append(_row(k * 0.01, cmd_roll=cmd, achieved_roll=ach,
                         roll_rate=(0.53 * 0.5 * cmd), indi=indi))
    return rows


def _build_underexcited(n=400):
    """A smooth straight: g converges but |cmd_roll| never reaches the fast-turn
    floor, so there are no high-demand segments -> INCONCLUSIVE (under-excited)."""
    rows = []
    for k in range(n):
        cmd = 0.02 * math.sin(2 * math.pi * (k / n) * 2.0)  # tiny bank only
        ach = 0.7 * cmd
        indi = _indi_snapshot(ghat_roll=_converging_g(k, n), u_roll=cmd * 1.9,
                              saturated_roll=False, q_err_roll=0.5 * (cmd - ach))
        rows.append(_row(k * 0.01, cmd_roll=cmd, achieved_roll=ach,
                         roll_rate=(0.5 * cmd), indi=indi))
    return rows


def _build_no_indi(n=50):
    """A PD-path run: NO ``indi`` key on any row."""
    rows = []
    for k in range(n):
        cmd = _slalom_cmd(k, n)
        rows.append(_row(k * 0.01, cmd_roll=cmd, achieved_roll=0.6 * cmd,
                         roll_rate=(0.5 * cmd), indi=None))
    return rows


# ---------------------------------------------------------------------------
# Verdict classifier — the four required cases.
# ---------------------------------------------------------------------------
def test_recovered_verdict(tmp_path, capsys):
    path = _write(tmp_path, "recovered.jsonl.gz", _build_recovered())
    rc = analyze(path, "recovered-synth")
    out = capsys.readouterr().out
    assert rc == 0
    assert "VERDICT: RECOVERED" in out
    # the read-out numbers are surfaced
    assert "achieved/commanded roll" in out
    assert "RECOVERABLE model/effectiveness mismatch" in out


def test_bandwidth_limited_verdict(tmp_path, capsys):
    path = _write(tmp_path, "bw.jsonl.gz", _build_bandwidth_limited())
    rc = analyze(path, "bandwidth-synth")
    out = capsys.readouterr().out
    assert rc == 0
    assert "VERDICT: BANDWIDTH-LIMITED" in out
    assert "TRUE rate/bandwidth wall" in out


def test_no_indi_exits_nonzero_with_helpful_message(tmp_path, capsys):
    path = _write(tmp_path, "pd.jsonl.gz", _build_no_indi())
    rc = analyze(path, "pd-synth")
    out = capsys.readouterr().out
    assert rc == 2  # non-zero
    assert "NO INDI telemetry" in out
    assert "--indi" in out


def test_underexcited_is_inconclusive(tmp_path, capsys):
    path = _write(tmp_path, "smooth.jsonl.gz", _build_underexcited())
    rc = analyze(path, "underexcited-synth")
    out = capsys.readouterr().out
    assert rc == 0
    assert "VERDICT: INCONCLUSIVE" in out
    assert "UNDER-EXCITED" in out


# ---------------------------------------------------------------------------
# Robustness of the parse / signal extraction.
# ---------------------------------------------------------------------------
def test_plain_jsonl_also_parses(tmp_path, capsys):
    """The opener handles a plain (non-gz) .jsonl exactly like iter36_compare."""
    path = _write(tmp_path, "recovered.jsonl", _build_recovered(), gz=False)
    rc = analyze(path, "plain")
    out = capsys.readouterr().out
    assert rc == 0
    assert "VERDICT: RECOVERED" in out


def test_g_does_not_converge_is_inconclusive():
    """A noisy, never-settling g_roll fails the convergence precondition even
    with strong fast-turn excitation -> INCONCLUSIVE (not a false verdict)."""
    n = 400
    indi_rows = []
    rows = []
    for k in range(n):
        # g oscillates wildly forever (never settles).
        g = 1.0 + 0.8 * math.sin(k * 0.7)
        cmd = _slalom_cmd(k, n)
        ach = 0.9 * cmd
        indi = _indi_snapshot(ghat_roll=g, u_roll=cmd * 1.9,
                              saturated_roll=False, q_err_roll=0.5 * (cmd - ach))
        rows.append(_row(k * 0.01, cmd_roll=cmd, achieved_roll=ach,
                         roll_rate=(0.5 * cmd), indi=indi))
        indi_rows.append((k, indi))
    gconv = g_convergence(indi_rows)
    assert gconv["converged"] is False
    disc = roll_discriminator(rows)
    verdict, expl = classify(gconv, disc)
    assert verdict == "INCONCLUSIVE"
    assert "did NOT converge" in expl


def test_q_err_reconstruction_when_no_cmd_roll(tmp_path, capsys):
    """When cmd_roll is absent the analyzer reconstructs the commanded roll from
    the INDI q_err_vec (roll_des ~= roll + 2*q_err[0]); a RECOVERED run still
    reads RECOVERED through that fallback path."""
    n = 400
    rows = []
    for k in range(n):
        cmd = _slalom_cmd(k, n)
        ach = 0.98 * cmd
        # q_err encodes the SAME commanded roll: roll_des = ach + 2*q_err -> set
        # q_err = (cmd - ach)/2 so the reconstruction returns ~cmd.
        indi = _indi_snapshot(ghat_roll=_converging_g(k, n), u_roll=cmd * 1.9,
                              saturated_roll=False, q_err_roll=0.5 * (cmd - ach))
        rows.append(_row(k * 0.01, cmd_roll=cmd, achieved_roll=ach,
                         roll_rate=(0.5 * cmd), indi=indi, include_cmd=False))
    path = _write(tmp_path, "recon.jsonl.gz", rows)
    rc = analyze(path, "recon")
    out = capsys.readouterr().out
    assert rc == 0
    assert "q_err_vec reconstruction" in out
    assert "VERDICT: RECOVERED" in out


def test_load_rows_skips_blank_lines(tmp_path):
    path = os.path.join(str(tmp_path), "blanks.jsonl")
    with open(path, "wt") as f:
        f.write(json.dumps({"a": 1}) + "\n")
        f.write("\n")
        f.write("   \n")
        f.write(json.dumps({"a": 2}) + "\n")
    rows = load_rows(path)
    assert rows == [{"a": 1}, {"a": 2}]


def test_empty_capture_returns_one(tmp_path, capsys):
    path = _write(tmp_path, "empty.jsonl.gz", [])
    rc = analyze(path, "empty")
    out = capsys.readouterr().out
    assert rc == 1
    assert "empty capture" in out


def test_missing_file_returns_one(capsys):
    rc = analyze("does/not/exist.jsonl.gz", "missing")
    err = capsys.readouterr().err
    assert rc == 1
    assert "not found" in err


def test_thresholds_bracket_the_053_anchor():
    """Sanity: the deficit band sits above 0.53 and below the recovered band, so
    a ~0.53 run reads BANDWIDTH-LIMITED and a ~1.0 run reads RECOVERED."""
    assert 0.53 < DEFICIT_RATIO < RECOVERED_RATIO < 1.0
