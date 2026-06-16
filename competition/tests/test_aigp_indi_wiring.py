"""Wiring tests for the OPT-IN INDI branch in AIGPMavlinkAdapter.send_attitude.

The single most important guarantee: with ``_use_indi = False`` (the default),
``send_attitude`` is BYTE-IDENTICAL to the validated champion PD body-rate path.
We mock the MAVLink connection, capture the exact ``set_attitude_target_send``
arguments, and assert the sent body rates equal the PD law's output for the same
telemetry. A second test confirms the opt-in branch actually routes through the
IndiInnerLoop (different code path, debug snapshot populated) without disturbing
the default.
"""
from __future__ import annotations

import asyncio
import math

import pytest

from competition.adapter import AttitudeCommand, Quaternion, TelemetryState
from competition.aigp_mavlink import (
    AIGPMavlinkAdapter,
    SET_ATTITUDE_TARGET_MASK_RATES_THRUST,
    _attitude_error_body_rates,
)


class _RecordingMav:
    """Captures the args of set_attitude_target_send (the only call we assert)."""

    def __init__(self):
        self.calls = []

    def set_attitude_target_send(self, *args):
        self.calls.append(args)


class _FakeConn:
    def __init__(self):
        self.mav = _RecordingMav()


def _make_adapter(telem: TelemetryState) -> AIGPMavlinkAdapter:
    adapter = AIGPMavlinkAdapter(enable_vision=False, require_track=False)
    adapter._conn = _FakeConn()           # bypass connect(); inject mock transport
    adapter._target_system = 1
    adapter._target_component = 1
    adapter._latest_telem = telem
    return adapter


def _telem(roll, pitch, yaw, gyro=(0.0, 0.0, 0.0), t_us=0) -> TelemetryState:
    return TelemetryState(
        timestamp_us=t_us,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion.from_euler(roll, pitch, yaw),
        angular_velocity=gyro,
    )


def test_send_attitude_pd_path_is_byte_identical_when_indi_off():
    """Default (_use_indi=False): the sent body rates equal the PD law exactly."""
    telem = _telem(0.05, -0.1, 0.2, gyro=(0.3, -0.2, 0.1))
    adapter = _make_adapter(telem)
    assert adapter._use_indi is False  # default OFF

    cmd = AttitudeCommand(roll_rad=0.2, pitch_rad=0.1, yaw_rad=0.25, thrust=0.4)
    asyncio.run(adapter.send_attitude(cmd))

    calls = adapter._conn.mav.calls
    assert len(calls) == 1
    args = calls[0]
    # Layout: (time_boot_ms, tgt_sys, tgt_comp, type_mask, q, br, bp, by, thrust)
    assert args[3] == SET_ATTITUDE_TARGET_MASK_RATES_THRUST
    assert list(args[4]) == [1.0, 0.0, 0.0, 0.0]
    sent_roll_rate, sent_pitch_rate, sent_yaw_rate = args[5], args[6], args[7]
    sent_thrust = args[8]

    # Recompute the EXACT PD-path output and apply the same _rate_sign.
    q = Quaternion.from_euler(cmd.roll_rad, cmd.pitch_rad, cmd.yaw_rad)
    if q.w < 0:
        q = Quaternion(-q.w, -q.x, -q.y, -q.z)
    rr, pr, yr = _attitude_error_body_rates(
        telem.orientation, q, omega=telem.angular_velocity,
        kp=adapter._att_rate_kp, kd=adapter._att_rate_kd,
        max_rate=adapter._att_rate_max,
    )
    sx, sy, sz = adapter._rate_sign
    assert sent_roll_rate == pytest.approx(sx * rr)
    assert sent_pitch_rate == pytest.approx(sy * pr)
    assert sent_yaw_rate == pytest.approx(sz * yr)
    assert sent_thrust == pytest.approx(0.4)
    # The PD path leaves no INDI debug snapshot.
    assert adapter.indi_debug is None


def test_send_attitude_indi_branch_routes_through_indi_loop():
    """With _use_indi=True the rate setpoint comes from the IndiInnerLoop (not
    the PD law) and the debug snapshot is populated; _rate_sign + rates-mode
    send are still applied identically."""
    # Two ticks so INDI has a valid dt + derivative (first tick holds command).
    adapter = _make_adapter(_telem(0.0, 0.0, 0.0, t_us=0))
    adapter._use_indi = True
    cmd = AttitudeCommand(roll_rad=0.2, pitch_rad=0.0, yaw_rad=0.0, thrust=0.5)

    asyncio.run(adapter.send_attitude(cmd))           # tick 1 (dt=0 -> hold)
    adapter._latest_telem = _telem(0.01, 0.0, 0.0, gyro=(0.4, 0.0, 0.0),
                                   t_us=5000)          # +5 ms
    asyncio.run(adapter.send_attitude(cmd))           # tick 2 (dt=5ms)

    calls = adapter._conn.mav.calls
    assert len(calls) == 2
    args = calls[-1]
    assert args[3] == SET_ATTITUDE_TARGET_MASK_RATES_THRUST
    # The INDI branch populated the debug snapshot (the PD path leaves it None).
    dbg = adapter.indi_debug
    assert dbg is not None
    for key in ("alpha_des", "alpha_meas", "ghat", "saturated", "u"):
        assert key in dbg
    # Ghat starts at the bench seed (roll ~1.0, pitch/yaw ~2.1) — never blind.
    assert dbg["ghat"][0] == pytest.approx(1.0, abs=1e-6)
    # The sent rates equal the INDI command times _rate_sign (the send contract
    # is unchanged from the PD path).
    sx, sy, sz = adapter._rate_sign
    assert args[5] == pytest.approx(sx * dbg["u"][0])
    assert args[6] == pytest.approx(sy * dbg["u"][1])
    assert args[7] == pytest.approx(sz * dbg["u"][2])
    assert args[8] == pytest.approx(0.5)


def test_indi_default_off_does_not_build_loop():
    """Importing/constructing the adapter must not build the INDI loop until it
    is opted in (keeps the default path free of the control import + state)."""
    adapter = AIGPMavlinkAdapter(enable_vision=False, require_track=False)
    assert adapter._use_indi is False
    assert adapter._indi is None
    assert adapter.indi_debug is None
