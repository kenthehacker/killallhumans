"""Controlled open-loop attitude/thrust bench for the live AIGP sim.

For each phase: SIM_RESET to a clean spawn, arm, then hold ONE fixed
SET_ATTITUDE_TARGET setpoint at ~50 Hz for a short window while recording
the drone's response (pos/vel/attitude/IMU). Reports net motion so we can
read off, empirically:

  * hover throttle      — which thrust holds altitude (vz ~ 0)
  * does the sim honor attitude at all — does commanded roll/pitch show up
    in the measured ATTITUDE, and does the drone translate accordingly?
  * command->motion sign — +pitch (facing yaw=pi, nose toward -X) should
    drive -X (forward along the course); +roll sign -> which way in Y?

Short windows (default 1.2 s) + reset between phases keep any runaway
bounded. Spawn heading is yaw=pi, so we command yaw=pi to avoid a slew.

Usage:  python -m scripts.aigp_bench [--hold 1.2]
"""
import argparse
import asyncio
import math
import time

import numpy as np

from competition.adapter import AttitudeCommand
from competition.aigp_mavlink import AIGPMavlinkAdapter

YAW = math.pi  # spawn heading


def _attitude_error_body_rates(q_cur, q_des, omega=(0.0, 0.0, 0.0),
                               kp=8.0, kd=0.0, max_rate=6.0):
    """PD body-rate command driving q_cur -> q_des.

    q_err = conj(q_cur) (x) q_des in the body frame; its vector part is
    ~sin(theta/2)*axis, so 2*kp*vec is a singularity-free proportional rate
    (euler errors cross-couple when tilted). The -kd*omega term damps the
    cascade (sim rate-loop + our P loop oscillates ~5 Hz without it).
    Shortest-path via w>=0.
    """
    def conj(q):
        return (q[0], -q[1], -q[2], -q[3])
    def mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        )
    qc = (q_cur.w, q_cur.x, q_cur.y, q_cur.z)
    qd = (q_des.w, q_des.x, q_des.y, q_des.z)
    ew, ex, ey, ez = mul(conj(qc), qd)
    if ew < 0:
        ex, ey, ez = -ex, -ey, -ez
    rates = [
        2.0 * kp * ex - kd * omega[0],
        2.0 * kp * ey - kd * omega[1],
        2.0 * kp * ez - kd * omega[2],
    ]
    return tuple(max(-max_rate, min(max_rate, r)) for r in rates)


def _send_attitude_with_yawrate(adapter, roll, pitch, yaw, thrust, yaw_rate):
    """Raw SET_ATTITUDE_TARGET: attitude quaternion for roll/pitch/yaw PLUS
    an explicit body yaw-rate (yaw-rate ignore bit CLEARED). type_mask = 3
    = ignore body roll-rate (bit0) + pitch-rate (bit1), USE yaw-rate (bit2
    clear) and USE attitude (bit7 clear). Tests whether the sim needs a
    yaw-RATE command to stop free-spinning."""
    from competition.adapter import Quaternion
    q = Quaternion.from_euler(roll, pitch, yaw)
    if q.w < 0:
        q = Quaternion(-q.w, -q.x, -q.y, -q.z)
    thrust = max(0.0, min(1.0, thrust))
    adapter._conn.mav.set_attitude_target_send(
        adapter._time_boot_ms(),
        adapter._target_system,
        adapter._target_component,
        0b00000011,  # ignore roll/pitch body rates; use yaw rate + attitude
        [q.w, q.x, q.y, q.z],
        0.0, 0.0, yaw_rate,
        thrust,
    )


async def _hold(adapter, label, roll, pitch, yaw, thrust, hold_s, rate_hz=50.0,
                yaw_rate=None, mode="attitude", kp=8.0, kd=0.0, max_rate=4.0,
                fixed_rate=(0.0, 0.0, 0.0)):
    """Reset to spawn, arm, hold one setpoint, record response.

    If ``yaw_rate`` is not None, command attitude+yaw-rate (mask 3) instead
    of pure attitude (mask 7) to test the sim's yaw-control mode.
    """
    await adapter.reset()
    # Let the drone actually settle back at spawn. 0.4 s was too short — a
    # drone left climbing by the previous phase was still moving when the next
    # phase began, contaminating its first samples. Poll until |vel| is small.
    settle_t0 = time.monotonic()
    while time.monotonic() - settle_t0 < 3.0:
        await asyncio.sleep(0.2)
        t = adapter.latest_telemetry
        if t is not None and t.velocity_ned is not None:
            v = t.velocity_ned
            if all(math.isfinite(x) for x in v) and (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5 < 0.5:
                break
    await adapter.arm()
    base = adapter.latest_telemetry
    p0 = np.array(base.position_ned) if base else np.zeros(3)

    rec = []
    dt = 1.0 / rate_hz
    t0 = time.monotonic()
    while time.monotonic() - t0 < hold_s:
        if mode == "idle":
            pass  # send nothing — does the armed sim spin on its own?
        elif mode == "rate":
            from competition.adapter import AttitudeRateCommand
            rr, pr, yr = fixed_rate
            await adapter.send_attitude_rate(
                AttitudeRateCommand(rr, pr, yr, thrust))
        elif mode == "att_via_rate":
            # Quaternion attitude-error -> body-rate PD loop (proposed real fix).
            from competition.adapter import AttitudeRateCommand, Quaternion
            telem = adapter.latest_telemetry
            qd = Quaternion.from_euler(roll, pitch, yaw)
            qc = telem.orientation if (telem and telem.orientation) else Quaternion()
            om = telem.angular_velocity if (telem and telem.angular_velocity) else (0.0, 0.0, 0.0)
            rr, pr, yr = _attitude_error_body_rates(
                qc, qd, omega=om, kp=kp, kd=kd, max_rate=max_rate)
            await adapter.send_attitude_rate(AttitudeRateCommand(rr, pr, yr, thrust))
        elif yaw_rate is not None:
            _send_attitude_with_yawrate(adapter, roll, pitch, yaw, thrust, yaw_rate)
        else:
            await adapter.send_attitude(AttitudeCommand(roll, pitch, yaw, thrust))
        await asyncio.sleep(dt)
        telem = adapter.latest_telemetry
        if telem is not None:
            q = telem.orientation
            meas_yaw = (math.atan2(2 * (q.w * q.z + q.x * q.y),
                                   1 - 2 * (q.y * q.y + q.z * q.z))
                        if q is not None else None)
            # measured roll/pitch from quaternion (ZYX)
            if q is not None:
                sinr = 2 * (q.w * q.x + q.y * q.z)
                cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
                meas_roll = math.atan2(sinr, cosr)
                sinp = 2 * (q.w * q.y - q.z * q.x)
                meas_pitch = math.asin(max(-1, min(1, sinp)))
            else:
                meas_roll = meas_pitch = None
            gyro = telem.angular_velocity or (0.0, 0.0, 0.0)
            rec.append((np.array(telem.position_ned),
                        np.array(telem.velocity_ned),
                        meas_roll, meas_pitch, meas_yaw, np.array(gyro)))

    if not rec:
        print(f"[{label}] NO TELEMETRY")
        return
    pf = rec[-1][0]
    vf = rec[-1][1]
    dp = pf - p0
    mr = np.mean([r[2] for r in rec if r[2] is not None]) if rec else 0
    mp = np.mean([r[3] for r in rec if r[3] is not None]) if rec else 0
    # Oscillation metric: direction reversals in measured roll/pitch = "bounce".
    rolls = np.array([r[2] for r in rec if r[2] is not None])
    pitches = np.array([r[3] for r in rec if r[3] is not None])
    def _reversals(a):
        return int(np.sum(np.diff(np.sign(np.diff(a))) != 0)) if len(a) > 2 else 0
    roll_rev = _reversals(rolls); pitch_rev = _reversals(pitches)
    roll_hz = roll_rev / (2 * hold_s) if hold_s > 0 else 0
    yaws = [r[4] for r in rec if r[4] is not None]
    yaw_span = 0.0
    if len(yaws) > 1:
        uw = np.unwrap(yaws)
        yaw_span = float(uw[-1] - uw[0])
    print(f"[{label}] cmd(roll={roll:+.2f} pitch={pitch:+.2f} yaw={yaw:+.2f} thr={thrust:.2f})")
    print(f"    net dPOS NED = (dX={dp[0]:+.2f} dY={dp[1]:+.2f} dZ={dp[2]:+.2f})  "
          f"final vel=({vf[0]:+.1f},{vf[1]:+.1f},{vf[2]:+.1f})")
    r0, r1 = (rolls[0], rolls[-1]) if len(rolls) else (0, 0)
    p0a, p1a = (pitches[0], pitches[-1]) if len(pitches) else (0, 0)
    max_tilt = float(max((abs(x) for x in np.concatenate([rolls, pitches])), default=0))
    gyros = np.array([r[5] for r in rec if len(r) > 5])
    gmean = gyros.mean(axis=0) if len(gyros) else np.zeros(3)
    gp95 = float(np.percentile(np.linalg.norm(gyros, axis=1), 95)) if len(gyros) else 0.0
    print(f"    meas roll {r0:+.2f}->{r1:+.2f}(rev={roll_rev},~{roll_hz:.1f}Hz) "
          f"pitch {p0a:+.2f}->{p1a:+.2f}(rev={pitch_rev}) yaw_span={yaw_span:+.2f}rad "
          f"({yaw_span/(2*math.pi):+.2f}rev)  n={len(rec)}")
    print(f"    GYRO mean=({gmean[0]:+.2f},{gmean[1]:+.2f},{gmean[2]:+.2f}) p95={gp95:.1f}  "
          f"max_tilt={max_tilt:.2f}rad {'<<FLIP' if max_tilt > 1.3 else 'ok'}")


async def _run(hold_s: float) -> None:
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    await adapter.connect()
    print("connected. Running controlled bench (resets between phases)…\n")

    HOV = 0.27  # measured hover throttle (zero-rate sweep: vz≈0 between 0.27-0.28)

    # --- 0. PER-AXIS RATE ID (raw, no sign flip): command a small body rate on
    #        ONE axis at a time and read the gyro sign+magnitude. Settles the
    #        contaminated "all-3-flipped, 2.5x" claim (which came from a single
    #        3-axis-simultaneous step). _rate_sign forced to +1 so we see the
    #        sim's RAW response. gyro_axis sign -> the true _rate_sign; gyro/0.2
    #        -> the true per-axis gain.
    print("=== 0. per-axis rate ID (raw _rate_sign=+1, 0.2 rad/s single axis) ===")
    saved_sign = adapter._rate_sign
    adapter._rate_sign = (1.0, 1.0, 1.0)
    for label, fr in [("roll+0.2", (0.2, 0.0, 0.0)),
                      ("pitch+0.2", (0.0, 0.2, 0.0)),
                      ("yaw+0.2", (0.0, 0.0, 0.2))]:
        await _hold(adapter, f"rawrate {label}", 0.0, 0.0, YAW, HOV, hold_s,
                    mode="rate", fixed_rate=fr)
    adapter._rate_sign = saved_sign  # restore

    # --- 1. MASK-7 ATTITUDE MODE RETEST: let the sim close its OWN rate loop.
    #        The handoff says this "spins", but that was measured with the
    #        sign/PD bugs live. If it holds attitude cleanly now, the entire
    #        homemade attitude->rate PD can be deleted. Cheapest possible win.
    print("\n=== 1. mask-7 attitude mode retest (sim's own inner loop) ===")
    adapter._use_rate_control = False
    for (label, roll, pitch) in [("level", 0.0, 0.0),
                                 ("pitch-0.2", 0.0, -0.2),
                                 ("roll+0.2", 0.2, 0.0)]:
        await _hold(adapter, f"mask7 {label}", roll, pitch, YAW, HOV, hold_s,
                    mode="attitude")
    adapter._use_rate_control = True  # restore

    # --- 2. LOW-GAIN attitude->rate PD sweep (att_via_rate = same path as
    #        send_attitude). Current (2.0,2.5,1.0) limit-cycles at 9 Hz. Find
    #        gains that hold attitude cleanly: meas converges, gyro p95 low,
    #        few reversals. Last row reproduces the bug as a control.
    print("\n=== 2. low-gain attitude->rate PD sweep (vs current 2.0/2.5/1.0) ===")
    for (kp, kd, mx) in [(0.5, 0.2, 0.5), (0.8, 0.2, 0.6),
                         (0.3, 0.1, 0.4), (2.0, 2.5, 1.0)]:
        for (label, roll, pitch) in [("pitch-0.3", 0.0, -0.3), ("roll+0.3", 0.3, 0.0)]:
            await _hold(adapter, f"pd kp{kp} kd{kd} mx{mx} {label}",
                        roll, pitch, YAW, HOV, hold_s,
                        mode="att_via_rate", kp=kp, kd=kd, max_rate=mx)

    await adapter.reset()
    await adapter.disconnect()
    print("\nbench complete.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="AIGP controlled attitude/thrust bench")
    p.add_argument("--hold", type=float, default=1.2, help="seconds to hold each setpoint")
    args = p.parse_args(argv)
    asyncio.run(_run(args.hold))


if __name__ == "__main__":
    main()
