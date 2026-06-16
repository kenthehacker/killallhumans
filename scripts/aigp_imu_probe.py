"""Passive rest-state IMU/attitude probe for the live AIGP sim.

Sends NO commands — just connects, lets telemetry flow, and reports the
gravity vector the sim's HIGHRES_IMU reports at rest, plus attitude and
position. This resolves the body-frame convention question:

  - FRD (z-down, what our NED pipeline assumes): at rest the IMU measures
    the normal/reaction specific force ≈ (0, 0, -9.81) — body-Z points DOWN,
    so the upward reaction reads NEGATIVE on z.
  - FLU (z-up, Unreal-native): rest zacc ≈ +9.81.

If zacc reads ~+9.8 the sim is Z-up and our thrust/roll mapping is inverted,
which would directly explain the live Z-climb + lateral (+Y) runaway.

Usage:  python -m scripts.aigp_imu_probe [--seconds 4]
"""
import argparse
import asyncio
import math
import time

from competition.aigp_mavlink import AIGPMavlinkAdapter


async def _run(seconds: float) -> None:
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    await adapter.connect()
    print("connected — reading passively (no commands sent)…")
    t0 = time.monotonic()
    samples = []
    while time.monotonic() - t0 < seconds:
        await asyncio.sleep(0.05)
        telem = adapter.latest_telemetry
        if telem is None:
            continue
        imu = getattr(telem, "imu", None)
        if imu is None:
            continue
        samples.append((
            imu.accel,
            imu.gyro,
            telem.position_ned,
            telem.velocity_ned,
            telem.orientation,
        ))
    await adapter.disconnect()

    if not samples:
        print("NO IMU SAMPLES — adapter saw no HIGHRES_IMU. Is the sim in VQ mode?")
        return

    import numpy as np
    acc = np.array([s[0] for s in samples], dtype=float)
    gyro = np.array([s[1] for s in samples], dtype=float)
    pos = np.array([s[2] for s in samples], dtype=float)
    vel = np.array([s[3] for s in samples], dtype=float)
    print(f"samples: {len(samples)}")
    print("mean accel (xacc,yacc,zacc) = (%.3f, %.3f, %.3f) m/s^2"
          % tuple(acc.mean(axis=0)))
    print("  |accel| mean = %.3f (gravity magnitude ~9.81 expected at rest)"
          % np.linalg.norm(acc.mean(axis=0)))
    print("mean gyro (rad/s) = (%.4f, %.4f, %.4f)" % tuple(gyro.mean(axis=0)))
    print("mean pos NED = (%.3f, %.3f, %.3f)" % tuple(pos.mean(axis=0)))
    print("mean vel NED = (%.3f, %.3f, %.3f)" % tuple(vel.mean(axis=0)))
    q = samples[-1][4]
    if q is not None:
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        print("last orientation q=(w=%.3f x=%.3f y=%.3f z=%.3f) -> yaw=%.3f rad"
              % (q.w, q.x, q.y, q.z, yaw))

    zacc = acc.mean(axis=0)[2]
    print("\n=== FRAME VERDICT ===")
    if zacc < -5:
        print("zacc ~ %.2f (NEGATIVE) -> body-Z DOWN -> FRD/NED. Pipeline assumption CORRECT." % zacc)
    elif zacc > 5:
        print("zacc ~ %.2f (POSITIVE) -> body-Z UP -> FLU. Pipeline assumes NED/FRD => "
              "thrust/Z and roll/Y mappings are INVERTED. This explains the climb+lateral runaway." % zacc)
    else:
        print("zacc ~ %.2f (near zero) -> ambiguous; drone may not be level/at rest." % zacc)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Passive AIGP rest-state IMU/frame probe")
    p.add_argument("--seconds", type=float, default=4.0)
    args = p.parse_args(argv)
    asyncio.run(_run(args.seconds))


if __name__ == "__main__":
    main()
