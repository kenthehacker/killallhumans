"""Read-only probe of the sim's race-start semantics after SIM_RESET.

Connects, then issues SIM_RESET twice and prints the race_start_boot_time_ms /
race_started / active_gate timeline after each reset. NO arming, NO setpoints —
the drone is never commanded, so this cannot "jump the start". The goal is to
learn exactly when (relative to the reset) the sim flips race_started to True for
the FRESH countdown, and whether it reports a STALE True immediately after reset.
"""
import asyncio
import time


async def main() -> None:
    from competition.aigp_mavlink import AIGPMavlinkAdapter

    adapter = AIGPMavlinkAdapter(enable_vision=False)
    await adapter.connect("udpin:127.0.0.1:14550")
    print("connected\n")

    for trial in range(2):
        rs = adapter.race_status
        before = rs.race_start_boot_time_ms if rs is not None else None
        print(f"--- trial {trial}: race_start_boot_time_ms BEFORE reset = {before} ---")
        await adapter.reset()
        t0 = time.monotonic()
        last = None
        while time.monotonic() - t0 < 7.0:
            await asyncio.sleep(0.1)
            el = time.monotonic() - t0
            rs = adapter.race_status
            if rs is None:
                continue
            now = rs.sim_boot_time_ms
            go = rs.race_start_boot_time_ms
            past_go = (go >= 0) and (now >= go)
            tup = (now // 200, go, past_go, rs.active_gate_index)
            if tup != last:
                print(f"  t={el:4.1f}s  sim_boot_ms={now}  start_boot_ms={go}  "
                      f"sim>=start(GO)={past_go}  active_gate={rs.active_gate_index}  "
                      f"race_started(prop)={rs.race_started}")
                last = tup
        print()

    await adapter.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
