# AI Grand Prix — Authoritative Rules Distilled

Sources: `VADR-TS-002 issue 00.02` (2026-05-08 PDF) + theaigrandprix.com/previousupdates/

## Course
- **Start gate → intermediate gates → finish gate**, passed in **strict order**.
- VQ1 has **< 10 gates**, VQ2 has **< 20 gates**. Positions change between VQ1 and VQ2.
- Same course for all teams within a qualifier; **course geometry identical, physics identical, deterministic** — no per-run randomness.
- Full 3D environment with **elevation changes**; non-gated decorative objects may be present.
- Gates "visually distinctive to the environment, but consistent throughout the Virtual Qualifier 1 track."

## Drone (competition)
- Chassis: **280 mm × 280 mm × 160 mm** (W×L×H).
- Mass / thrust / inertia not specified — must be inferred via SITL calibration.

## Gates
- Outer frame: **2700 mm × 2700 mm × 260 mm** (W×H×depth).
- **Inner square opening: 1500 mm × 1500 mm × 260 mm**.
- Frame thickness on each side: (2700 − 1500) / 2 = **600 mm**, with a **140 mm bevel** at the corners.

## Coordinate frames
- MAVLink2 NED.
- `MAV_FRAME_LOCAL_NED`: origin at the arm point on the ground, fixed.
- `MAV_FRAME_BODY_NED`: origin at vehicle; X forward, Y right, Z down.
- **No GPS, no global position, no depth data, no engine RPMs, no battery SoC, no steering inputs**.

## Camera
- **Tilted 20° upward** from body-frame.
- Pinhole, no distortion.
- Resolution **640 × 360**.
- Intrinsics `[cx, cy] = [320, 180]`, `[fx, fy] = [320, 320]`.
- VFoV = **90°** (so HFoV ≈ 2·atan(320/320) = 90° as well — square FoV from these intrinsics).
- Camera frames in repo currently default to 640×480 — **fix to 640×360**.

## Communication
- MAVLink2 over **UDP** via `c_library_v2` / MAVSDK-compatible interfaces.
- Vision stream: **separate UDP**, port **5600**, **30 Hz**, JPEG, chunked with 24B little-endian header (`frame_id` u32, `chunk_id` u16, `total_chunks` u16, `jpeg_size` u32, `payload_size` u32, `sim_time_ns` u64).
- **Physics rate: 120 Hz** | **Command rate: < 100 Hz** | **Heartbeat ≥ 2 Hz**.

## Supported MAVLink messages
- Sim → Client: `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU`, `TIMESYNC` (and `HIGHRES_IMU` listed twice).
- Client → Sim: `SET_POSITION_TARGET_LOCAL_NED`, `SET_ATTITUDE_TARGET`.
- Body ↔ IMU is identity.

## Runtime
- Python 3.14.2 known-good. Other env permitted.
- DCL sim is **Windows 11** only (8 GB VRAM minimum). Linux **NOT supported**.
- Min hardware: i5-10400F / Ryzen 5 3600, RTX 2060 Super / 9060XT, 16 GB RAM.

## Scoring & rules
- **VQ1: focus on completion** (just finish the course).
- **VQ2: fastest valid time** (gates must be in correct order).
- **Max run duration: 8 minutes per attempt**.
- Unlimited attempts within the qualification window.
- "No human interaction during runs"; "no rewriting or altering game files"; "no color changes, screen tricks, or disabling collision detection" → DQ.

## Timeline
- VQ1 opened May 2026, closes at end of VQ2.
- VQ2 launches June, closes mid-to-late July 2026.
- Physical qualifier: **September 2026** (California).
- Grand Prix Final: **November 2026** (Ohio).

## Officially undefined (room for our adversarial harness)
- No explicit crash-penalty spec — but our self-test must treat crash as fail.
- No explicit "wrong-order pass" handling — but the rules say correct order, so we enforce it.
- No simulator-bug disclosures.

## Implications for our stack
1. Gate inner-opening **1.5 m** (we currently default to 1.2 m). The geometric crash zone, `pass_through_margin`, and trajectory clearance must derive from the actual gate spec, not from a hard-coded 1.2 m.
2. Drone **W=280mm** is bigger than our CF2X model (~92 mm). Collision-clearance margins must account for the larger footprint.
3. **Forward camera tilted 20° upward** — gate visibility math (e.g. perception-aware lookahead) must include this tilt.
4. **120 Hz physics** with `<100 Hz` command rate — our control loop targeting 100 Hz is on the edge; verify we don't accidentally overrun.
5. We must support `SET_POSITION_TARGET_LOCAL_NED` AND `SET_ATTITUDE_TARGET` paths — the second is the low-level fallback if MPC saturates.
6. The JPEG-chunk UDP vision protocol on port 5600 is **separate** from MAVLink — needs its own reassembler; our current `competition/mavlink_bridge.py` may not handle it.
