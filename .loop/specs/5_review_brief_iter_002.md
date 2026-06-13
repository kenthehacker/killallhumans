# Adversarial Review Brief — Iter 002

You are one of EIGHT agents adversarially reviewing the iter-002
implementation at `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/`.
The breakdown is: 1× Opus 4.7 max-thinking, 2× GPT-5.5 extra-high, 5× Composer 2.5.

Iter-002 shipped 6 patches addressing iter-001 review findings. Your one
job: find what they got wrong, what they didn't go far enough on, and
what new issues they introduced.

## What was shipped in iter-002 (6 commits, branch aigp-vq1-loop)
1. `cf03729` — iter-001b review fixes: calibration physics (Opus F1 BLOCKER) + RacePipeline DQ/crash termination + CameraFrame default 360
2. `f98e4f5` — Opus F2 (multi-gate-per-tick) + F3 (future-gate strut crash) + F14 (DQ strict opening)
3. `04a429d` — Opus F9 (vision jpeg_size validation) + F5 (ILC relative threshold)
4. `434801e` — composer-25-4 F8 (HFoV/VFoV consistency) + 8-minute VQ1 timeout
5. `346c626` — B4 (vision UDP → mavlink_bridge wiring) + B5 (camera pitch threaded to intrinsics)
6. `8214b9c` — MINOR cleanups (GateGeometry default, ILC step rounding, bench crashed/disqualified split)

Total: ~1200 LOC changed across the 6 commits, 334 tests passing.

## Read first
1. `.loop/specs/0_charter.md` — hard constraints
2. `.loop/synthesis/iter_001_review.md` — what was on the iter-001 punch list (your bar)
3. The 6 commits above (`git show <hash>`).
4. The new files: `competition/vision_udp.py` (added VisionUdpListener), `gate_sequencing/sequencer.py` (DQ/crash logic).

## What the iter-001 review identified that ITER-002 *should* have closed
(Use this list to grade iter-002 — anything still broken is a HARD MISS.)
- **B1 calibration physics** (Opus F1, BLOCKER) — should be FIXED in cf03729. Verify hover sample case returns positive k_t ≈ 21.8.
- **B2 RacePipeline DQ/crash termination** — should be FIXED in cf03729. Verify `should_stop` and the control-callback early-return both honour `is_disqualified`, `is_timed_out`, `last_crash`.
- **B3 CameraFrame default 360** — should be FIXED in cf03729.
- **B4 Vision UDP wiring** — should be FIXED in 346c626. Verify `MAVLinkBridge.get_camera_frame` actually returns the listener's latest frame, NOT None.
- **B5 Camera tilt in PnP** — should be FIXED in 346c626. Verify `PipelineConfig.camera_pitch_offset_rad` flows through to `CameraIntrinsics.pitch_offset_rad`. Note: position-only PnP with coincident origins doesn't NEED the tilt (Opus F13), but the wiring should still be present.
- **Opus F2 multi-gate per tick** — should be FIXED in f98e4f5.
- **Opus F3 future-gate strut crash** — should be FIXED in f98e4f5.
- **Opus F14 DQ strict opening** — should be FIXED in f98e4f5.
- **Opus F9 vision jpeg_size validation** — should be FIXED in 04a429d.
- **Opus F5 ILC absolute threshold** — should be FIXED in 04a429d.
- **8-min VQ1 timeout** — should be FIXED in 434801e.
- **HFoV/VFoV math** — should be FIXED in 434801e.
- **GateGeometry default 1.2m** — should be FIXED in 8214b9c.
- **Bench overloads crashed=True for DQ** — should be FIXED in 8214b9c.
- **ILC pt_to_step truncation** (Opus F6) — should be FIXED in 8214b9c.

## What the iter-001 review identified that iter-002 EXPLICITLY DEFERRED
(Flag if any are now production-critical with the new code paths in place.)
- enforce_in_order false-DQ on legitimate replanner recovery (3/7 reviews MINOR)
- `_delivered_ids` cap allows duplicate frames after ~64 (Opus F10 + others)
- `aigp_default.json` is topology-simplified, not a true 1.25× scale (Opus F7)
- `vel_scale=0.5` high default is median-of-race_01 (Opus F8)
- SimplePositionTracker silently ignores use_residual (Opus F11)
- Adversarial sequencer test docstring/name drift (Opus F12)

## What to specifically hunt for in iter-002 NEW code

**`gate_sequencing/sequencer.py`** —
- Multi-gate-per-tick drain loop: does it correctly handle the case where ONLY the next-next gate is crossed (skipping current target)? Should not credit the next-next if the current target wasn't credited.
- Crash classification inside the drain loop: does the dedupe check work correctly when multiple gates' struts are hit in one segment?
- Future-gate crash branch: if a future gate's strut is hit AND a current gate's opening is also crossed in the same segment, which wins? Is the ordering correct?
- `mark_timed_out` interactions with mark_collision: what if a tick fires both?

**`competition/vision_udp.py`** —
- `VisionUdpListener.start()` followed by `start()` again — does it leak the first transport?
- `latest_frame()` returns the LATEST decoded frame, but the receiver might have multiple complete frames in flight; is `pop_latest_frame` semantics right?
- Decode-on-demand: a 100Hz poll re-decodes the same JPEG every tick until a new frame arrives. Could be wasted CPU.
- `_VisionDatagramProtocol.error_received` increments `errors` for arbitrary OS errors; should that trigger a stream-restart?

**`competition/calibration.py`** —
- Sign-flipped test rejected only when k_t turns negative — but with sign-flipped synthetic data, k_t fits to a LARGE positive number (bias absorbed). Is the positivity guard adequate, or do we need an RMSE-based outlier check?
- `read_calibration_json` has no schema validation. A user-supplied calibration with garbage values gets accepted.

**`scripts/benchmark.py`** —
- The 8-minute timeout is enforced in `RacePipeline` via `time.monotonic()`, but the synthetic bench tracks `sim_time = step * dt` — does the bench ever exceed the 8-min cap? (Probably not at 30s default duration, but the contract is unclear.)
- Bench's `pass_through_margin=1.5` (synthetic) vs default `1.0` (PyBullet) — does this create per-platform DQ behaviour drift?

**`race_pipeline.py`** —
- `time.monotonic() - self._race_start_time` is wall-clock, NOT sim time. In a paused / slowed-down sim, this would falsely time out. Should the bench inject sim time instead?

**`competition/mavlink_bridge.py`** —
- Vision listener start ordering: opens AFTER MAVSDK connect, BEFORE offboard arm. If MAVSDK hangs at connect, vision socket never opens. Acceptable?
- `disconnect()` calls `await self._vision_listener.stop()`. If MAVSDK connection fails in `connect()`, was the listener ever started? Stop is idempotent but the error path may not run it.

## Output format (strict)
Write to `/Users/kenichi.matsuo/Personal/killallhumans-aigp-vq1/.loop/iter_002_review/<your_model_slug>.md`.

Same skeleton as iter-001's brief — see `.loop/specs/4_review_brief.md`.

## Hard constraints
- No `giga_chad_llm_*` calls.
- Read-only review — no source edits.
- Stay inside the worktree.
- Cite line numbers.
