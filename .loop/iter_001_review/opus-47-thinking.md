# Iter 001 Adversarial Review — Claude Opus 4.7 (max thinking)

## Summary
Iter-001 hit most of the headline items (sequencer DQ, AIGP defaults,
curvature-derived ILC sections, vision UDP, residual MLP, calibration
stub), and the new adversarial tests do prove that the prior known
issues are *symptomatically* gone — but several of the new artefacts
ship with physics/algorithmic mistakes that the new tests are too
narrow to catch. The standouts: (1) the calibration least-squares fits
the *wrong* relation (`a + g` should be `a − g`) and the unit test
generates synthetic data that matches the *code*, not the physics, so
the bug is invisible until real DCL telemetry arrives; (2) when a
drone segment crosses two gate planes in one tick the new
`enforce_in_order=True` DQ logic silently drops the second gate's
pass and may DQ on the wrong gate; (3) the `_point_in_gate_opening`
DQ check uses `pass_through_margin` which, for AIGP defaults at
margin=1.0, leaves a `[0.75, 1.35]m` lateral annulus per axis where a
future-gate strut hit is **neither DQ'd nor crashed** — it is silently
ignored. The vision UDP receiver is solid in isolation but never
reaches `MAVLinkBridge.get_camera_frame()` (already flagged by other
reviewers); I focus this review on the things the other seven agents
are unlikely to have caught.

## Findings (ordered by severity)

### F1. `DroneCalibrator` regression solves the wrong equation; the unit test is tautological — [BLOCKER]
- **File(s)**: `competition/calibration.py:93` (`y = a + self.gravity`),
  `competition/calibration.py:104` (`X = np.column_stack([-u, -v])`),
  `tests/test_calibration.py:36` (`a = -k_t * u - k_d * v - gravity`)
- **Issue**: In NED with gravity = +9.81 along +z (z-down), thrust opposes
  gravity (pushes −z), so the correct physics is
  `a_z = g − k_t·u − k_d·v_z`, equivalently `a_z − g = −k_t·u − k_d·v_z`.
  The code's regression has `y = a + g`, which differs from the correct
  `y = a − g` by `+2g`. There is no bias term in `X`, so the offset is
  absorbed into `[k_t, k_d]` with the wrong sign. At hover
  (`a ≈ 0`, `u ≈ g/k_t ≈ 0.45`, `v_z ≈ 0`) the fit literally inverts:
  `9.81 = k_t·(−0.45) → k_t ≈ −22`. `assumed_mass_kg=1.5` then writes
  `max_thrust_n = −33 N` to `drone_calibration.json`, and `read_calibration_json`
  has no positivity check. Any downstream controller that consumes this
  will silently get a negative thrust limit. The test
  `test_recovers_seeded_ratios_within_10_percent` only passes because
  `_synth_samples` generates data using **the same wrong equation** the
  code regresses against (line 36: `a = -k_t * u - k_d * v - gravity`,
  which gives `a + g = −k_t·u − k_d·v` — matches the code, not the
  drone). The test is tautological — it cannot catch this bug.
- **Repro**: Synthesize physically correct data:
  `a = gravity - k_t*u - k_d*v` (note plus-then-minus, not all-minus),
  then call `identify_thrust_drag_ratios`. Recovered `thrust_per_mass`
  will be ≈ `−k_t_true` (sign-flipped) with RMSE ≈ 2·g ≈ 19.6 m/s²
  — well above any reasonable acceptance band. Equivalently, push a
  hover-only sample set (u=0.45, v=0, a=0) and observe `k_t ≈ −22`.
- **Fix sketch**: Change `y = a + self.gravity` to `y = a - self.gravity`
  on line 93 (no other change needed — sign of X is consistent with
  the physics-correct form). Rewrite `tests/test_calibration.py::_synth_samples`
  to use `a = gravity - k_t * u - k_d * v` so the test exercises the
  physics, not the code. Add an assertion that recovered `thrust_per_mass`
  is **positive**. Also: gate `read_calibration_json` / `write_calibration_json`
  on `k_t > 0` and surface a `ValueError` (or warn) on negative fits;
  and reject calibrations where `|2g − RMSE_naive|` is the systematic
  bias signature of this exact bug.
- **Confidence**: high — the math is unambiguous and the test is
  literally generating data from the code's own equation. Independent
  cross-check: at any realistic hover throttle, the recovered `k_t`
  will be negative by inspection.

### F2. `enforce_in_order=True` DQ loop silently drops the second pass in a single-tick double-gate crossing — [MAJOR]
- **File(s)**: `gate_sequencing/sequencer.py:246-326` (single-pass-per-update
  on `_gates[self._current_idx]`), `gate_sequencing/sequencer.py:343`
  (`for future_gate in self._gates[self._current_idx + 1:]`)
- **Issue**: The current-target pass branch only credits **one** gate per
  `update()` call. After crediting `G_i`, `_current_idx` advances by
  1, then the DQ loop iterates `_gates[self._current_idx + 1:]`, which
  starts at `G_{i+2}` — `G_{i+1}` (the new current target) is NOT
  checked for either a pass credit or a DQ. So if the prev→curr
  segment legitimately crosses both `G_i`'s and `G_{i+1}`'s openings
  in one tick:
    * `G_i` gets credited.
    * `G_{i+1}`'s pass is silently lost (no entry in `_passed`).
    * `G_{i+1}`'s plane was crossed inside the opening, but is not
      DQ'd (because the DQ loop skipped it).
  Next tick, `_current_idx = i+1` (now points at `G_{i+1}`) and the
  drone is already past `G_{i+1}`'s plane, so `_plane_was_crossed`
  is False forever (unless the drone U-turns). The pass is never
  credited; the run will eventually DQ on `G_{i+2}` (a *different*
  gate than the actual skip) when the drone keeps flying forward.
  At competition replay this looks like a phantom skip.
- **Repro**: Two gates at `(5, 0, 1.5)` and `(6.5, 0, 1.5)` (1.5 m
  apart), drone at `(4.0, 0, 1.5)` at t=0, `(7.5, 0, 1.5)` at t=0.1
  (i.e., one tick at 10 Hz, speed = 35 m/s — implausible at race
  pace but trivially reachable during a `dt=0.05` synthetic test or
  a stutter). `seq.passed_gate_ids` will contain only the first gate;
  `seq.is_disqualified` is False until the drone moves another step
  and crosses `G_2`'s plane.
- **Fix sketch**: Wrap the current-target pass branch in a `while`
  loop that re-runs while `_plane_was_crossed(prev, pos, _gates[_current_idx])`
  is True and the segment hasn't been entirely consumed — credit
  every gate the segment passes. Alternatively, after crediting,
  re-include `_gates[_current_idx]` in the DQ scan (i.e., iterate
  `_gates[_credited_this_tick + 1:]` from the **original**
  `_current_idx + 1`). Add a regression test
  `test_segment_passing_two_gates_credits_both`.
- **Confidence**: high — single read of the update() control flow
  confirms the new DQ loop iterates after `_current_idx` advances.
  No existing test in `gate_sequencing/tests/test_sequencer.py` or
  `gate_sequencing/tests/test_sequencer_adversarial.py` exercises a
  multi-gate-per-tick segment.

### F3. Future-gate strut hits with AIGP defaults are NEITHER DQ'd NOR crashed (silent skip annulus) — [MAJOR]
- **File(s)**: `gate_sequencing/sequencer.py:351`
  (`if self._point_in_gate_opening(crossing, future_gate)` — DQ
  branch checks **only the opening**), `gate_sequencing/sequencer.py:285-307`
  (crash branch — applies **only to current gate**, not future gates),
  `competition/aigp_geometry.py` (interior 1.5 m, border 0.6 m)
- **Issue**: The DQ branch uses `_point_in_gate_opening`, which is
  `interior_half × pass_through_margin`. For AIGP defaults (interior
  1.5 m, border 0.6 m, outer 2.7 m) with the **default**
  `pass_through_margin=1.0`, the DQ test fires for `|y| < 0.75 m`
  and `|z − 1.5| < 0.75 m`. The crash branch only runs against
  `self.current_gate`, not against future gates. So a drone clipping
  a future gate's **strut** in the annulus `0.75 m ≤ |y| ≤ 1.35 m`
  (or the equivalent z annulus) on a future gate:
    * does NOT trip the future-gate DQ (outside the lenient opening
      check),
    * does NOT trip the crash branch (which only checks current gate),
    * is recorded as a `miss` if anything (via the `not crash_classified
      and plane_crossed` branch — but that branch only fires for
      `self._gates[self._current_idx]`, so even `miss` doesn't fire
      for future gates).
  Net: a future-gate frame strike inside `[opening, outer_frame]` is
  **silently ignored**. The drone keeps flying and may DQ on a yet-
  further gate, mis-attributing the violation. The brief explicitly
  asked about this regime; current code does not handle it.
- **Repro**: AIGP geometry; drone at `(0,0,1.5)` t=0; segment to
  `(11.5, 1.0, 1.5)` while gates are at `(5,0,1.5)` (current) and
  `(10,0,1.5)` (future). The segment hits future-gate plane at
  `y ≈ 1.0` (inside outer 1.35, outside opening 0.75). Neither
  `seq.crashed_gate_ids` nor `seq.is_disqualified` records it.
- **Fix sketch**: Symmetrise the geometry classification: in the DQ
  loop at line 343, also classify a future-gate crossing inside the
  outer-frame-but-outside-opening as a `crash` (with `gate_id` set to
  the future gate). Or simpler: change the DQ branch to test
  `_point_in_outer_frame(crossing, future_gate)` (any frame contact
  is terminal whether opening or strut). Both close the silent annulus.
  Add a test `test_future_gate_strut_hit_is_crash_or_dq` parameterised
  by `pass_through_margin ∈ {1.0, 1.5}`.
- **Confidence**: high — confirmed by reading both branches and the
  geometry math. The AIGP outer_half = 1.35 m vs opening_half = 0.75 m
  difference of 0.6 m on each axis is exactly the border width.

### F4. `CameraFrame` dataclass default is still 640×480, not 640×360 — AIGP spec drift in the adapter surface — [MAJOR]
- **File(s)**: `competition/adapter.py:145-146`
  (`width: int = 640`, `height: int = 480`)
- **Issue**: VADR-TS-002 mandates 640×360. `CameraIntrinsics()` in
  `estimation/gate_pnp.py` was correctly defaulted to 360 in iter-001,
  but the `CameraFrame` dataclass — which is the *actual transport
  type* returned from `get_camera_frame()` everywhere — still defaults
  to 480. `competition/pybullet_adapter.py:145` constructs a
  `CameraFrame` with explicit width/height from the actual rendered
  image so it's fine *in this code path*. But any future caller that
  constructs `CameraFrame(timestamp_us=..., image=arr)` (the dataclass
  permits omitting width/height) gets the wrong shape, and any
  downstream consumer that uses `.height` as a fallback when the
  image array is unavailable (e.g. in synthetic test fixtures or in
  the `VisionUdpReceiver` decode helper at `competition/vision_udp.py:283`,
  which DOES pass explicit `width/height=img.shape` so it's fine, but
  any other consumer is not) silently uses the wrong cy.
- **Repro**: `CameraFrame(timestamp_us=0, image=np.zeros((360, 640, 3),
  dtype=np.uint8))` returns `.height = 480` — a contradiction with the
  image shape. No test asserts the dataclass default.
- **Fix sketch**: Change `height: int = 360` on line 146. Add a
  one-line assertion in `decode_jpeg_to_camera_frame` that
  `frame.height == img.shape[0]`. Add a test that
  `CameraFrame(timestamp_us=0, image=np.zeros((1,1,3))).width == 640
  and .height == 360`.
- **Confidence**: high — this is a single-line dataclass default. It
  is distinct from F11 in `composer-25-5.md` which catches the same
  symptom on `GatePnPEstimator()` defaults; the `CameraFrame`
  dataclass default is a separate problem.

### F5. ILC `derive_section_boundaries` uses an absolute `threshold ≤ 1e-6` fallback that misclassifies bimodal acceleration profiles — [MAJOR]
- **File(s)**: `planning/ilc_sections.py:104` (`accels.max() <= 1e-6`),
  `planning/ilc_sections.py:108` (`if threshold <= 1e-6: return [single low]`)
- **Issue**: The fallback "everything is one low section" fires whenever
  the 60th-percentile acceleration is ≤ 1e-6 m/s². For a track with
  ~60% straight-line low-accel cruise and ~40% sharp turns, the
  quantile = 0 even though the upper 40% are *clearly* high. The
  current logic then drops the entire high-curvature partition. The
  threshold is **absolute** rather than **relative** to `accels.max()`;
  a smooth long-cruise track with one corner is misclassified as
  "all low" even if that corner is 20 m/s². The test
  `test_alternating_curvature_segments` passes because the alternating
  pattern keeps the 60th-percentile well above 1e-6, but real race
  trajectories where straight cruise dominates time will trip this.
- **Repro**: Construct trajectory points with `acceleration=(0,0,0)`
  for `i in range(120)` and `acceleration=(20,0,0)` for `i in
  range(120, 200)`. `np.quantile(accels, 0.6) = 0` →
  `threshold ≤ 1e-6` → returns single low section. The 40% high
  curvature is invisible to ILC.
- **Fix sketch**: Use a relative threshold:
  `threshold = max(np.quantile(accels, quantile), 0.05 * accels.max())`
  or `threshold = max(threshold, 1e-3 * accels.max())`. Add a test
  `test_bimodal_low_dominant_high_minority_still_partitions` that
  uses the 120/80 ratio above.
- **Confidence**: high — single quantile/threshold check, easy
  to verify with the literal numbers above.

### F6. `_pt_to_step` integer truncation silently drops short sections when trajectory point spacing is finer than `dt` — [MINOR]
- **File(s)**: `planning/ilc_sections.py:125-128` (`int(points[i].time / dt)`),
  `planning/ilc_sections.py:134` (`if e_step <= s_step: continue`)
- **Issue**: When trajectory sample spacing `dt_pt` < step `dt`,
  multiple consecutive points map to the same step index. A
  point-space run `[2, 3)` with `dt_pt=0.005, dt=0.01` maps to
  `s_step = int(0.010/0.01) = 1, e_step = int(0.015/0.01) = 1` →
  dropped silently at line 134. Once `_merge_below_min` runs, the
  small run would have been merged into a neighbour anyway, so for
  current values (`min_steps = max(int(0.2/0.01), 4) = 20`) the loss
  is bounded; but if iter-002 tightens to a higher trajectory
  resolution (e.g. for the Hessian-derived planner) without
  re-checking, partition information will leak.
- **Repro**: `derive_section_boundaries(points=[_StubPoint(time=i*0.005,
  acceleration=(20,0,0)) for i in range(30)] + [_StubPoint(time=0.150
  +i*0.005, acceleration=(0,0,0)) for i in range(30)], dt=0.01,
  n_total_steps=30, n_max=4, min_steps=1)` — count returned sections;
  some will be lost on the boundary.
- **Fix sketch**: Round instead of truncate
  (`int(points[i].time / dt + 0.5)`), or use a half-open ceiling
  for the end index. Document that `dt_pt ≥ dt` is required and
  add an assertion. Add an `assert dt > 0` and an explicit guard
  for `dt_pt < dt` cases (raise or warn).
- **Confidence**: medium — partition is "good enough" for current
  bench (race_01 uses overrides anyway), but the silent drop is
  a footgun for iter-002 trajectory-resolution tuning.

### F7. AIGP placeholder track `aigp_default.json` is NOT a 1.25× scaled `race_01`; topology was simplified — [MINOR]
- **File(s)**: `sim_pybullet/configs/aigp_default.json` (6 gates,
  simple curves), `sim_pybullet/configs/race_01.json` (12 gates,
  helix segment), `.loop/synthesis/iter_001.md` (A18: "rough scaling
  of race_01 by 1.25×")
- **Issue**: The synthesis claims `aigp_default.json` is a 1.25× scaled
  `race_01` to match the 1.5 m / 1.2 m opening ratio. Inspection shows
  it has 6 gates (race_01 has 12) and *omits* the helix segment
  (race_01 gates 7-12 with 1 m vertical spacing). The geometric
  primitives (gate sizes) are scaled, but the **topology** is
  simplified to flat curves. So iter-002 ILC regression against
  `aigp_default` would *systematically under-test vertical maneuvers*
  while still claiming "the placeholder matches race_01 shape." The
  `placeholder: true` flag is set, so this isn't strictly a contract
  violation, but the comment in the synthesis is misleading and any
  multi-track CI matrix (A17) needs a vertically-stacked AIGP-class
  track too.
- **Repro**: `jq '.gates | length' sim_pybullet/configs/aigp_default.json`
  → 6; `jq '.gates | length' sim_pybullet/configs/race_01.json` → 12.
  No vertical spacing > 0.6 m in `aigp_default`.
- **Fix sketch**: Either generate a *true* scaled clone of race_01
  (preserve helix), or add a second placeholder
  `aigp_helix.json` with an AIGP-scaled helix segment so the
  multi-track matrix exercises both planar and 3D maneuvers. Update
  the synthesis comment to "topology-simplified, gates only" rather
  than "scaled."
- **Confidence**: high — direct file diff.

### F8. New ILC `vel_scale = 0.5` "high" default is itself a course-coupled magic number — [MINOR]
- **File(s)**: `config/ilc_defaults.json:9` (`vel_scale: 0.5` for high),
  `sim_pybullet/configs/race_01.json:21-26` (race_01's `ilc_section_overrides`
  use `vel_scale ∈ {0.4, 0.5, 0.7}` for three different high-curvature
  sections)
- **Issue**: The synthesis (A8/A9) says "pull current race_01 values as
  the 'high-curvature' default." But race_01 has **three distinct**
  high-curvature `vel_scale` values (0.4, 0.5, 0.7), each tuned to a
  specific helix loop. Picking the median (0.5) and shipping it as the
  *global default* trades course-specific magic for "averaged course-
  specific magic." For an AIGP track with longer straights and tighter
  hairpins, 0.5 may under- or over-slow the drone. The charter says
  "no course-specific magic numbers"; the new default doesn't
  technically come from a *named course*, but it's the median of a
  single course's tuned values — same anti-pattern, different
  packaging.
- **Repro**: `grep '"vel_scale"' sim_pybullet/configs/race_01.json` →
  three values (0.4, 0.5, 0.7). `cat config/ilc_defaults.json | jq
  '.sections.high.vel_scale'` → 0.5 (the median).
- **Fix sketch**: Derive `vel_scale` from local curvature **per
  section** instead of a global constant — e.g.,
  `vel_scale = clip(1.0 / (1.0 + 0.5·κ_max), 0.3, 0.9)`. Or, ship
  three named curvature bins (mild/moderate/sharp) instead of two,
  with formulae documented. Until then, mark this as legacy-defaults
  in the comment and add the formula derivation to
  `.loop/synthesis/iter_001.md`.
- **Confidence**: medium — depends on whether the synthesis intended
  "median of race_01" or "neutral defaults." I read it as the
  former; the JSON commit message ("Pull race_01 tuned high values
  as the curvature high defaults") confirms it's median-of-race_01.

### F9. `VisionUdpReceiver` accepts `payload_size == 0` and never validates `jpeg_size == sum(chunk sizes)` after assembly — [MINOR]
- **File(s)**: `competition/vision_udp.py:90-95`
  (`parse_packet`: only validates `pay_sz == len(payload)`, not
  `pay_sz > 0`), `competition/vision_udp.py:220` (`_ReassemblyBuf.assemble`:
  `out = bytearray(jpeg_size)` pre-sized, fill by slice assignment)
- **Issue 9a (zero-size chunk)**: `parse_packet` raises
  `ValueError` only if `pay_sz != len(payload)`. If `pay_sz = 0` and
  the chunk's `total_chunks > 0`, the chunk slot
  `_chunks[cid] = b""` is silently empty. The assembled output has
  unwritten zeros at that chunk's offset region (assumed offset 0 for
  a single chunk). `cv2.imdecode` returns `None` for the all-zero
  buffer; the receiver pops a frame with `image=None` and the caller
  silently drops it.
- **Issue 9b (size mismatch)**: After all chunks arrive,
  `_ReassemblyBuf.assemble` does `out[offset:offset+len(chunk)] = chunk`
  for each chunk; `offset += len(chunk)`. There is no check that
  `sum(len(chunk) for chunk in chunks.values()) == jpeg_size`. If
  per-chunk `payload_size` totals < jpeg_size, the tail of `out`
  is zero-filled (corrupted JPEG); if totals > jpeg_size, the
  bytearray's slice-extend semantics grow `out` beyond `jpeg_size`
  (also corrupted). Both produce silently-broken `cv2.imdecode` calls.
  Other reviewers flagged the size-mismatch case; the
  `payload_size=0` case is independent and additional.
- **Repro 9a**: Build a packet with `jpeg_size=10, payload_size=0,
  total_chunks=1, chunk_id=0`, feed it to `VisionUdpReceiver`. The
  buffer reports `is_ready=True`; `assemble()` returns a 10-byte
  zero buffer. `cv2.imdecode` on it returns `None`.
- **Repro 9b**: Three chunks with payloads `b"abc", b"def", b"g"`
  (total 7 bytes) but `jpeg_size=10`. `assemble()` returns
  `b"abcdefg\x00\x00\x00"` (3-byte zero tail). No error raised.
- **Fix sketch**: In `parse_packet`, reject `payload_size == 0` with
  `ValueError("zero-size chunk")`. In `assemble()` (or in
  `is_ready`), assert `sum_payload == jpeg_size`; raise or return
  `None` so the receiver drops the frame instead of yielding a
  corrupt JPEG. Add tests covering both cases.
- **Confidence**: high — direct read of `parse_packet` and
  `_ReassemblyBuf.assemble`.

### F10. `_delivered_ids` deduplication uses `O(N)` list-membership; trivial nit but `set` is cleaner — [NIT]
- **File(s)**: `competition/vision_udp.py:172-176`
  (`self._delivered_ids: List[int]`, `if pkt.frame_id in self._delivered_ids`)
- **Issue**: `_delivered_ids` is bounded (cap = `max_buffered_frames * 8 = 64`)
  but `list.__contains__` is `O(N)` per packet. At 30 fps with ~16
  chunks/frame that's ~500 lookups/s × 64 = 32 k ops/s — negligible.
  But using a `set` is `O(1)` and removes a foot-gun if the cap is
  raised.
- **Fix sketch**: Replace `List[int]` with a `collections.deque`
  + accompanying `set` (deque for FIFO eviction, set for membership).
- **Confidence**: high — purely a perf/clarity nit.

### F11. `SimplePositionTracker` silently ignores `use_residual` — silent footgun if `use_geometric_tracker=False` toggles — [NIT]
- **File(s)**: `control/mpc_tracker.py:268-334` (no residual branch),
  `control/mpc_tracker.py` `TrackerConfig.use_residual` (consumed only
  in `GeometricTracker`)
- **Issue**: If anyone constructs `SimplePositionTracker(TrackerConfig(use_residual=True))`,
  the flag is silently ignored — no warning, no assertion. The
  iter-001 bench always uses `GeometricTracker`, so this is hidden.
  Other reviewers flagged this (composer-25-3 F13, composer-25-4 F17,
  composer-25-5 F11) — I include it for cross-confirmation and to
  note that `scripts/visual_demo.py:406` *does* permit
  `SimplePositionTracker` via CLI, which means a user toggling
  `--no-geometric` with `use_residual=True` in config gets silent
  no-op residual. Suggest: raise in `TrackerConfig.__post_init__`
  when `use_residual=True` is set but the tracker class can't honor it.
- **Confidence**: high.

### F12. Adversarial sequencer test docstrings/asserts no longer match new DQ semantics — semantic drift, misleading test names — [NIT]
- **File(s)**: `gate_sequencing/tests/test_sequencer.py::test_crossing_non_highlighted_gate_does_not_credit`
- **Issue**: With `enforce_in_order=True` as the default, "crossing a
  non-highlighted gate" now DQs the run, not just "does not credit."
  The test still passes (its assertions only check that no credit was
  given), but the test name and docstring make a *weaker* claim than
  the actual current behavior. A future developer reading the test
  could conclude "skip-then-correct is allowed" when in fact it now
  DQs. Same drift affects `test_skip_then_correct_path` and
  `test_u_turn_after_skip_does_not_recover` if they construct
  `GateSequencer(gates)` without explicit `enforce_in_order=False`.
- **Repro**: Read the test code; assertions are `assert pass_count == 0`
  but the new code now also sets `RaceState.DISQUALIFIED`. The test
  is not asserting `not seq.is_disqualified`.
- **Fix sketch**: Either rename to make the new DQ semantics explicit
  (`test_crossing_non_highlighted_gate_dqs`), or pass
  `SequencerConfig(enforce_in_order=False)` to keep the old "no
  credit but no DQ" semantics under test and add a parallel
  enforce-in-order variant.
- **Confidence**: high — direct read of test file.

### F13. `CameraIntrinsics.pitch_offset_rad` is stored but never applied to the camera→world rotation; the horizon test verifies a *formula*, not the *code path* — [NIT]
- **File(s)**: `estimation/gate_pnp.py:96` (`pitch_offset_rad: float`),
  `estimation/gate_pnp.py::gate_pose_to_drone_position` (does not
  apply `pitch_offset_rad`), `tests/test_camera_geometry.py::test_horizon_projects_below_image_center_with_upward_tilt`
- **Issue**: The 20° upward tilt is stored on `CameraIntrinsics` but
  the only place it would matter — `gate_pose_to_drone_position` —
  ignores it. For position-only PnP (camera frame → drone NED with
  coincident optical center) the tilt is geometrically irrelevant
  to the *position* output, but any code that derives orientation
  (yaw seed, FOV-aware visibility check, lever-arm correction) needs
  it. The "horizon test" computes the predicted pixel for a horizon
  point analytically and asserts it lands below `cy` — it verifies
  the *trigonometry*, not that anywhere in `gate_pnp.py` actually
  consumes `pitch_offset_rad`. Other reviewers also caught the
  storage-without-use; my addition is that the test gives a *false
  sense* that the tilt is threaded through the pipeline.
- **Fix sketch**: Add a `R_cam2drone = Rot_y(-pitch_offset_rad)`
  application in `gate_pose_to_drone_position`'s yaw extraction path
  (the position path is unaffected if the optical center is at the
  drone origin, so document that explicitly). Add a test
  `test_pitch_offset_actually_rotates_gate_yaw_estimate` that
  constructs a gate at known yaw and asserts the recovered yaw
  changes by 20° when `pitch_offset_rad` changes by 20° (i.e., the
  test exercises the *code*, not the analytic horizon formula).
- **Confidence**: medium — depends on what downstream uses the
  rotation. Position-only PnP is correct; orientation handoffs are
  not.

### F14. Synthetic bench's `pass_through_margin=1.5` conflicts with `race_01`'s 1.2 m × 0.18 m geometry — false DQ region on future gates — [MINOR]
- **File(s)**: `scripts/benchmark.py::run_synthetic_benchmark`
  (`SequencerConfig(pass_through_margin=1.5, ...)`),
  `sim_pybullet/configs/race_01.json` (`gate_defaults`: interior 1.2,
  border 0.18 → outer_half = 0.78 m)
- **Issue**: For race_01 geometry, `opening_half = 0.6 m * 1.5 = 0.9 m`,
  `outer_half = 0.78 m`. The "opening" check is now **larger** than
  the outer frame. The DQ check at sequencer.py:351 uses
  `_point_in_gate_opening` (= 0.9 m), so a drone passing through the
  plane of a future race_01 gate at lateral `|y| ∈ [0.78, 0.9]`
  (*outside* the physical frame entirely — i.e., it would have
  missed the gate completely in real geometry) gets DQ'd because
  the lenient opening test fires. This is a synthetic-bench-only
  artefact (PyBullet bench uses margin=1.0) and only affects future
  gates, but it produces false-positive DQs in the synthetic bench
  that don't correspond to physical violations. Combined with F3
  (silent annulus on AIGP defaults), the DQ behaviour is *config-
  geometry-coupled* in a non-obvious way.
- **Repro**: `run_synthetic_benchmark` on race_01 with a trajectory
  that grazes a future gate's plane at lateral 0.85 m. Expected:
  physical miss (no frame contact). Actual: DQ.
- **Fix sketch**: Decouple the DQ opening check from
  `pass_through_margin`. Use a fixed strict opening (margin=1.0) for
  the DQ test, regardless of the pass-through lenience. Then the DQ
  fires only when the drone actually traversed the gate opening, and
  the lenient pass-through margin only affects how leniently we
  *credit* a pass on the current gate.
- **Confidence**: high — math is direct (1.5 * 0.6 = 0.9 > 0.78).

### F15. `AIGP_VQ1_MAX_RUN_DURATION_S` (480 s) is defined but neither bench nor sequencer enforces it; bench's 30 s cap is unrelated — [NIT]
- **File(s)**: `competition/aigp_geometry.py` (`AIGP_VQ1_MAX_RUN_DURATION_S = 480.0`),
  `scripts/benchmark.py` (`THRESHOLDS["max_total_time_s"] = 30.0`),
  `gate_sequencing/sequencer.py` (no `TIMED_OUT` transition)
- **Issue**: Other reviewers already flagged the 8-min spec gap; my
  addition: even the `RaceState.TIMED_OUT` enum value is *defined*
  but **never assigned** anywhere in `sequencer.py`. The state is
  dead code. So even if a caller wanted to detect "exceeded 8 min,"
  there is no mechanism. The bench's 30 s threshold is unrelated to
  the spec.
- **Fix sketch**: Add `start_time_s` to `SequencerConfig`; transition
  to `TIMED_OUT` when `current_time - start_time > AIGP_VQ1_MAX_RUN_DURATION_S`.
  Wire `start_time_s` to bench/runner `t=0`. Add a regression test
  `test_sequencer_times_out_at_480s`.
- **Confidence**: high — `grep TIMED_OUT gate_sequencing/sequencer.py`
  finds only the enum definition; no assignment.

## Things iter-001 got right
- **Sequencer in-order DQ (when the segment crosses ≤ 1 gate per
  tick)**: the explicit opening-check approach correctly fires on
  the U-turn / skip-back pattern documented in I-1, and the
  adversarial tests in `gate_sequencing/tests/test_sequencer_adversarial.py`
  do cover the common single-skip case end-to-end.
- **AIGP geometry centralisation in `competition/aigp_geometry.py`**: a
  single import lets the rest of the codebase converge on the spec'd
  values. The constants are clearly named (`AIGP_GATE_INTERIOR_M`,
  `AIGP_DRONE_WIDTH_M`, etc.) and the test
  `test_camera_geometry.py` asserts the defaults flow through to
  `CameraIntrinsics()`.
- **Residual MLP safety composition**: clamp at the model (±0.05 rad
  on tilt), then add, then re-clamp to `max_tilt_rad` at the
  consumer — order is correct in `GeometricTracker`, and the test
  `test_disabled_path_is_byte_identical` and
  `test_zero_init_residual_is_byte_identical` provide strong
  byte-equality guarantees against accidental degradation.
- **Vision UDP reassembler unit tests**: 15 tests in `tests/test_vision_udp.py`
  cover OoO, partial-GC, duplicates, eviction — a thorough
  *unit*-level pass on the synchronous reassembler. The remaining
  integration gap (MAVLink bridge not wired) is well-flagged.
- **ILC section JSON config (config/ilc_defaults.json)**: cleanly
  removes the literal `[0,200],[200,440],...` step boundaries from
  `benchmark.py`. The race_01 override mechanism gives a clean
  escape hatch for legacy tuning.

## What I did NOT review
- `competition/mavlink_bridge.py` send_* / offboard / telemetry loops
  beyond the `get_camera_frame` stub (other reviewers covered the
  vision-wiring gap exhaustively).
- `competition/pybullet_adapter.py` IMU emulation correctness vs the
  PyBullet rigid-body integrator.
- `planning/dynamic_replanner.py` cooldown semantics and the
  interaction with the new sequencer DQ logic during a recovery
  reroute (other reviewers covered the false-DQ-on-replan concern).
- `control/tests/test_tracker.py` cross-validation between
  `SimplePositionTracker` and `GeometricTracker` — only spot-checked.
- The proto/MAVSDK version skew and any Windows DCL binary behaviour
  (not in worktree).
- Actually running the 310-test pytest matrix or a full bench — this
  is a read-only static review.
