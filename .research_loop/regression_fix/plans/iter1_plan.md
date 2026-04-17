# Iteration 1 Plan — Add CSV Telemetry Logging to visual_demo.py

## Bottleneck
`telemetry_logging_missing` — visual_demo.py has no CSV logging. All downstream debugging (gate-by-gate error decomposition, attitude saturation detection, target-jump analysis) depends on high-frequency telemetry. Existing CSV files in `logs/` are from April 3 benchmark runs, not visual_demo.

## Exact Changes

### File: `scripts/visual_demo.py`

#### 1. Add imports (top of file, after existing imports)
- `import csv`
- `import datetime`

#### 2. Add `--no-render` CLI flag (argparse section, ~line 570)
- New arg: `--no-render`, action="store_true", help="Disable visualization (headless mode)"
- Pass through to `VisualDemo.__init__` as `no_render` parameter

#### 3. CSV logger initialization (`__init__`, after line ~297)
- Create `logs/` dir if not exists
- Open CSV file: `logs/visual_demo_{timestamp}.csv`
- Write header row with 28 columns:
  ```
  sim_time, step_count, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
  ref_pos_x, ref_pos_y, ref_pos_z, ref_vel_x, ref_vel_y, ref_vel_z,
  target_pos_x, target_pos_y, target_pos_z, target_vel_x, target_vel_y, target_vel_z,
  tracking_error_m, current_gate_id, gates_passed, target_source, loop_dt_ms
  ```
- Store file handle and csv.writer as instance vars

#### 4. Per-step logging (`run()` method, after step 8 timing, before step 9 render)
- Determine `target_source`: "trajectory" or "gate_fallback"
- Write one row per physics step with all 28 fields
- This runs at full sim frequency (~240 Hz for CF2X at ctrl_freq=48 with sim substeps)

#### 5. Close CSV on exit (`run()`, after while-loop)
- Close the file handle
- Print path to CSV

#### 6. Skip rendering in `--no-render` mode
- In the render block (step 9), skip cv2.imshow and cv2.waitKey when no_render=True
- Still compute ref for telemetry
- Skip cv2.destroyAllWindows if no_render

## Expected Behavior
- Every visual_demo run writes a timestamped CSV to `logs/`
- CSV has one row per control loop iteration (~48 Hz for 48 Hz ctrl_freq)
- `--no-render` allows headless automated runs
- No change to drone behavior, trajectory, or control

## Success Criterion
- CSV file is created in `logs/` with correct 28-column header
- File has >100 rows for a 90s run
- `--no-render` flag works without errors
- No regression: same gates_passed (4), same crash behavior

## Rollback Criterion
- CSV writing causes crash or exception
- Gates passed decreases from 4
- avg_tracking_error increases >10%

## Risks
- File I/O overhead could slow the control loop — mitigated by using csv.writer (buffered)
- If sim crashes before CSV close, data could be lost — mitigated by flushing periodically
