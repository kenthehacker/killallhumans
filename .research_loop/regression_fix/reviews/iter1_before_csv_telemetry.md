# Red-Team Review — Iteration 1 (BEFORE edit)

## Plan Summary
Add 28-column CSV telemetry logging at control-loop frequency to `scripts/visual_demo.py`, plus a `--no-render` headless flag.

## Failure Modes Analyzed

### 1. File I/O stalls the control loop — Severity: LOW
- **Risk**: csv.writer with default buffering could occasionally stall on disk flush.
- **Mitigation**: Python csv.writer uses the file object's buffering (8 KB default). At ~28 float columns × ~20 bytes each ≈ 560 bytes/row, the buffer holds ~14 rows before flushing. At 48 Hz, flushes happen ~3.4×/s. Negligible impact.
- **Verdict**: Acceptable.

### 2. File not closed on crash/exception — Severity: LOW
- **Risk**: If PyBullet crashes or Python segfaults, CSV may lose last few rows.
- **Mitigation**: Flush every N rows (e.g., every 240 rows = 5s). Also wrap close in finally block.
- **Verdict**: Acceptable with periodic flush.

### 3. `--no-render` breaks termination — Severity: MEDIUM
- **Risk**: Without cv2.waitKey, the 'q' key quit won't work. If the sim hangs, there's no way to stop it.
- **Mitigation**: `--max-time` flag already exists and will break the loop. Also SIGINT (Ctrl+C) works. In `--no-render` mode, we skip the key-check block entirely — the loop terminates on max_time, crash, or race-complete.
- **Verdict**: Acceptable. Document that `--no-render` requires `--max-time`.

### 4. CSV column mismatch with future parsers — Severity: LOW
- **Risk**: If column order or count differs from what iter00_root_cause.md specifies, downstream iterations will break.
- **Mitigation**: Use exact column list from the spec. Write a header row. Verify column count after implementation.
- **Verdict**: Acceptable.

### 5. Accidental behavior change — Severity: LOW
- **Risk**: Adding logging code accidentally modifies control flow (e.g., variable shadowing, early return).
- **Mitigation**: The logging code is strictly append-only: after all physics/control logic, before render. No existing variables are modified.
- **Verdict**: Acceptable.

## Overall Assessment: **PASS** — No high-severity issues. Proceed with implementation.
