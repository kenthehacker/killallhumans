# VQ2 Gate 1 recovery status — 2026-07-23

## User-authorized bounded live exception

Later on 2026-07-23, the user explicitly authorized rapid powered simulator
iteration. That instruction admits only the bounded 0.60-second
`gate1-recenter` diagnostic with frozen no-passage bounds and the normal
powered lifecycle. It does not accept M1/M2, pixel-rate authority, Gate 1
passage, or `full-lap`; those claims still require their own evidence.

## Disposition

The recovery is checkpointed as an **offline-only candidate**. It is not an
accepted or powered Gate 1 capability. Both `full-lap` and `gate1-recenter`
are absent from the fast powered wrapper, normal runner CLI, and powered
dispatcher. A direct programmatic `run_live(...)` request for either identity
also rejects before live transport imports, adapter construction, replay
setup, or simulator contact.

No recovery command launched or connected to FlightSim, ran preflight, took
the live lease, reset, armed/disarmed, or sent a flight target. A later
read-only process listing showed an already-running FlightSim process; it was
left untouched.

## Durable separation

- Original base: `5545a60de268588f3de1136bededa99f4f445a99`.
- Exact unaccepted tracked snapshot:
  `50eb038179e9c2ebb259219922651fda89e14d5c`, retained by branch
  `checkpoint/vq2-20260722-unaccepted`.
- Isolated recovery worktree:
  `C:\Users\John\killallhumans-gate1-recovery`, branch
  `recovery/vq2-gate1-recenter-20260723`.
- Quarantine checkpoint: `9b12f56`, which removes full-lap powered admission,
  restores the live-proved Gate 0 horizontal sign, restores exact-zero
  transition authority, and removes rejected aggressive authority from the
  admitted path.

The original dirty worktree was not used for recovery edits.

## Offline candidate

The non-dispatched candidate freezes the handoff's provisional law and bounds:

- `clamp(0.12*x_error + 0.025*x_rate, -0.05, +0.05)` rad;
- exact-zero pitch objective and zero yaw rate;
- roll/pitch command rates at most `0.12 rad/s`;
- fixed thrust `0.275`, checked inside `0.21..0.30`;
- a hard `0.60 s` authority window with exact wire-start receipts and a
  reserved cleanup-command slot;
- strict same-generation, fresh-primary, frame/provenance, race, attitude,
  body-rate, watchdog, and contact checks;
- abort after more than 24 pixels of divergence following three fresh control
  frames;
- a three-fresh-frame `abs(normalized_x) <= 0.35` corridor hold, followed by a
  paced final authority and convergence recheck.

The result records entry/final error, least-squares error slope, fresh control
frames, attitude and command extrema, target area/width, maximum authoritative
gate index, safety classification, and cleanup state. It deliberately keeps
`success = false`: the offline seam can establish only
`recenter_criteria_met`; powered lifecycle cleanup has not been integrated or
proved.

## Authoritative blockers

The 2026-07-18 and 2026-07-20 handoffs conflict with immediately powering the
2026-07-22 position-plus-pixel-rate candidate:

1. The accepted Package 3B work is explicitly only a passive receiver tranche;
   its task record says Package 2 and M1 remain incomplete. Powered causal
   response/timing evidence is absent.
2. The private evidence root contains five `.vq2replay` bundles, all from
   passive Gate 0 preflight timing work. One is invalid/incomplete and none
   contains the Gate 1 transition, approved labels, or tracker-isolation
   acceptance needed for M2.
3. The archived fast-cycle `session.jsonl.gz` files are processed telemetry,
   detections, and tracker output, not a rerunnable approved pixel corpus with
   a frozen final processor and isolation wrapper.
4. No reviewed close-geometry threshold proves the separate no-passage
   requirement. Race-index change is an abort, but its packet latency alone is
   not a geometric stop boundary.

Consequently the required fixed-variable sign comparison cannot be accepted.
As a descriptive diagnostic only, the latest rejected full-lap trace
`20260723T051544Z-full-lap-482c66f5` (runner source
`1ce57e428e738f5dcf4120d49d4e5e0577f83a72124ad00456633a142afeb05e`,
trace
`a7696040539e17aff3436560e9a2631dfcfa371f0e1ad9e000b352fb19a32ba9`)
has 18 distinct primary frames through the first 0.562 seconds. The bounded
positive-sign law is positive on all 18 and saturated at `+0.05` on 17, while
the rejected recorded controller sent a negative roll rate on 17 and a
positive one on 1. Horizontal error grew from 198 to 203 pixels with a
`+4.524 px/s` least-squares slope. This supports investigating the positive
sign; it is not a controlled opposite-sign replay or acceptance evidence.

## Exact-tree non-live verification

- Focused runner: `328 passed`.
- Focused fast-cycle wrapper: `10 passed`.
- `test-vq2`: `2473 passed, 1 skipped`.
- `test-fast`: `3611 passed, 21 skipped, 42 deselected`.
- `test-unit`: `3611 passed, 21 skipped, 42 deselected`.
- `git diff --check`: clean.
- Independent safety review found no remaining reproducible defect in the
  offline delta and separately cleared the early live-stage rejection guard.

No `live`, slow, benchmark, promotion, or powered test was run.

## Resume boundary

Before any `gate1-recenter` powered admission:

1. finish and accept M1 timing/simulator evidence;
2. provide and accept a Gate 1 recorded replay plus tracker-isolation result;
3. review and freeze a conservative close-geometry no-passage bound;
4. rerun independent lifecycle/safety review on the integrated stage,
   including cleanup confirmation;
5. freeze one exact source and then run recenter-only repeats under the
   existing scoped authorization.

Do not design Gate 1 passage, restore full-lap authority, or claim Gate 1,
Gate 2, lap, or speed progress at this checkpoint.
