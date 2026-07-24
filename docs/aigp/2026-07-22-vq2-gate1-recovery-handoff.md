# AIGP VQ2 Gate 1 recovery handoff — 2026-07-22

> This handoff records the result of the long 2026-07-22 powered-development
> thread after it was intentionally stopped at 22:28:50 PDT. It is a recovery
> document, not evidence that the current dirty candidate is accepted.
>
> For build-3385 interface facts and the live safety contract,
> `2026-07-18-vq2-handoff.md` remains authoritative. In particular, its next
> live milestone is a bounded Gate 1 recenter, not a full lap. If this document
> conflicts with that safety contract, stop and obtain human review.

## Executive status

The campaign did not pass Gate 1.

It did useful work on the fast-cycle launcher, evidence capture, cleanup,
race-state handling, full-lap scaffolding, diagnostics, and fail-closed safety
checks. It also proved that the controller reaches Gate 1 and actively sends
commands there. However, those commands consistently move the vehicle away
from the visible Gate 1 target. The dominant failure is therefore
control/trajectory logic, especially the Gate 1 roll sign and the aggressive
recenter envelope, rather than an idle state machine, a missed race credit, or
an inability to detect the gate.

The dirty full-lap controller must not be promoted or used as the next live
entry point. First disable its powered entry, separate the reusable
infrastructure from the experimental control law, and implement the bounded
Gate 1 recenter stage described below.

No code should be deleted wholesale. Preserve the evidence and useful
infrastructure, but remove or disable the failed control authority identified
in this handoff.

## Repository snapshot at handoff

- Repository: `C:\Users\John\killallhumans`
- `HEAD`: `5545a60de268588f3de1136bededa99f4f445a99`
- `origin/main`: `5545a60de268588f3de1136bededa99f4f445a99`
- Checkpoint commit for the dirty campaign: none
- Dirty diff size at audit: 4,293 insertions and 359 deletions
- Current `scripts/aigp_vq2_run.py` SHA-256:
  `7804fee1c4a748def8a313b899a48f4ee73d170c1282e4756a840a15453f9aa1`
- Current `tests/test_aigp_vq2_runner.py` SHA-256:
  `b92d03a87914b391215913a503b4af282e33644b74a853808b99a5ee05605428`
- Pre-handoff dirty-patch audit identity:
  `5f38e69deb43802d037640529884503f969af852`

The six pre-existing dirty files are:

- `docs/aigp/fast-flight-cycle.md`
- `scripts/aigp_vq2_fast_cycle.py`
- `scripts/aigp_vq2_run.py`
- `scripts/dev.ps1`
- `tests/test_aigp_vq2_fast_cycle.py`
- `tests/test_aigp_vq2_runner.py`

The exact current runner hash has not been powered-flown. The latest powered
run used source `1ce57e428e73…`, which was subsequently rejected and reverted.
After that revert, the thread ran 296 focused runner tests but no further
powered flight. Do not attribute the latest flight to the current source.

The stopped Codex session record is:

`C:\Users\John\.codex\sessions\2026\07\22\rollout-2026-07-22T11-40-30-019f8b21-1b5b-7992-a351-86a902789c0a.jsonl`

Session ID:
`019f8b21-1b5b-7992-a351-86a902789c0a`

## What the campaign actually accomplished

The powered-cycle audit found 136 total cycles:

| Stage | Result |
| --- | ---: |
| Calibration | 1/1 succeeded |
| Gate 0 | 9/10 succeeded |
| Gate 0 observe | 1/1 succeeded |
| Full lap | 0/124 succeeded |

Of the 124 full-lap attempts:

- 109 received authoritative Gate 0 credit and entered Gate 1 control.
- The maximum authoritative gate index was 1.
- Gate 1 crossing was armed 0 times.
- Gate 1 was credited 0 times.
- Gate 2 was entered 0 times.
- `race_finished` or a completed lap was observed 0 times.
- Cleanup was confirmed on all 124 attempts.

The terminal outcomes among the 109 Gate 1 control attempts were:

| Outcome | Count |
| --- | ---: |
| Primary target lost | 77 |
| Collision/body-rate safety abort | 29 |
| Untracked-contact-risk abort | 2 |
| Roll-limit abort | 1 |

This is meaningful negative evidence. The controller is not one small threshold
away from success; it repeatedly follows the wrong physical trajectory.

The first ten full-lap Gate 0 passes had a median time of about 2.672 seconds.
The final twenty had a median of about 2.563 seconds. That difference is not
accepted evidence of an overall speed improvement: source changed almost every
run, packet timing is noisy, and downstream progress remained zero.

## Latest powered run

Evidence directory:

`C:\Users\John\aigp-evidence\fast-flight-cycles\20260723T051544Z-full-lap-482c66f5`

Identities:

- Powered source: `1ce57e428e73…` — rejected and no longer current
- Manifest: `2bd54b0fd5d1…`
- Trace SHA-256: `a7696040539e…`

Observed sequence:

1. Gate 0 was credited at about 2.562 seconds.
2. Gate 1 was acquired around image position `x=518`, `y=53`, with an
   approximately 99-by-93-pixel box and race status 1.
3. Gate 1 control continued for about 2.2 seconds.
4. The target moved farther right and left the image.
5. The stage stopped with `primary gate target lost`.
6. Cleanup was confirmed.

This run is useful diagnostic evidence only. It is not evidence for the current
post-revert tree.

## Gate 1 diagnosis

### The runner is trying

The full-lap state machine reaches `_run_course_gate`, acquires Gate 1, and
sends repeated recenter commands. The failure is not that the code remains in
Gate 0 or waits indefinitely.

Across the latest 22 comparable runs:

- median target `x` moved from approximately 517 to 625.5;
- median roll changed by approximately `-0.356 rad`;
- 0 runs left recenter for approach or crossing.

In the latest run there were three acquisition ticks and 71 recenter ticks,
with zero approach ticks. Initial recenter commands were approximately
`roll=-0.25`, `pitch=-0.25`, and `thrust=0.35`. Attitude moved from roughly
`roll=-0.032`, `pitch=-0.052` to approximately `roll=-0.40`,
`pitch=-0.41`, near the safety guard. The target briefly moved from `x=518` to
`x=511`, then moved through approximately 563 to 623 and exited the right edge.

### Roll direction is empirically wrong

`course_gate_roll_target(normalized_x, recenter=True)` deliberately reverses
the image-error sign. A target to the right, represented by positive
normalized `x`, produces a negative roll target. The current test
`test_course_gate_roll_target_counter_rotates_then_banks_toward_gate` freezes
that behavior as expected.

The live evidence shows that this negative roll moves the target farther right,
not toward center. The cyan course-line observation after Gate 0 also
consistently indicates a physical right bend. Gate 0's positive preturn is then
followed by a negative exit counterroll and negative Gate 1 recenter, opposing
the observed turn.

The next experiment must isolate roll sign. Do not combine the sign test with
pitch, thrust, tracker, or threshold changes.

### Recenter is a persistent state trap

`course_gate_recenter_required` remains true while either:

- elapsed recenter time is below 0.60 seconds; or
- absolute normalized horizontal error exceeds 0.35.

On a 640-pixel-wide image, the latter corresponds to remaining outside roughly
`x=208..432`. The target enters around `x=518` and moves farther right, so the
controller never satisfies the transition condition. A nominal 0.60-second
phase therefore continued for more than 2.2 seconds while the wrong control law
kept increasing the error.

### The current envelope is too aggressive

The dirty controller uses:

| Quantity | Dirty full-lap value | Bounded recenter plan |
| --- | ---: | ---: |
| Roll target limit | `0.41 rad` | `0.05 rad` |
| Body-rate limit | `0.25 rad/s` | `0.12 rad/s` |
| Pitch target | as low as `-0.55 rad` | exact-zero pitch basis |
| Thrust | `0.35` | `0.21..0.30` |
| Intended recenter window | `0.60 s`, but condition can extend it | hard `0.60 s` |
| Passage authority | enabled through full-lap | prohibited |

The negative pitch and 0.35 thrust add forward acceleration while horizontal
error is increasing. With the target already high and right-clipped, this
causes it to split into fragments and leave the image before lateral alignment
is recovered.

### This was not a missed crossing credit

The latest run's maximum Gate 1 target area was 25,276 pixels squared; the
campaign maximum was 27,030. Maximum observed width was roughly 181 to 192
pixels. The crossing guard requires, among other conditions, area up to
161,280 pixels squared, width at least 512 pixels, full vertical clipping, and
centering. Gate 1 never approached those conditions and crossing was never
armed.

Tracker and fragment-selection changes sometimes extended visibility. They
cannot correct the physical path and must not be treated as evidence of
progress toward passage.

## Code disposition

### Retain and checkpoint separately

Retain these changes after review, separated from experimental controller
tuning:

1. Fast-cycle infrastructure: live lease, small manifest, trace/result
   capture, cleanup verification, and evidence-directory handling.
2. Fail-closed safety: fresh vision-generation checks, untracked-contact-risk
   handling, watchdogs, command bounds, exact race transitions, and confirmed
   disarm/reset/cleanup.
3. Diagnostics: `cyan_course_line_observation`, raw target/detection logging,
   target trajectory metrics, and source/manifest/trace identities.
4. Race-state and offline full-lap scaffolding:
   `full_lap_crossing_status_decision`, `_official_lap_time_s`,
   `_complete_course_gate`, and the bounded `_run_full_lap` orchestration.
   Retain authoritative gate-index transitions, looping only within the gate
   bound, official lap time only from `race_finished`, and the overall timeout.
5. `_wait_for_next_flight_command_slot` and its fail-closed inter-stage 50 Hz
   pacing.
6. The fresh-frame, vision-generation, and exact race-state acquisition
   structure in `_acquire_course_gate`, but with exact-zero transition thrust
   until thrust is separately proved.
7. Edge/fragment selector helpers and their offline tests, provided they stay
   tracker-only. A composite or stale target must not gain crossing or powered
   control authority.

The full-lap state machine may remain as disabled/offline scaffolding. Retain
does not mean it is admitted as a powered stage.

### Remove or disable before another powered candidate

The following behavior is contradicted by live evidence or violates the
bounded Gate 1 milestone. Remove it, or disable it from all powered dispatch
until it is replaced and independently accepted:

1. Remove `"full-lap"` from `FAST_POWERED_STAGES` in
   `scripts/aigp_vq2_fast_cycle.py`.
2. Remove `"full-lap"` from the allowed `flight-cycle` stages in
   `scripts/dev.ps1`.
3. Remove the powered `full-lap` advertisement and invocation examples from
   `docs/aigp/fast-flight-cycle.md`.
4. Disable the aggressive `_run_course_gate` passage path from live dispatch.
   Keep it only as offline scaffold if doing so is useful.
5. Restore or replace the Gate 0 centering law. The dirty
   `gate0_centering_roll_target` changes the previously live-proven
   `+0.15*x_error` behavior to `-0.15*x_error`; no downstream Gate 1 pass
   validates that sign change.
6. Remove the reversed recenter branch in
   `course_gate_roll_target`, including the current `direction=-1` behavior.
7. Remove or rewrite
   `test_course_gate_roll_target_counter_rotates_then_banks_toward_gate`; it
   asserts the empirically divergent sign as correct.
8. Remove the current powered recenter-rate amplification and aggressive
   constants:

   - `COURSE_RECENTER_ROLL_GAIN = 0.80`
   - `COURSE_RECENTER_ROLL_LIMIT_RAD = 0.41`
   - `COURSE_RECENTER_ROLL_RATE_SCALE = 1.60`
   - `COURSE_RECENTER_PITCH_RATE_SCALE = 1.60`
   - `COURSE_RECENTER_MAX_RATE_RAD_S = 0.25`
   - `COURSE_RECENTER_PITCH_GAIN = 0.80`
   - `COURSE_RECENTER_MIN_PITCH_RAD = -0.55`
   - `COURSE_RECENTER_THRUST = 0.35`

9. Replace `course_gate_recenter_required`. Its open-ended horizontal-error
   clause turns the 0.60-second recenter phase into an unbounded state trap.
10. Remove cyan-line actuation from the acceptance candidate. Passive
    `cyan_course_line_observation` tracing can remain, but
    `course_line_preturn_roll`, `course_line_exit_counterroll`, and the Gate 0
    preturn/counterroll command block are unproved.
11. In particular, disable Gate 0 course-line exit counterroll from powered
    continuation:

    - `COURSE_LINE_EXIT_COUNTERROLL_ONSET_AREA_SCALE`
    - `COURSE_LINE_EXIT_COUNTERROLL_RAD = 0.08`
    - `_run_full_lap(... course_line_exit_counterroll_enabled=True)`

    This behavior was retained on proxy results, but it leaves the vehicle
    banked against the physically observed right bend.
12. Remove Gate 0 launch tuning and race-packet timing from the acceptance
    candidate, preserving it only on an experiment branch:

    - `COURSE_GATE0_BOOST_UNTIL_S = 0.80`
    - related Gate 0 exit-pitch and thrust overrides
    - `gate0_phase_alignment_delay_s`
    - `_align_gate0_race_phase`

    Race-phase alignment produced the first collision-free Gate 1
    continuation in `20260722T205125Z-full-lap-5fd625ea`, source
    `8b9628cad7f2…`, trace `1f19d4136b4e…`. That is a useful hypothesis, not
    proof of lap progress, and it currently optimizes a guessed 2.50-second
    target-loss phase.
13. Remove the dead rejected Gate 0 close-thrust residue if it is not used by a
    separately accepted stage:

    - `COURSE_GATE0_CLOSE_THRUST_FLOOR = 0.30`
    - `COURSE_GATE0_CLOSE_FLOOR_MAX_CONTROL_Y_PX`
    - tests whose only purpose is to freeze those values

    The last full-lap wiring for this candidate was rejected and removed;
    `_run_full_lap` currently uses `COURSE_GATE0_MIN_THRUST = 0.21`. Leaving
    dormant constants and normative tests makes the rejected behavior easy to
    resurrect accidentally.
14. Restore exact-zero Gate 1 observation authority. Remove nonzero
    `hold_thrust` from the observation path and
    `COURSE_TRANSITION_THRUST = 0.35` from the acceptance candidate until
    independently proved.
15. Remove or rewrite tests that imply the aggressive `0.41`, `0.25`,
    `-0.55`, and `0.35` envelope is accepted or that the full-lap lifecycle is
    admitted for powered use. Tests may continue to exercise the disabled
    state machine as offline code, but their names and assertions must state
    that boundary.

Do not delete immutable run evidence, traces, manifests, session logs, or
cleanup records, including evidence from rejected candidates.

### Verify these rejected candidates remain absent

The long thread implemented and later rejected or removed each of the
following. Do not resurrect them without a new, isolated hypothesis and new
evidence:

- optical-flow relief;
- neutral-bank collapse recovery;
- early-bank capture;
- one-token or two-token post-credit precharge;
- composite-to-primary vertical-derivative reset;
- dormant Gate 0 close-thrust wiring;
- forced physical-bank switches;
- `2.0` roll-response scaling;
- immediate or delayed `0.32-rad` roll caps;
- `-0.57-rad` pitch floor;
- `1.8` pitch-response scaling;
- `-0.10-rad` Gate 0 exit pitch.

Tilt compensation was discussed at the end of the thread but was not
implemented, tested, or flown. It is a proposal, not retained work.

### Re-prove before promotion

The following dirty changes may be useful but have only proxy or
single-candidate evidence:

- Gate 0 boost duration `0.80 s`;
- Gate 0 exit pitch `0.0 rad`;
- Gate 0 minimum thrust `0.21`;
- course-line preturn limit `0.13 rad`;
- race-packet phase alignment;
- edge/fragment continuation and fusion;
- course-gate visual crossing thresholds, especially the 70%-frame area cap;
- the exact size/area thresholds in `select_untracked_contact_risk`;
- acquisition, approach, and full-lap state-machine logic;
- any behavior in the current exact post-revert runner hash.

Re-prove them with a fixed source hash and repeated runs. Judge Gate 0 tuning
by its effect on safe Gate 1 recenter entry, not solely Gate 0 time, peak
target area, or target lifetime.

## Required recovery sequence

### 1. Preserve and separate the dirty work

Before behavioral edits, preserve the current diff and evidence, then split it
into at least two reviewable checkpoints in an isolated branch or worktree:

1. infrastructure, safety, race-state, and diagnostics;
2. experimental controller behavior, clearly marked unaccepted.

Do not merge the second checkpoint into the powered path merely because its
unit tests pass.

### 2. Add a dedicated bounded `gate1-recenter` stage

The next powered capability should be `gate1-recenter`, not `full-lap`. It
must:

- use the already proved Gate 0 and Gate 1 acquisition sequence;
- stop before Gate 1 passage;
- treat any gate-index change as an abort, not success;
- latch an exact-zero pitch basis;
- use yaw target zero;
- use a roll target based only on measured horizontal error and error rate:
  `clamp(0.12*x_error + 0.025*x_rate, -0.05, +0.05) rad`;
- clamp roll and pitch rates to `±0.12 rad/s`;
- keep thrust within `0.21..0.30`;
- enforce a hard 0.60-second recenter window;
- disarm/reset and confirm cleanup at completion or any abort.

These bounds come from
`2026-07-20-vq2-development-continuation-handoff.md` and
`vq2_predictive_controller.md`. The dirty full-lap controller exceeds them and
must not silently redefine the milestone.

### 3. Define recenter success before running it

A recenter attempt succeeds only when all of the following are true:

1. race status authoritatively reports Gate 1;
2. the Gate 1 target is present on at least three distinct fresh frames;
3. within 0.60 seconds, absolute horizontal error decreases across fresh
   frames;
4. the target reaches a conservative corridor, provisionally
   `abs(normalized_x) <= 0.35`;
5. the corridor is held for three fresh frames;
6. target freshness, command bounds, contact safety, and stream generation
   remain valid;
7. the stage stops without attempting passage and cleanup is confirmed.

Abort early if:

- absolute horizontal error grows by more than 24 pixels beyond entry after
  three fresh control frames;
- the primary target is lost;
- stream generation changes;
- collision/contact risk is observed;
- attitude or command bounds are approached;
- race index changes;
- the hard 0.60-second window expires.

### 4. Isolate the sign experiment

Use replay/offline traces first to compare exactly one sign choice against its
opposite. The empirically indicated physical turn is to the right, but the
live comparison must use identical source, bounds, pitch basis, thrust, and
tracker logic.

For any powered comparison:

- keep one source hash fixed;
- change only roll sign;
- run at least three repeats per sign, preferably five;
- do not edit source between repeats;
- reject on any cleanup failure or safety event;
- select using fresh-frame horizontal-error convergence, not target lifetime
  or one-frame peak area.

Only after one fixed candidate achieves at least 3/3 clean corridor holds
should a separately reviewed Gate 1 passage stage be designed. Full-lap
authority comes later.

### 5. Use a fast evidence loop

After each edit, run directly affected focused tests. Run `test-vq2` once a
candidate has survived offline review and is accepted for live/promotion
consideration; do not run the entire VQ2 suite before every scalar experiment.

Each trace summary should automatically report:

- source, manifest, and trace hashes;
- Gate 1 entry and final horizontal error;
- fresh-frame horizontal-error slope;
- fresh control-frame count;
- minimum and maximum roll and pitch;
- maximum target area and width;
- authoritative maximum gate index;
- contact/safety outcome;
- cleanup confirmation.

## Experiment-quality findings

The campaign ran 124 full-lap attempts using 107 distinct runner hashes. That
is nearly one revision per flight and provides little replication.

The session audit also found:

- 54 `test-vq2` invocations;
- 205 targeted runner-test invocations;
- 322 patches;
- 47 explicit rejection messages;
- roughly 40 rejected configurations across 13 hypothesis families;
- 83 reverts, removals, or baseline restorations.

Several repeated reports of approximately 2,300 to 2,400 “relevant” passing
tests excluded the same nine host-environment failures. They were not fully
green canonical-suite results.

Specific process errors to avoid:

- an early pitch experiment edited hover rather than Gate 0, invalidating its
  conclusion;
- multiple runs changed coupled variables and could not identify causality;
- roll-sign interpretation repeatedly alternated between image recentering and
  physical course direction;
- stale-frame continuation was once counted as about 0.25 seconds of apparent
  improvement;
- terminal summaries were sometimes interpreted before trace review and later
  reversed;
- single-frame peaks were nearly promoted as thresholds;
- proxy metrics improved while authoritative Gate 1 credit remained zero;
- the full VQ2 suite before almost every candidate added roughly five-minute
  gaps around flights that lasted only seconds;
- there was no durable checkpoint separating infrastructure from provisional
  tuning.

The new loop should favor one-variable hypotheses, fixed-hash replication,
fresh-frame metrics, and rapid rejection.

## Campaign chronology

| Local time | Work family | Handoff disposition |
| --- | --- | --- |
| 11:44–12:01 | Gate 0 boost and pitch timing | Longer boost rejected. Pitch-blend conclusions invalid because the edit affected hover, not Gate 0. |
| 12:16–13:51 | Full-lap path, transitions, race timing | Infrastructure retained. Many thrust/boost/brake/sign variants rejected. Race-phase alignment remains re-prove. |
| 14:24–15:22 | Gate 1 recenter and clipped-target tracking | Edge and fragment helpers useful offline. Forced physical-bank changes and 2.0 response rejected. |
| 15:30–16:38 | Pitch and roll response tuning | Proxy values `0.80`, `1.60`, and `-0.55` survived in the dirty tree but produced no Gate 1 credit and must leave powered authority. |
| 16:49–20:44 | Scalar sweeps and counterroll timing | Plateaued. Best peak area was not accepted because an exact-source repeat produced two threat-2 impacts and a 15.26-rad/s spike. |
| 21:05–22:01 | Structural recovery attempts | Optical flow, neutral-bank recovery, early-bank capture, and precharge rejected/removed. Generation and contact guards retained. |
| 22:03–22:09 | Gate 0 close-thrust wiring | Rejected and removed from full-lap wiring; dead constants remain. |
| 22:09–22:19 | Vertical-derivative reset | Rejected and removed. Latest powered source is not current. |
| 22:26–22:28 | Tilt compensation review | Proposal only; not implemented, tested, or flown. |

Representative evidence:

- First collision-free Gate 1 continuation:
  `C:\Users\John\aigp-evidence\fast-flight-cycles\20260722T205125Z-full-lap-5fd625ea`
- Best single peak-area run:
  `C:\Users\John\aigp-evidence\fast-flight-cycles\20260723T022325Z-full-lap-8ddd4776`
- Exact-source repeat with safety problems:
  `C:\Users\John\aigp-evidence\fast-flight-cycles\20260723T022454Z-full-lap-017f1610`
- Latest powered run:
  `C:\Users\John\aigp-evidence\fast-flight-cycles\20260723T051544Z-full-lap-482c66f5`

Evidence root:

`C:\Users\John\aigp-evidence\fast-flight-cycles`

## Resume checklist

1. Read the authoritative 2026-07-18 handoff and this recovery handoff.
2. Confirm `HEAD`, worktree status, and current file hashes before changing
   anything.
3. Preserve the dirty diff and immutable external evidence.
4. Split reusable infrastructure from unaccepted controller behavior.
5. Remove `full-lap` from powered entry points.
6. Remove or disable the failed Gate 1 control authority and normative tests
   listed above.
7. Add focused tests for a hard-bounded, no-passage `gate1-recenter` stage.
8. Run affected tests and the non-live VQ2 safety suite for the accepted
   candidate.
9. Compare roll sign offline with every other variable fixed.
10. If powered work is explicitly authorized under the live safety contract,
    run fixed-hash recenter repeats only.
11. Do not design Gate 1 passage until recenter succeeds repeatedly.
12. Do not restore full-lap authority until Gate 1 passage is separately
    accepted.

## Claims that must not be made

Until new authoritative evidence exists, do not state that:

- a full lap has completed;
- Gate 1 has been passed or nearly passed;
- Gate 2 has been reached;
- the dirty current hash has been powered-flown;
- the aggressive recenter controller is accepted;
- the campaign proved an overall speed improvement;
- peak target area or target lifetime is equivalent to downstream progress;
- the repeated “relevant tests passed” reports represent fully green
  canonical suites.

The honest resume point is: Gate 0 is routinely credited, Gate 1 is visible,
and the current Gate 1 controller has been empirically falsified. The next
useful powered milestone is a short, low-authority, no-passage recenter that
demonstrably reduces horizontal error on fresh frames.
