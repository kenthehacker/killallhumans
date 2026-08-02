# VQ2 Training-Mode Course Continuation Prompt

You are continuing an active AI Grand Prix controller-development campaign.
This is not a fresh architecture exercise. Work from the current repository
state, preserve verified behavior, and iterate quickly against the official
FlightSim build 3385 in Training mode until the simulator authoritatively
reports `race_finished`.

## Objective and definition of done

Complete the entire build-3385 Training-mode course using the production path:

```text
UDP JPEG vision + HIGHRES_IMU + race status
                  -> target tracking and IMU attitude estimation
                  -> safety-gated body-rate/thrust commands
```

There is no usable pose or gate-map stream in this build. Do not invent either
one or revive a VQ1 assumption that conflicts with verified VQ2 behavior.

Success means all of the following:

- race status shows the sequential authoritative gate transitions;
- race status reports `race_finished=true`;
- the run used one exact, committed candidate;
- navigation and cleanup outcomes are reported separately;
- the trace, result, run manifest, and commit are preserved.

`0 -> 1` means Gate 0 was passed and Gate 1 became active. Gate 1 is not passed
until the authoritative transition is `1 -> 2`. A visually plausible crossing,
internal state transition, detector event, or controller COMMIT is not credit.
Once `race_finished` is achieved, stop powered iteration, preserve the exact
candidate and evidence, and report it. Do not begin speed optimization unless
the user asks.

## Authority and required reading

Read these before editing or flying, in this order:

1. `Agents.md`, especially the standing rapid-course iteration policy.
2. `docs/aigp/2026-07-18-vq2-handoff.md`, authoritative for the verified
   interface and safety contract.
3. `docs/aigp/2026-07-28-vq2-course-architecture-reset-handoff.md`, for the
   clean-controller architecture and rejected legacy direction.
4. `docs/aigp/fast-flight-cycle.md`, for the bounded live workflow.
5. `docs/autonomous_iteration.md`, as promotion/background guidance only.
   Its replay/synthetic-before-live tier ordering is superseded for the active
   rapid `visual-course` campaign by the standing policy in `Agents.md`.

Inspect current Git state and the newest external evidence rather than trusting
old counters in a handoff. Public VQ1 material is historical context, not an
override for build 3385.

The July 18 handoff remains authoritative for the verified build-3385
interface and powered-flight safety contract. Its historical statements that
no full-course attempt had occurred, and its older instruction not to attempt
one yet, are superseded by the standing rapid-course policy and the later live
evidence below. Flag this status conflict for human review; do not discard the
handoff's interface or safety findings.

The user authorizes repeated, bounded `visual-course` attempts in FlightSim
build 3385 Training mode for this continuation, until `race_finished` or an
explicit revocation. This does not authorize another powered stage, physical
hardware, removal of safety gates, or concurrent live workflows.

## Evidence precedence — live simulator first

For navigation and closed-loop course decisions, use this evidence order:

1. authoritative live race transitions, `race_finished`, and safety events;
2. the exact-hash live `result.json` and JSONL trace from FlightSim;
3. fixed-hash live repeats when one run leaves a material ambiguity;
4. recorded-trace counterfactuals or replay;
5. synthetic fixtures, unit tests, and legacy simulation.

FlightSim build 3385 is the primary truth for this course. It is cheap enough
to run promptly and it includes the real closed loop: a command changes the
aircraft state and therefore changes future camera images. Replay and tests may
reject unsafe logic, isolate a mechanism, and preserve envelope or directional
invariants. They cannot establish that a candidate improved the live
trajectory, cleared a gate, or completed the course.

If a replay prediction conflicts with the same candidate's live telemetry,
the live result wins. Mark the replay hypothesis unvalidated, locate the first
input/state/trajectory divergence, and fix the causal model. Do not tune a
replay until it says the hoped-for answer, add a new replay abstraction, or
delay a green candidate's bounded simulator attempt merely to improve offline
fidelity. Recorded regression tests should prefer bounded directional and
envelope behavior over exact commands, internal modes, or track identities.

Human video is useful for spotting symptoms, but use the synchronized live
trace to distinguish target motion, aircraft attitude, command timing, and
authoritative credit. An earlier target-roll change, internal COMMIT, lower
offline error, or visually plausible crossing is not a navigation improvement
unless live clearance geometry or authoritative progress improves.

## Starting snapshot — verify before relying on it

The latest clean, flown, and pushed controller baseline on 2026-08-02 is F179,
commit `13c870e53ebd72f572f7f90160276455fe8c7aac`. Verify that this commit is an
ancestor of the current `origin/main`; do not trust branch names or this dated
snapshot without checking Git and the external evidence.

The decisive fixed-hash live runs are:

- F178: `20260802T104817Z-visual-course-9c051bdc`,
  `20260802T105051Z-visual-course-44075b1d`, and
  `20260802T105319Z-visual-course-937cf42d`;
- F179: `20260802T112058Z-visual-course-254ce53f`.

All four runs received authoritative `0 -> 1` credit. F178 repaired the Gate 0
ambiguous-bottom censor collapse while preserving compensated climb. Treat
that as a repeated live milestone: freeze the Gate 0 launch, vertical arrival,
and passage behavior unless new live evidence directly shows a regression.

F179 did not receive `1 -> 2` credit. The correct Gate 1 track was acquired
cleanly and remained fresh/current through the initial miss, so early tracking
loss was not the first blocker. F179's response horizon created a positive-roll
pulse at `t=4.797`, revoked it at `t=4.891`, and did not reverse positively for
good until `t=5.156`, when the gate was already centered. Actual roll reached
zero 329 ms after center instead of F178's 453–469 ms, but the last valid Gate
1 position still escaped at about `x=+0.18` to `+0.19` with high rightward image
rate. Gate 1 became engulfing, entered PREDICT at `t=5.860`, never entered
COMMIT, and received no credit. This is internal latency improvement without a
course-level improvement; the three-phase roll target explains the shakier
turn-in.

The F179 corridor-continuation mechanism also did not solve the miss. At
proximity it retained only a non-live, continuity-only corridor, and its x/y
tubes remained more than five times their safe halves. This evidence must not
be used to loosen freshness or body-clearance containment. Passage admission
was downstream of the uncontrolled lateral arrival.

F179 is also the canonical replay/live mismatch. Its recorded counterfactual
expected a nonnegative target roll at the equivalent of `t=5.110`; live F179
still had `target_roll=-0.040` there. Its fitted-corridor test expected continued
live custody, while the real corridor stopped refreshing before proximity.
The tests exercised code paths, but their assumed closed-loop input sequence
did not survive the real trajectory.

The course worktree may contain intentional post-F179 edits from another
session. Inspect and preserve them. Never reset, stash, overwrite, merge, or
claim those edits unless they are committed and explicitly selected. Recheck
free disk space, but do not delete historical evidence, captures, worktrees,
or unknown user data.

## Immediate technical steering

The first current blocker is Gate 1 lateral arrival control. Preserve the
verified Gate 0 path and inspect a window spanning Gate 1 acquisition through
the first engulfing/loss event. Compare:

- outer Gate 1 x position and physical image-x rate;
- target roll, actual roll, roll rate, and wire command;
- each response-target sign change and total target variation;
- the timing of actual-roll zero relative to outer-center zero;
- closure/TTC, last exact observation, censor/engulfing onset, and PREDICT;
- corridor freshness and containment, separately from steering authority;
- pitch/thrust and vertical image motion to guard against a vertical regression.

The next candidate should change one current-gate lateral mechanism. Diagnose
why F179's anticipated reversal was revoked instead of stacking another gain
on its noisy intercept trend. Prefer a stable, rate-limited or hysteretic
arrival/stopping owner that can retire the carried bank and establish enough
opposite bank before center crossing. It must respond to actual lateral
translation and plant delay, not merely make the target-roll sign flip earlier.

Judge it by the live plant: actual bank and rightward image rate should be
materially lower before center, the correction should remain smooth and
sustained, and the last valid Gate 1 offset should shrink. The primary success
criterion remains authoritative `1 -> 2` credit. A transient target pulse,
earlier internal reversal, or replay-only improvement is insufficient.

Keep steering, passage admission, target identity, and recovery causally
separate. `continuity_only` geometry may conservatively preserve lineage but
is not fresh passage authority. Do not loosen freshness or containment to make
COMMIT reachable. Do not patch post-miss reacquisition first: in F179 the
correct track survived through the initial miss, while later fragment locks,
SEARCH heading ratcheting, and vertical pursuit were downstream effects.

Continue to respect finite prediction age, uncertainty, stream freshness,
authoritative gate ownership, and every watchdog. Once Gate 1 receives credit,
move to the next earliest blocker and preserve the completed gate behavior.
The objective is the whole course, not a Gate 1-only controller.

Change one causal mechanism at a time. If two consecutive flights fail through
the same mechanism, compare their decisive live windows and take a larger-step
control/state audit before adding another patch. Ask whether the controller is
optimizing the wrong observable, differentiating a noisy projection, reacting
after the plant's stopping horizon, or unable to express the needed sustained
action. Flat candidate numbers are not progress.

## Required rapid iteration loop

For each candidate:

1. Inspect the newest exact-hash live `result.json` and compressed JSONL trace
   under `C:\Users\John\aigp-evidence\fast-flight-cycles`. Identify the first
   causal navigation blocker before consulting a replay or terminal symptom.
2. State one bounded hypothesis, the expected live change at the decisive
   timestamps, and any replay assumption that the live run must confirm. Make
   the smallest coherent change and add or adjust a focused regression test
   without pretending synthetic data proves closed-loop flight.
3. Run directly affected tests, for example:

   ```powershell
   .\scripts\dev.cmd test-target tests/test_aigp_vq2_clean_course_stage.py
   ```

4. Run the dedicated non-live safety suite exactly once for the accepted
   candidate:

   ```powershell
   .\scripts\dev.cmd test-vq2
   ```

5. If green, ensure the candidate is clean, commit it, and push the exact
   commit to the continuation branch. Never force-push. Do not include private
   captures, credentials, external evidence, caches, or unrelated user edits.
6. Confirm that no other live workflow owns the host-wide lease, then run one
   bounded attempt:

   ```powershell
   .\scripts\dev.cmd flight-cycle visual-course
   ```

7. Record the exact commit, run ID, authoritative transitions, maximum/final
   gate index, `race_finished`, first causal failure, and cleanup outcome. At
   the decisive live window, compare the predicted mechanism with actual
   target, command, attitude, image position/rate, and tracker state.
8. Record any replay expectation versus live observation mismatch. Keep or
   reject the candidate explicitly based on live navigation evidence, then
   begin the next iteration.

Target roughly 15–25 minutes from a diagnosed failure to the next powered
attempt. Do not run `test-fast`, `test-unit`, benchmark, promotion, broad replay
matrices, another general review, or an abstraction/proof detour before every
flight. Reserve those for a meaningful course milestone or final promotion,
unless a check reveals a hard blocker involving:

- nonfinite or out-of-envelope commands;
- wrong authoritative race-gate ownership;
- stale control inputs;
- the in-flight collision watchdog;
- live-lease concurrency; or
- inability to establish a usable simulator state.

A failed flight is useful evidence. Tests and replay can reject bad logic, but
they cannot prove a closed loop whose commands change future images. When the
first causal explanation remains ambiguous, prefer another bounded exact-hash
simulator attempt over inventing a replay-only conclusion.

## Powered-flight safety contract

Never weaken or bypass:

- proved reset epoch;
- countdown and authoritative GO;
- fresh JPEG, IMU, and race-status requirements;
- bounded body-rate/thrust commands and command pacing;
- hard attempt timeout;
- attitude, rate, stale-stream, and collision watchdogs;
- the nonblocking host-wide FlightSim live lease;
- best-effort exact-zero, disarm, reset, and final disarm from a `finally`
  cleanup path.

Report navigation separately from cleanup. A recorder or diagnostic failure
must not erase a real navigation milestone, but a simulator state that cannot
support the next run must be repaired or FlightSim relaunched before
continuing. Training mode is configured session state and is not
machine-readable in build 3385; do not claim fresh visual proof of it.

Do not add manual screenshots, interactive preflight ceremony, promotion
freezes, full dependency inventories, or console challenges to the rapid
flight loop. Do not launch a duplicate FlightSim process. GUI selection may
still require the user if the simulator is not in the required session state.

## Git and workspace discipline

Start from the latest clean `origin/main`, preferably on a dedicated
continuation branch. First run `git fetch --prune origin`, inspect
`git status --short --branch`, and verify ancestry. Never discard a dirty
worktree you did not create. If another worktree contains post-F179 or
unrelated changes, leave it intact.

Every flown candidate must correspond to the clean commit recorded by the
flight result. Do not claim an uncommitted working tree was the recorded
candidate. Do not merge the continuation branch to `main` unless the user asks.

## Advisory larger-blocker checkpoint

After two consecutive live failures through the same mechanism, or 90 minutes
without authoritative progress, request one read-only independent review while
continuing any already-green rapid-loop candidate. The review must take a step
back: test whether the controller owns the right state and observable, whether
the live plant invalidated an offline assumption, and whether patch-stacking is
hiding a larger architectural blocker.

If a Kimi reviewer is available in the session, it may be used for this
advisory audit. If its integration is unavailable or broken, do not spend a
course iteration repairing reviewer plumbing; use another read-only reviewer
or the Codex invocation below. Never start duplicate reviews, and never let a
review delay the next bounded sim after directly affected tests and `test-vq2`
are green.

From the active worktree, a suitable Codex invocation is:

```powershell
$reviewPrompt = @'
Read Agents.md and the authoritative VQ2 handoffs. Inspect the current Git
history/diff and the two decisive exact-hash visual-course results and traces
under C:\Users\John\aigp-evidence. Treat authoritative live simulator data as
primary. Report actual gate progress, the first causal divergence, any
replay/live mismatch, whether the controller is optimizing the right state,
whether the iteration direction is coherent or patch-stacking, and one bounded
mechanism for the next candidate. Distinguish measurement from inference. Do
not edit, test, fly, commit, or push.
'@
$worktreePath = (Get-Location).Path
codex -C $worktreePath --add-dir C:\Users\John\aigp-evidence `
  -s read-only -a never exec --ephemeral $reviewPrompt
```

Use any review only as advice. Verify it against live trace and race authority,
and summarize adopted steering in the next status report. Reviewer failure or
unavailability must not erase evidence or block the next candidate.

## Communication and stop conditions

After every powered attempt, give a short status containing:

- candidate/commit and run ID;
- authoritative transition chain and whether Gate 1 (`1 -> 2`) was passed;
- first causal blocker;
- what changed relative to the prior run;
- replay expectation versus live observation, when applicable;
- navigation outcome and cleanup outcome separately;
- the single next hypothesis.

Continue autonomously through routine failed attempts. Stop and ask the user
only when required authority is missing, a human GUI action is required, the
simulator cannot be restored to a usable state, disk/evidence integrity is at
risk, a safety invariant would need to change, or `race_finished` is achieved.

Begin by verifying that `origin/main` contains F179, preserving any dirty
post-F179 worktree, and inspecting the decisive F178/F179 live Gate 1 windows.
Explain why F179's first response-horizon reversal was revoked and choose one
stable current-gate lateral-arrival mechanism. Enter the rapid loop promptly,
then continue through each newly exposed blocker until authoritative
`race_finished`.
