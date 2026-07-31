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
5. `docs/autonomous_iteration.md`, only where it does not conflict with the
   faster standing policy in `Agents.md`.

Inspect current Git state and the newest external evidence rather than trusting
old counters in a handoff. Public VQ1 material is historical context, not an
override for build 3385.

The user authorizes repeated, bounded `visual-course` attempts in FlightSim
build 3385 Training mode for this continuation, until `race_finished` or an
explicit revocation. This does not authorize another powered stage, physical
hardware, removal of safety gates, or concurrent live workflows.

## Starting snapshot — verify before relying on it

At the time this prompt was written on 2026-07-30:

- `origin/main` was clean at commit `0745685741b28f9234af1bdd041aa4bc56a34338`
  (`F102: one crossing policy — gate 0 crosses via the energy-budgeted COMMIT`).
- F78 through F102 produced no authoritative progress beyond gate index 1.
- The most recent visually inspected exact-F102 run was
  `20260730T183518Z-visual-course-aaf88876`.
- That run made the authoritative transition `0 -> 1`, did not make `1 -> 2`,
  then reported environment collisions and a body-rate-limit abort. Cleanup
  was confirmed.
- The preceding F102 run,
  `20260730T182343Z-visual-course-334c208e`, had the same maximum gate index.
  Gate 1 became horizontally well aligned, but its vertical error worsened.
  The controller requested a strong descent near spawn-relative low altitude,
  the sink became unrecoverable despite high thrust, and the vehicle hit the
  floor. The body-rate and fragmented-track symptoms appear downstream of the
  vertical/ground-impact sequence.
- The terminal failure currently occurs while Gate 1 is active, and the
  low-altitude vertical endgame is the leading hypothesis. Its causal origin
  is not settled: Gate 0 can receive credit while delivering a bad altitude,
  vertical velocity, attitude, angular-rate, or closure state into Gate 1.
  Gate 0 credit proves sequencing, not a good handoff.

There may still be an uncommitted F103 experiment in
`C:\Users\John\killallhumans-course-reset`. It tapers requested descent near
spawn-relative ground altitude toward approximately `-0.15 m/s` and adds
`test_course_leg_vz_des_sink_is_capped_near_the_ground`. It was intentionally
excluded from `main`. Preserve it if present. Inspect it as a hypothesis, but
do not silently overwrite, discard, or assume it is accepted. If adopting the
idea, do so deliberately on the continuation branch, validate it, and keep the
candidate small.

The C: drive was approximately 97.75% full with only 5.35 GiB free. Recheck it.
Do not delete historical flight evidence, user worktrees, captures, or other
unknown data. Remove only disposable artifacts you created yourself, or ask
the user if material cleanup is needed.

## Immediate technical steering

Do not preassign the blocker to a gate merely from the active gate index at the
terminal event. For every run, inspect a transition-centered window spanning
at least 1.5 seconds before and after authoritative `0 -> 1`. Compare:

- Gate 0 tracking and crossing geometry;
- spawn-relative altitude estimate and vertical velocity;
- pitch, roll, yaw, and body rates;
- forward closure/expansion rate and braking state;
- Gate 1's last reliable pre-credit observation and first reliable
  post-credit position, motion, scale, and custody;
- the commands that established those states before and after credit.

Attribute the blocker to the earliest causal divergence:

- no `0 -> 1` credit or an unsafe approach is a Gate 0 problem;
- credit with a bad state already established before crossing is a Gate 0
  exit/handoff problem;
- a reasonable handoff followed by divergence caused by post-credit commands
  is a Gate 1 controller problem;
- correct authoritative ownership followed by adoption of the wrong visual
  target is a transition/custody problem.

Preserve stable parts of the Gate 0 path, but allow a focused Gate 0 approach
or exit change when evidence shows it created Gate 1's bad initial condition.
Do not equate credit with a healthy exit, and do not stack another broad
architecture rewrite on the clean controller.

For Gate 1:

- Diagnose the first divergence from the trace, not merely the terminal
  collision or the video.
- Treat the low-altitude commanded sink as the leading hypothesis, not a
  settled physical fact. There is no authoritative pose. Distinguish measured
  IMU/image behavior from inferences such as vortex-ring state or ground
  clearance.
- A briefly censored gate does not require constant visual lock. Carry bounded
  target/crossing custody using recent image motion, IMU attitude, and the
  estimated velocity through the existing finite prediction horizon.
- Do not dead-reckon indefinitely. Respect measurement age, uncertainty,
  `predict_max_gap`, COMMIT energy/measurement gates, stream freshness, and the
  collision/attitude/rate watchdogs.
- Do not adopt newborn corner fragments merely because the real aperture
  becomes engulfing or censored near the plane.
- Use authoritative race status as gate ownership. Vision may guide control
  but must neither fabricate credit nor veto an observed authoritative
  transition.
- Human visual observations from earlier runs included late turn initiation,
  excessive climb, and a brief wrong-direction turn after Gate 0. Several
  later candidates attempted to address these. Quantify them in the current
  trace before reopening them; do not assume either that they remain or that
  they are solved.

Change one causal mechanism at a time. If two consecutive flights fail through
the same mechanism, compare their decisive trace windows before adding another
patch. If milestone progress stays flat while candidate numbers grow, stop
patch-stacking and reassess the controller state/measurement interaction.

## Required rapid iteration loop

For each candidate:

1. Inspect the newest `result.json` and compressed JSONL trace under
   `C:\Users\John\aigp-evidence\fast-flight-cycles`. Identify the first causal
   navigation blocker and state one bounded hypothesis.
2. Make the smallest coherent change and add or adjust a focused regression
   test that represents the observed mechanism without pretending synthetic
   data proves closed-loop flight.
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
   gate index, `race_finished`, first causal failure, and cleanup outcome.
8. Keep or reject the candidate explicitly, then begin the next iteration.

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
they cannot prove a closed loop whose commands change future images.

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
worktree you did not create. If another worktree contains F103 or unrelated
changes, leave it intact.

Every flown candidate must correspond to the clean commit recorded by the
flight result. Do not claim an uncommitted working tree was the recorded
candidate. Do not merge the continuation branch to `main` unless the user asks.

## Independent Codex checkpoint every 90 minutes

At the start, and after each attempt, check elapsed wall time since the last
independent review. At 90 minutes or more, request a read-only Codex review.
Do not start a second review while one is already running. From the active
worktree, a suitable PowerShell invocation is:

```powershell
$reviewPrompt = @'
Read Agents.md and the authoritative VQ2 handoffs. Inspect the current Git
history/diff and the newest visual-course result and trace under
C:\Users\John\aigp-evidence. Report actual authoritative gate progress, the
first causal blocker, whether the iteration direction is coherent or
patch-stacking, and one concrete steering recommendation. Distinguish measured
evidence from inference. Do not edit, test, fly, commit, or push.
'@
$worktreePath = (Get-Location).Path
codex -C $worktreePath --add-dir C:\Users\John\aigp-evidence `
  -s read-only -a never exec --ephemeral $reviewPrompt
```

Use the review as advice, verify it against the trace and authority, and
summarize any adopted steering in the next status report. If Codex is
unavailable or the review fails, note that and continue the rapid loop; it
must not erase evidence or block the next candidate.

## Communication and stop conditions

After every powered attempt, give a short status containing:

- candidate/commit and run ID;
- authoritative transition chain and whether Gate 1 (`1 -> 2`) was passed;
- first causal blocker;
- what changed relative to the prior run;
- navigation outcome and cleanup outcome separately;
- the single next hypothesis.

Continue autonomously through routine failed attempts. Stop and ask the user
only when required authority is missing, a human GUI action is required, the
simulator cannot be restored to a usable state, disk/evidence integrity is at
risk, a safety invariant would need to change, or `race_finished` is achieved.

Begin by verifying the snapshot, inspecting the latest two F102 traces and any
preserved F103 diff, and analyzing the transition-centered window. Then choose
the smallest candidate aimed at the earliest evidence-backed divergence,
whether it lies in Gate 0 approach/exit, the handoff, or Gate 1 control, and
enter the rapid loop.
