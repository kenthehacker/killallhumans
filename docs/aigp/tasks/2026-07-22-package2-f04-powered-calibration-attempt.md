# Package 2 F04 powered calibration attempt

- Task ID: `vq2-package2-f04-powered-calibration-attempt`.
- Parent: `vq2-package2-f03-powered-calibration-attempt`.
- Starting local-main commit:
  `754fbe24173304d3cdcf50b6065d3b125cf2ab83`.
- Simulator target: FlightSim build 3385, Training mode.
- State: `identity candidate; not frozen or consumed`.
- Contract date: `2026-07-22`.

## Authority and stop boundary

The user's 2026-07-22 direct instruction authorizes autonomous preparation,
publication, and one consumption of exact `F04-A01` by the existing
`calibration-excite` wrapper without another permission prompt. The capture
authority is `conversation-2026-07-22-package2-f04-sim-capture`, direct user
instruction, simulator-only, and task-local. The user cannot monitor or answer
an interactive console prompt.

This authority does not authorize a retry, another attempt identity, `sign-id`,
`hover`, `gate0`, `gate0-observe`, Gate 1 control, a lap, another build/mode,
physical or HIL use, calibration acceptance, public release, or external
upload. A cleanup failure is a failed stage, and fallback use invalidates F04.

## Delegated Training attestation

The user explicitly delegated local console operation to Codex because they
cannot watch or answer the prompt. This does not waive the visual Training-mode
fact. Before candidate freeze, Codex captured and inspected the exact current
FlightSim payload window at original resolution:

- payload PID: `26176`;
- payload creation FILETIME: `134291804693001801`;
- payload image SHA-256:
  `9064dd1547a30afea1e3fb87652cc8194c3f5af556be40629dc491bb4f681362`;
- window handle: `2074873510`;
- private screenshot:
  `C:\Users\John\aigp-review\2026-07-22-current-flightsim-window.png`; and
- screenshot SHA-256:
  `77b1ece493f73bb8388ab1d295de335afa64ffc1f293832263707f142df8b76c`.

The image shows build `1.0.3385` at the known Training launch grid and gate,
unminimized and responsive. `FLIGHT MODE ACRO` is the vehicle control mode,
not the simulator competition mode. The exact simulator process has remained
unchanged since the operator-attested Training launch.

A fresh, hash-pinned F04-only console relay may transcribe this delegated
visual attestation. It must wait for and validate F04's canonical prechild
process proof, require the exact payload identity and window above, take a
fresh private capture of that same window, attach only to the proved wrapper
console, read exactly one generated `TRAINING <32hex>` prompt, re-prove the
window immediately before input, inject only that exact response plus carriage
return through `WriteConsoleInputW`, and verify the published attestation before
child evidence. Blind challenge echo, global input, a different process/window,
or any failure must produce no input and let the wrapper fail closed. Relay and
screenshot artifacts remain outside the six-file evidence root. The production
wrapper and its attached-console challenge contract remain byte-unchanged.

## Immutable F03 predecessor

`F03-A01` is terminal-invalid and poisoned. It reached the Training-attestation
checkpoint, but received no response within 30 seconds. It created no powered
child and sent no reset, arm, attitude target, thrust, or other powered command.
Preserve its private root and terminal evidence exactly; do not recover, clear,
or reuse it as F04 authority.

- predecessor root:
  `C:\Users\John\aigp-evidence\2026-07-22-package2-f03-powered-calibration-attempt`;
- attempt envelope SHA-256:
  `4082339925ad6da687161b05ade3bdbc3c1f7ce39c045a4e119acaf552193e4a`;
- attempt-invalid SHA-256:
  `b8df4b33b30f9a5b267b4650e6db91a55c6e9236b0f5c46651eb87f36e3a820f`;
- live-poison SHA-256:
  `185bdb4a3b970ab323e9a40f4fcbfb805134aca32ad9c463d973a78e298f1fb3`;
- process-final-proof SHA-256:
  `732ccdabf97283ee62cffb9b767e802fa349d8f90c608cc0d23eb2c8fcf72a88`;
  and
- live-lease SHA-256:
  `8908f04433eab6ae0794a0cf357e0364a7e1f9075cd9c56ed755beb792824f4b`.

## Frozen F04 identities

- session: `F04`;
- sole attempt: `F04-A01`;
- private evidence root:
  `C:\Users\John\aigp-evidence\2026-07-22-package2-f04-powered-calibration-attempt`;
- detached live worktree:
  `C:\Users\John\aigp-worktrees\wt-package2-f04-powered-calibration-attempt-live`;
- freeze ID:
  `vq2-package2-f04-powered-calibration-attempt-f04-a01-live-freeze`;
- freeze path: `<private-root>\live-freeze-F04-A01.json`;
- scheduled task: `AIGP-P2-F04-A01-Launch`;
- run ID: `F04-A01/reset-epoch-1/excitation-1`;
- split: `discovery_fit`;
- plan ID: `vq2-build3385-training-f04-excite-v1`;
- plan canonical-object SHA-256:
  `fae9d932e269e7de6513589d6f7bfd19862696d7222f1edad6eb3226292de773`;
  and
- plan canonical-file SHA-256:
  `52daf4306d8daba477464fbcd6292f2108509516c5ca0199b1895761b24c9f90`.

## Change boundary and verification

F04 changes task-local identities only. The exact waveform, 20 ms pacing, 245
ticks, 0.235 thrust, zero yaw, command bounds, watchdogs, reset-epoch proof,
countdown/GO timing, and cleanup contract remain unchanged from F03. The
process-image-specific 128 MiB hash ceiling also remains unchanged.

Run the focused identity assertions before accepting the candidate. Run the
canonical `test-vq2` suite at the promotion boundary. Before consumption,
derive and independently review a new six-file create-new bundle, publish its
live freeze last, and use only a pristine exact-commit detached worktree. F03
evidence, launcher, freeze, environment, and poison are not reusable.
