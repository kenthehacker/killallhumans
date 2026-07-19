# Autonomous iteration for the VQ2 stack

This guide replaces the legacy recurring ten-minute VQ1 benchmark prompt. The
current target is FlightSim build 3385 in Training mode, using vision,
`HIGHRES_IMU`, and race status without usable pose or gate-map data. The live
state and all safety invariants remain authoritative in
`aigp/2026-07-18-vq2-handoff.md`.

## Promotion ladder

Work from the cheapest valid evidence upward:

| Tier | Trigger | Command/evidence |
|---|---|---|
| T0 | Every edit | `.\scripts\dev.cmd test-target <affected paths>` plus import checks |
| T1 | Every accepted candidate | `.\scripts\dev.cmd test-vq2` plus the configured private golden-replay policy gate |
| T2 | Relevant control/planning candidate | One warm prepared synthetic simulation |
| T3 | Promising simulation candidate | Changed-domain track subset |
| T4 | Pre-merge/nightly | `test-benchmark` and `test-full-non-live` outside the inner loop |
| T5 | Explicit human promotion | One bounded, authorized official-simulator trial |

The default `pytest`/`test-fast` selection excludes `slow`, `benchmark`, and
`live`, uses strict marker registration, and has a hard timeout. The VQ2 suite
is deliberately fast; use the current `test-vq2` result rather than a copied
test-count baseline, because safety coverage is expected to grow.

Completion and safety are lexicographic gates:

1. no collision, disqualification, stale-stream flight, unsafe command, or
   cleanup failure;
2. correct gate sequence and completion reliability;
3. centering/stability margin;
4. race time.

Never collapse these into a score that trades safety failure for speed. Use
successive halving: cheap evidence for every proposal, broader deterministic
evaluation only for survivors, and live evaluation only for finalists.

## Per-candidate loop

1. State one bounded hypothesis and the directly affected modules.
2. Record the starting commit/diff and resolved configuration.
3. Run T0, make the smallest coherent change, and rerun T0.
4. Run T1 before accepting the candidate.
5. Promote to synthetic tiers only when the changed domain needs closed-loop
   dynamics that replay/unit tests cannot supply.
6. Compare safety/completion first and timing only after validity matches.
7. Keep or revert the candidate explicitly; do not stack an unexplained
   regression under another experiment.

Synthetic/PyBullet matrices model legacy modules and are not universal VQ2
truth. Replay can validate perception, estimation, and open-loop command
generation, but it cannot prove a closed loop whose commands change future
images.

## Official-simulator boundary

`.\scripts\dev.cmd preflight` is passive. Every powered stage requires explicit
user authorization and must prove a fresh reset epoch, observe countdown and
GO, arm only on fresh authoritative state, enforce all stream/attitude/rate/
collision watchdogs, pace bounded commands, then disarm/reset and prove
cleanup. A failed cleanup fails the trial. Generic test tasks never power the
simulator.

## Durable records

Do not append to `benchmark_history.jsonl`: it is a preserved stream of
pretty-printed historical objects rather than valid JSONL, and the old
PyBullet-skipped `overall_passed` values are not a trustworthy comparison
series. New evaluators must be versioned; structured results belong in the
implemented resumable SQLite trial ledger. The replay format, policy gate,
cohort successive-halving commands, isolated-worktree scheduler, and explicit
T5 authorization boundary are documented in
[`aigp/durable_improvement_loop.md`](aigp/durable_improvement_loop.md).

For each promoted result record code/diff hash, fully resolved config and hash,
environment fingerprint, evaluator version, seed, cache state, phase/end-to-end
timings, safety/completion fields, and failure output. Preserve historical
`.loop`/`.research_loop` files but omit them from routine search/indexing.

Keep new full decoded-frame replays private and ignored (or in an approved
external artifact store); preserve the repository's existing historical
capture evidence. Before submission, use
`.\scripts\dev.cmd sbom` and complete the human-reviewed disclosure template in
`disclosures/ai-and-tools-template.md`; neither artifact alone proves
license or rules compliance.
