# Wave 3 offline merger dogfood task manifest

- Task ID: `vq2-wave3-merger-dogfood`.
- Parent: `2026-07-18-vq2-execution-plan-handoff`.
- State: `post_merge_verified`.
- Objective: exercise the real `SingleMerger` positive fast-forward path with
  a fully promotion-valid synthetic T0-T4 ledger chain and an exact descendant
  commit in a disposable local Git repository.
- Starting commit: `a5c11ab9924b1f250948e9ebdee459427489c924`.
- Branch: `wave3-merger-dogfood`.
- Worktree: `C:\Users\John\aigp-worktrees\wt-wave3-merger-dogfood`.
- Integration owner: `/root`.
- Simulator access: `none`.
- Owned files: `tests/test_aigp_loop_scheduler.py` and this task record only.
- Excluded files: production scheduler/ledger/promotion code, shared policy,
  trusted manifest, shared handoffs, runner, transport, supervisor, simulator
  tooling, and captures.
- Safety boundary: the fixture publishes unit-only synthetic ledger evidence in
  a temporary repository. It does not run evaluators, access FlightSim or a
  network, use a replay corpus, send a command, or claim official-simulator,
  powered, or live evidence.
- Required behavior:
  - `SingleMerger` validates an exact completed T0-T4 chain;
  - the target must be clean and the candidate must be a descendant;
  - the merge advances to the exact recorded candidate commit with clean Git
    status;
  - checkpoint evidence remains unchanged;
  - T0 remains affected-test evidence, T1 remains causal open-loop replay
    evidence, and T2-T4 remain deterministic synthetic nonpowered evidence;
  - dirty and divergent merge targets fail without advancing and release the
    singleton orchestration lease.
- Operational boundary: a genuine clean-candidate T0-T4 scheduler run remains
  blocked by the absent approved replay corpus, production replay processor,
  and administrator-owned pinned isolation wrapper. The checked-in promotion
  command and identity documents are templates and do not satisfy those
  prerequisites.
- Campaign boundary: campaign/T5 planning and all powered execution are out of
  scope. This unit dogfood neither derives nor uses a powered authorization
  phrase.
- Required gates: the new merger tests, the complete scheduler test module,
  `scripts\dev.cmd test-vq2`, `scripts\dev.cmd test-fast`, `git diff --check`,
  and a clean committed worktree.
- Candidate evidence:
  - positive exact fast-forward plus dirty/divergent target cases: `3 passed`;
  - complete scheduler test module: `62 passed, 6 skipped`;
  - canonical `test-vq2`: `743 passed`;
  - repository `test-fast`: `1,838 passed, 20 skipped, 42 deselected`;
  - `git diff --check`: clean before commit.
- Behavioral commit:
  `fe59e42a27ad84a4d9a8b10d5311a4b1bc44ad69`.
- Integration-owned review:
  - the three new merger cases independently reproduced: `3 passed`;
  - trusted-manifest inventory remained exactly `123 -> 123`, with no added or
    removed files and only `tests/test_aigp_loop_scheduler.py` changing;
  - reviewed manifest semantic identity:
    `60680aec1f26b3661576b65221ab4aeba4fab5df8959e46076dd8f99fce8fe41`;
  - reviewed/tracked manifest file SHA-256:
    `0658ef17b864ce60312917aca208cd66263144f44d8fcc86ba4c37be1ebb2be5`.
  - integrated candidate commit:
    `ab62cde9464442e4b448f293ba8efd31ad601c27`;
  - main fast-forward: `a5c11ab9924b1f250948e9ebdee459427489c924`
    to the integrated candidate;
  - post-merge main `test-vq2`: `743 passed` in `30.05s`, followed by an
    empty tracked Git status.
- Evidence limits: all promotion rows and evaluator results are synthetic unit
  fixtures. They exercise validation and merge mechanics only; they are not a
  scheduler-run candidate, replay result, simulator result, or promotion of
  this repository branch.
- Integration note: this task intentionally does not update the shared trusted
  manifest or policy. The integration owner must review the changed test-file
  hash at promotion time.
