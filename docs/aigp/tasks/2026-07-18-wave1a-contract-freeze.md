# Wave 1A shared-contract freeze task manifest

- Task ID: `vq2-wave1a-contract-freeze`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `integration_pending`
- Objective: freeze the build-3385 timestamp, frame identity, latency,
  observation, relative-state, command-proposal, and tier-evidence interfaces
  needed by the three Wave 1 implementation branches.
- Non-goals: no FlightSim connection; no arm, reset, flight target, or powered
  stage; no gate-1 steering; no replacement of the safety supervisor; no claim
  that replay or synthetic evaluation is official-simulator evidence.
- Starting main commit: `1e5ec01f16a102f14e2a5b5cf9ab70c5061d5cf1`
- Branch: `wave1a-contract-freeze`
- Worktree: `C:\Users\John\aigp-worktrees\wt-contract-freeze`
- Integration owner: `/root`
- Lease owner: `/root`
- Heartbeat date: `2026-07-18`
- Simulator access: `none`
- Artifact root: `C:\Users\John\aigp-worktrees\artifacts\wave1a-contract-freeze`
- Cache root: `C:\Users\John\aigp-worktrees\.artifact-cache`
- Owned interfaces: VQ2 contracts and compatibility adapters, contract tests,
  VQ2 test policy/count, trusted-file manifest, and contract documentation.
- Serialized integration-hot files: replay/observation schema, promotion
  evidence schema, and trusted-file manifest are changed only by the integration
  owner for this task.
- Initial interface versions: `/1` for each newly frozen contract.
- Required direct tests: contract/adaptor tests and promotion contract tests.
- Required candidate gate: `scripts\dev.cmd test-vq2`.
- Required integration gates: `scripts\dev.cmd test-fast` and the non-live
  scheduler/promotion checks affected by trusted-manifest changes.
- Acceptance: strict immutable schemas reject non-finite or ambiguous timing,
  frame, covariance, authority, and evidence data; adapters preserve the proved
  Gate 0 command and capture shapes; T0/T1 cannot claim closed-loop authority;
  T2-T4 remain explicitly synthetic and nonpowered; main is clean after the
  committed candidate is integrated.

## Frozen interface inventory

- Contract implementation: `competition/vq2_contracts.py`.
- Legacy bbox compatibility: `gate_detection/src/vq2_observation_adapter.py`.
- Tier-domain contract: `aigp_loop/evidence.py`.
- Schemas: frame identity, frame timing, prediction time, latency event, gate
  authority epoch, gate observation, relative gate state, command proposal,
  supervisor-approved command, and tier evidence scope, each at `/1`.
- Cross-object validators bind frame sequences, observation-to-state sources,
  state sequences, proposal-to-state sources, approval sequences, latency
  traces, and approval-to-send evidence.
- Wire changes to fields, units, frames, enum/mask meanings, ordering, authority,
  or safety semantics require a new schema version. A stricter check may remain
  `/1` only when it enforces an invariant already stated by the frozen reference.
- Exact reference: `docs/aigp/vq2_contracts.md`.

## Review and validation record

- Multiple independent adversarial passes covered camera-epoch relabeling,
  source/update replay, authority regression, clipped fitted geometry, legacy
  pixel representability, proposal-to-transport bypass, approval amplification,
  stale/superseded control ticks, tier-domain claim aliases, duplicate evidence,
  and resumed promotion decisions.
- Superseded pre-acceptance runs include 87 focused contract/evidence tests,
  281 scheduler/promotion tests with 12 expected skips, and a 350-test VQ2
  policy run. The accepted contract policy now collects exactly 356 tests.
- Simulator access remained `none`; no preflight, process launch, network
  connection, reset, arm/disarm, target, or powered command occurred.

- Trusted manifest: 116 entries; builder identity
  `d6e4cd31177281fe9010eeeeb7df1667c248c464a45444d692d5a8225a6dc033`;
  canonical file SHA-256
  `45514d8edaad2874c79a95946ff4b7632d5b4ada7a0294bf1f08c3f730701253`.
  Review and canonical copies matched byte-for-byte. Relative to the 113-entry
  starting manifest, the three new files under already trusted roots entered
  the inventory and ten existing entries changed hash; no trust root expanded.
- Candidate acceptance:
  - affected promotion/contract group: `450 passed, 12 skipped`;
  - final evidence/promotion/scheduler regression group after closeout audit:
    `183 passed, 6 skipped`;
  - direct contract suite: `53 passed`;
  - canonical and isolated-cache VQ2 gates: `356 passed` each;
  - final `test-fast`: `1448 passed, 20 skipped, 42 deselected`;
  - pre-closeout `test-unit`: `1445 passed, 20 skipped, 42 deselected`;
  - `git diff --check`: clean.
- Integration/post-merge verification: pending.
- Implementation commits:
  `fd51af3c587e7c3431719b79c1713344e7cc6d6f` (`Freeze VQ2 Wave 1 shared
  contracts`) and `a6782cd9dcc34aee94e0f064021399985e0f6839`
  (`Harden Wave 1A evidence scope`).
- Result/failure provenance: all accepted evidence was produced in the named
  offline worktree with the repository development environment and external
  bytecode caches. Three ignored source-tree cache directories created during
  adversarial review were explicitly removed before isolated acceptance; the
  accepted tree contained no source `__pycache__`, `.pyc`, or `.pyo` evidence.
  No powered or official-simulator claim is made.
