# Wave 1 offline integration task manifest

- Task ID: `vq2-wave1-offline-integration`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `active`
- Objective: integrate the runtime-timing, clipped-geometry, and relative-state
  branches on the frozen Wave 1A `/1` contracts, preserve Gate 0 behavior, and
  produce green offline evidence for the M1/M2/M4 prerequisites.
- Non-goals: no FlightSim connection; no preflight, process launch, network
  probe, reset, arm/disarm, flight target, or powered stage; no gate passage;
  no production T1/private-corpus claim; no controller transport authority.
- Starting main commit: `3de33c3a568bc86638d9d7ac4dac6124f1e15397`.
- Branch: `wave1-integration`.
- Worktree: `C:\Users\John\aigp-worktrees\wt-wave1-integration`.
- Integration owner and lease owner: `/root`.
- Heartbeat date: `2026-07-18`.
- Simulator access: `none`.
- Candidate branches: `wave1-runtime-timing`, `wave1-gate-geometry`, and
  `wave1-relative-estimation`.
- Interface contract: `docs/aigp/vq2_contracts.md`, generation `/1`; wire
  meanings may not change in this integration task.
- Serialized ownership: integration conflict resolution, shared handoff/task
  records, test policy/count if collection changes, and trusted-manifest review.
- Required direct evidence: each branch's affected tests and adversarial review.
- Required integration evidence: combined affected tests, `test-vq2`,
  `test-fast`, clean diff/worktree, and post-merge `test-vq2` on main.
- Promotion claim: offline tests only unless a separately configured, exact
  tier evaluator produces valid evidence; synthetic results are never relabeled
  as official-simulator or powered evidence.
- Final commits, counts, artifact hashes, integration record, and residual
  limitations: pending.
