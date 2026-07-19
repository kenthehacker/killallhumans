# Wave 2 offline integration task manifest

- Task ID: `vq2-wave2-offline-integration`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `active`
- Objective: integrate a pure predictive controller, offline system-
  identification tooling, and mapless guidance/state logic on the post-Wave 1
  frozen `/1` contracts, with deterministic evidence and no transport authority.
- Non-goals: no FlightSim process launch or connection; no external network
  access; no private capture ingestion; no preflight, reset, arm/disarm, flight
  target, command send, gate passage, or powered stage; no runner or safety-
  supervisor behavior change; no claim of official-simulator evidence.
- Starting main commit: `e9a416714b01c2845786a0a22b168a9037f379ec`.
- Branch: `wave2-integration`.
- Worktree: `C:\Users\John\aigp-worktrees\wt-wave2-integration`.
- Integration owner and lease owner: `/root`.
- Heartbeat date: `2026-07-18`.
- Simulator access: `none`.
- Candidate branches: `wave2-predictive-control`, `wave2-system-id`, and
  `wave2-mapless-guidance`.
- Frozen interface: `docs/aigp/vq2_contracts.md`, generation `/1`; downstream
  modules consume exact contracts without changing wire fields or meanings.
- Authority seam: `RelativeGateStateV1 -> guidance objective -> pure controller
  -> CommandProposalV1`; proposals have no send authority and transport remains
  unreachable from every Wave 2 candidate.
- Serialized ownership: integration conflict resolution, cross-workstream tests,
  shared handoffs/task states, exact VQ2 policy/count, and trusted-manifest review.
- Required branch evidence: direct/adversarial tests, `test-vq2`, `test-fast`,
  clean committed worktree, exact no-simulator provenance, and residual limits.
- Required integration evidence: combined affected tests, exact VQ2 policy,
  isolated candidate VQ2, `test-fast`, trusted-manifest review if discovery
  changes, clean diff/worktree, and post-merge `test-vq2` on main.
- Promotion claim: offline unit/synthetic evidence only. A pure proposal or
  experiment definition is never relabeled as a supervised command, actuator
  response, replay result, official-simulator result, or powered evidence.
- Final commits, counts, hashes, integration record, and remaining authorization
  or data blockers: pending.
