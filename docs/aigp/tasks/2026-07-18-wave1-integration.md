# Wave 1 offline integration task manifest

- Task ID: `vq2-wave1-offline-integration`
- Parent: `2026-07-18-vq2-execution-plan-handoff`
- State: `integration_pending`
- Objective: integrate the runtime-timing, clipped-geometry, and relative-state
  branches on the frozen Wave 1A `/1` contracts, preserve Gate 0 behavior, and
  produce green offline evidence for the M1/M2/M4 prerequisites.
- Non-goals: no FlightSim process launch or connection; no preflight, external
  network probe, reset, arm/disarm, flight target, or powered stage; no gate passage;
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

## Candidate and integration record

- Runtime timing: implementation
  `11c35a5431587b892fd3950c7a589ce3ff312652`, evidence tip
  `9114eb7af4ea483902b59bffd18270f480470e5f`, merge
  `08dd5ad47127f043c5337bc13b55bd9f3490df52`.
- Gate geometry: implementation
  `0db37d89c2045bde8461622c4ed22543a960442c`, evidence tip
  `7069b5ebc511b532b98884b9fd1d4266c9a11560`, merge
  `1c0450b584f10703efafffdbf1b224c78e5974fa`.
- Relative estimation: implementation/evidence tip
  `4a9833abbbcdd1d88b1f2e4d378fcf14c39de4d2`, merge
  `7d1d5b5ffe59a17e57abef5402cbffb19537b9b6`.
- Cross-workstream test, exact VQ2 policy, and trusted-manifest candidate:
  `361d0060f16dbaec753de00ba491f1a085707eb1`.
- Frozen `/1` contract meanings were unchanged. The narrow runner correction
  identifies distinct frames by `(generation, frame_id)` rather than the opaque
  camera source-time token; no safety bounds or powered-stage semantics changed.

## Accepted offline evidence

- Runtime branch: 92 direct tests; 235 broader competition/vision tests;
  381 VQ2 tests; pre-final fast result 1,472 passed, 20 skipped, 42 deselected.
- Geometry branch: 79 direct tests; 377 VQ2 tests; fast result 1,469 passed,
  20 skipped, 42 deselected.
- Estimator branch: 14 direct estimator tests; 53 contract tests; 370 VQ2
  tests; fast/unit result 1,462 passed, 20 skipped, 42 deselected.
- Integrated affected group: 187 passed.
- Canonical and isolated candidate VQ2 policy: exactly 418 passed each.
- Integrated `test-fast` and `test-unit`: 1,510 passed, 20 skipped,
  42 deselected each.
- Trusted manifest: 119 entries; semantic identity
  `fd1d09e16c34dd3c77fb45877102dda56ca1da888b8b6c3cf5bf1408ffe0d4b8`;
  canonical file SHA-256
  `44f985274eb41a5c6d12b6fa17e4d553facb14f8b854901d084eabc99c88af5e`.
  Review and canonical copies matched byte-for-byte. Relative to the 116-entry
  baseline, three intended tests under already trusted roots were added, four
  existing entries changed hash, no entry was removed, and no trust root
  expanded.
- `git diff --check`: clean. The ignored `.pytest_cache` produced by accepted
  tests was resolved to the exact integration worktree and removed. No source
  `__pycache__`, `.pyc`, or `.pyo` remained.
- Simulator access remained `none`: no FlightSim process launch or connection,
  preflight, external network access, reset, arm/disarm, target, or powered
  command occurred. Offline tests may use local Python workers and loopback
  test sockets.

## Residual limitations

- M1 still needs production event wiring through actuator/gyro response,
  scheduler integration behind the supervisor/transport seam, and explicitly
  authorized simulator timing/delay measurements.
- M2 still needs approved Gate 0 and recorded top-clipped Gate 1 replay
  acceptance, active/shadow tracking, and crossing-residue isolation.
- M4 still needs IMU derotation, measured command-effect prediction, replay p95
  comparison, the pure controller, and runtime/shadow evidence.
- Main integration, post-merge `test-vq2`, clean-main proof, and the final record
  commit remain pending.
