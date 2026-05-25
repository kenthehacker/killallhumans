# Adversarial review: iter-009i F9 fix (commit `b926734`)

## Verdict

**The implementation captures the unanimous *diagnosis* and the dominant *architectural* recommendation from the four `f9_*.md` agents, but it does not implement the full research consensus as a layered defense.** It ships a single, well-scoped mechanism—**path–velocity decoupling** via a dedicated `select_velocity_mps` for the sim oracle’s `TrajectoryOptimizer`—while leaving **safety-blind scoring**, **multi-fidelity / multi-scenario robustness**, **`_kinematic_eval` consistency**, and **Opus’s T1 (Heilmeier QP)** explicitly out of scope. For F9 regression closure that is proportionate; for “full 4-agent consensus” as a checklist, **critical pieces are missing**.

## What `b926734` actually delivers

- `RacingLineConfig.select_velocity_mps` (default 15) drives `_select_by_sim`’s min-snap build; `max_velocity_mps` is documented as **informational** for callers (execution hint).
- `scripts/benchmark.py` passes auto-derived `max_velocity_mps` **and** locks `select_velocity_mps=15`, re-enabling the deferred wire-up without re-coupling basin choice to execution speed.
- Cache key includes `select_velocity_mps`, **not** `max_velocity_mps`—intentionally so execution-speed changes do not churn the cache (this **diverges** from GPT‑5.5/Opus T2 text that wanted velocity in the key for aliasing diagnosis; here the design choice is “decouple, don’t fragment”).
- New tests assert **bit-identical geometry** across four `max_velocity_mps` values when `select_velocity_mps` is fixed, plus a diagnostic that changing `select_velocity_mps` *can* move geometry.

## Coverage vs. each agent note

| Consensus theme | In `b926734`? |
|-----------------|---------------|
| Min-snap / BO scorer is **velocity-coupled**; that causes basin switching | Yes — root cause aligned across Opus, GPT‑5.5, Composer, Gemini. |
| **Decouple** geometry selection from execution velocity (TUM / Heilmeier-style story) | Yes — fixed reference speed for oracle trajectories is exactly Composer’s “two-pass / freeze `V_ref`” primary path and matches Gemini’s “evaluate at high nominal velocity” intent (implementation uses explicit `select_velocity_mps` rather than `max(v,15)`). |
| **Safety / feasibility** before ranking (plan validator, clearance, sequencer replay) — Opus **T2**, GPT‑5.5 feasibility phase | **No** — `_select_by_sim` still never calls `planning/plan_validator.py`; the “safety-blind scalarizer” half of Opus’s one-liner diagnosis remains true. |
| **Multi-fidelity / dual-velocity scoring** (evaluate at `v_auto` and `v_legacy`, robust aggregate) — GPT‑5.5, Gemini §2 | **No** — single-fidelity oracle at `select_velocity_mps` only. |
| **`_kinematic_eval` hardcoded `max_speed = 15.0`** misaligned with trajectory cap — Composer secondary | **No** — grep confirms `max_speed = 15.0` is still fixed inside `_kinematic_eval`; the oracle remains internally inconsistent if `select_velocity_mps` were ever changed away from 15 without updating that clamp. |
| **Heilmeier minimum-curvature QP** replacing L-BFGS + sim — Opus **T1** | **No** (explicitly larger than iter‑009i). |
| **Tests**: Opus proposed “offsets feasible under plan_validator for v∈{5,8,12,15}”, not identical geometry | **Partially** — shipped tests codify **invariance** (stronger than feasibility) for the decoupling property, but **do not** assert validator-safe lines across velocities or strut clearance. |

## Completeness gap (adversarial)

1. **Opus T2 is absent.** The crash mechanism Opus describes is “polynomial is unsafe while tracker error is small.” Decoupling execution speed from the *ranking* min-snap shape mitigates the *basin-shift* pathway; it does **not** add the defensive filter Opus scheduled as iter‑010. A future change to `select_velocity_mps`, scoring weights, or L-BFGS seeds could still pick a validator-failing line with no guardrail.

2. **GPT‑5.5’s multi-scenario scorer is absent.** The consensus was not only “don’t couple” but “if you keep the sim scorer, make it robust (feasibility first, worst-of-two-velocities, etc.).” iter‑009i chose the cheaper branch: **remove the coupling** instead of **hardening the coupled oracle**. That is a valid engineering tradeoff, not full consensus coverage.

3. **`_kinematic_eval` drift remains.** Composer flagged this as necessary for an honest velocity-aligned scorer. iter‑009i sidesteps alignment by freezing selection at 15 m/s, but the dead constant still contradicts the stated config surface if anyone tunes `select_velocity_mps`.

4. **Cache philosophy splits the swarm.** GPT‑5.5 wanted velocity-aware keys to stop aliasing; iter‑009i **deliberately** omits execution velocity from the key once selection is decoupled. Coherent, but not “everything every agent wrote.”

## Bottom line

iter‑009i is a **surgical, literature-consistent** implementation of the **path–velocity decomposition** slice all four agents endorsed, plus regression tests and bench wiring. It is **not** a complete rollup of every ranked technique in the notes (T2 feasibility, multi-fidelity scoring, kinematic-eval honesty, Heilmeier rewrite). Label it **MVP consensus on the binding F9 mechanism**, not **full defensive depth** from the research packet.
