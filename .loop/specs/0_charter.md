# AI Grand Prix VQ1 Loop — Charter

## Mission
Make the killallhumans drone-racing stack production-ready for AI Grand Prix
Virtual Qualifier 1. The stack must:
1. **Generalize** to any course the competition issues (positions and counts unknown until release).
2. **Honestly score itself** — crashes are fail, gates must be passed in the correct order, no false-positive passes.
3. **Run on the actual competition surface**: MAVLink2 UDP against a 280×280×160mm drone in DCL Sim — not on the in-repo PyBullet harness alone.
4. **Add lightweight ML where it materially helps**, on top of a deterministic baseline.

## Hard constraints (do not violate)
- **No course-specific magic numbers.** Anything tuned to `race_01.json` (e.g. `inflection_start=2.0/dt`, `helix_start=7.4/dt` in `scripts/benchmark.py:322`) must be parameterised, derived from track geometry, or deleted.
- **No `giga_chad_llm_investigate` / `giga_chad_llm_code` / `giga_chad_llm_plan` / `giga_chad_llm_scout` / `giga_chad_llm_spawn*` calls.** Read-only RAG / code-graph / widget / proto tools are OK if needed. (Per user direction in the /loop invocation.)
- **Crashes are terminal.** A crash should end the run with a fail metric — never silently continue.
- **Gate order is enforced.** Skipping a gate then coming back to it later is a fail, not a recovery.
- **Testbench is suspect.** If the synthetic / PyBullet bench says PASS, do not believe it without an adversarial test that tries to trip it.
- **Drone-model mismatch is acknowledged.** The PyBullet drone is not the competition drone. Either calibrate or validate via MAVLink against the DCL sim binary the competition ships.
- Work in this worktree (`~/Personal/killallhumans-aigp-vq1`, branch `aigp-vq1-loop`). Never touch other worktrees.

## Loop structure (drives ~300 iterations max)
1. **Plan round** (3 cursor-agents in parallel: Opus 4.7 max-thinking, GPT-5.5 xhigh, Composer 2.5).
2. **Resolve plan** (claude main loop synthesises into one coherent plan, resolves incompat).
3. **Implement** (apply changes via Edit/Write; commit on green).
4. **Test** (run synthetic + PyBullet + any new adversarial harnesses).
5. **Review round** (8 cursor-agents: 1 Opus 4.7 max-thinking + 2 GPT-5.5 xhigh + 5 Composer 2.5, adversarial).
6. **Apply fixes** from the review; back to plan or test.

State machine lives in `.loop/state/iter_state.json`.

## Optimization-stall protocol
When the loop is **stuck on an optimization metric** (race time won't drop,
tracking error plateaus, ML accuracy stuck, etc.) AND the stall is **not caused
by a bug** (tests green, no crash, no obviously broken code path), do NOT keep
grinding tuning knobs. Spin up a 4-model **research swarm** via `cursor_spawn`:

- `gemini-3.1-pro`
- latest Codex (or `gpt-5.5-extra-high` if no current Codex slug; else `gpt-5.3-codex-xhigh`)
- `claude-opus-4-7-thinking-max`
- `composer-2.5`

Each agent gets:
- the stuck metric + numbers
- what's been tried already
- "list research-proven techniques (papers / conferences / year) that could unblock this metric. Top 3 candidates with cost (S/M/L) and expected gain."

Output: `.loop/iter_NNN_research/<model>.md` per agent.
Main loop then synthesises into one ranked candidate list, picks one,
implements it, and re-runs the test matrix.

If tests are **failing** or behaviour is broken, debug first — research swarm
comes *after* the bug is closed.

## Reference files
- AIGP spec distilled: `.loop/specs/1_aigp_spec_distill.md`
- Confirmed code issues: `.loop/specs/2_known_issues.md`
- Iteration brief for agents: `.loop/specs/3_planning_brief.md`
- The original PDF: `/Users/kenichi.matsuo/Desktop/260508_Technical_Spec_0002.pdf`
- The competition update page: https://www.theaigrandprix.com/previousupdates/
