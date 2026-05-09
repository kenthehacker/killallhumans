# Autonomous AI Agent Iteration Guide

How to set up Claude Code (or any LLM agent) to recursively improve this drone racing codebase.

## Overview

The system is designed for a **benchmark-driven iteration loop**:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ Run Benchmark│────→│ Parse Metrics │────→│ Identify Gap │────→│ Edit Code    │
│ (headless)   │     │ (JSON stdout) │     │ (worst metric)│    │ (targeted fix)│
└──────┬───────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
       │                                                              │
       └──────────────────────────────────────────────────────────────┘
```

The agent runs `scripts/benchmark.py`, reads structured JSON output, finds the worst metric, edits the relevant module, and repeats.

## Headless vs GUI

**Headless is strongly preferred for LLM agents.** The benchmark runner:
- Requires no display server (no X11/Wayland/macOS GUI)
- Outputs machine-readable JSON to stdout
- Human-readable summary to stderr (suppressed with `--json-only`)
- Returns exit code 0 (pass) or 1 (fail)

The visual demo (`scripts/visual_demo.py`) requires OpenCV `imshow` and is useful for **human review** after the agent has made improvements. An agent should NOT try to parse visual output.

## Setup

### Prerequisites
```bash
cd ~/Personal/killallhumans
pip install -r requirements.txt
```

### Verify the benchmark works
```bash
python3 scripts/benchmark.py --mode unit 2>/dev/null | python3 -m json.tool
```

## The Iteration Loop in Detail

### Step 1: Run Benchmark
```bash
python3 scripts/benchmark.py --mode full 2>/dev/null
```

The `2>/dev/null` suppresses the human-readable summary, leaving clean JSON on stdout.

### Step 2: Parse the JSON

Key fields to inspect:

```jsonc
{
  "overall_passed": false,          // Goal: make this true
  "unit_tests": {
    "pass_rate": 1.0,               // Must be 1.0
    "tests": [
      {"name": "ekf_convergence", "passed": true, "time_ms": 12.3},
      {"name": "trajectory_generation", "passed": false, "error": "..."}
    ]
  },
  "simulation": {
    "gates_passed": 8,              // Goal: match total_gates
    "total_gates": 12,
    "gate_pass_rate": 0.667,
    "avg_tracking_error_m": 1.45,   // Goal: < 0.5m
    "p95_tracking_error_m": 3.2,
    "max_tracking_error_m": 4.8,
    "ekf_uncertainty_m": 0.234,
    "avg_loop_hz": 145.0,
    "crashed": false,
    "termination_reason": "time_limit",  // "race_complete" is ideal
    "per_gate_avg_error": {
      "gate-1": 0.5,
      "gate-5": 3.2               // ← this gate has problems
    },
    "threshold_failures": [
      "gate_pass_rate 67% < 50%"   // ← specific failure reasons
    ]
  }
}
```

### Step 3: Identify the Worst Issue

Priority order:
1. **Crashes** (`crashed: true`) — fix control/trajectory
2. **Unit test failures** — fix the specific module
3. **Gate pass failures** — trajectory doesn't reach gates
4. **High tracking error** — controller gains or trajectory quality
5. **High EKF uncertainty** — noise parameters
6. **Low loop Hz** — computational bottleneck

### Step 4: Edit the Right Module

| Issue | Module to Edit | What to Change |
|-------|----------------|----------------|
| Crash | `control/mpc_tracker.py` | Reduce `max_tilt_rad`, increase damping |
| Crash | `planning/trajectory_optimizer.py` | Reduce `max_velocity`, add altitude constraints |
| Missing gates | `gate_sequencing/sequencer.py` | Increase `pass_through_margin` |
| Missing gates | `planning/racing_line.py` | Reduce `max_lateral_offset` |
| High tracking error | `control/mpc_tracker.py` | Tune `kp_xy`, `kd_xy`, `kp_z`, `kd_z` |
| High tracking error | `planning/trajectory_optimizer.py` | Reduce `max_velocity`, increase `dt_sample` |
| EKF divergence | `estimation/ekf.py` | Tune `position_noise_std`, `velocity_noise_std` |
| Slow loop | `planning/trajectory_optimizer.py` | Increase `dt_sample`, pre-compute lookups |

### Step 5: Verify and Repeat

```bash
python3 scripts/benchmark.py --mode full 2>/dev/null
```

Compare the new JSON with the previous run. If the metric improved, continue to the next worst metric. If it regressed, revert and try a different approach.

## Using giga_chad_llm MCP for Parallel Research

The `giga_chad_llm` MCP server provides powerful tools for autonomous iteration:

### Investigation (before editing)
Use `giga_chad_llm_investigate` to understand how a module works before changing it:

```
Tool: mcp__giga_chad_llm__giga_chad_llm_investigate
Args:
  repo_path: /Users/kenichi.matsuo/Personal/killallhumans
  instruction: "How does the geometric tracker compute thrust and attitude commands? What are the gain sensitivities?"
```

### Scouting (find relevant files)
```
Tool: mcp__giga_chad_llm__giga_chad_llm_scout
Args:
  repo_path: /Users/kenichi.matsuo/Personal/killallhumans
  instruction: "Find all files related to trajectory optimization and speed profiling"
```

### RAG Query (search indexed knowledge)
```
Tool: mcp__giga_chad_llm__giga_chad_llm_query_rag
Args:
  query: "EKF covariance divergence fix"
```

### Full Autonomous Coding
For multi-file changes, spawn the full pipeline:
```
Tool: mcp__giga_chad_llm__giga_chad_llm_code
Args:
  repo_path: /Users/kenichi.matsuo/Personal/killallhumans
  instruction: "Reduce average tracking error from 1.5m to 0.5m by tuning controller gains and trajectory sampling"
  pipeline: "standard"
```

### Parallel Spawn (multiple independent tasks)
```
Tool: mcp__giga_chad_llm__giga_chad_llm_spawn
Args:
  tasks: [
    {"prompt": "Investigate why gate-5 has 3.2m tracking error", "cwd": "/Users/kenichi.matsuo/Personal/killallhumans"},
    {"prompt": "Research optimal PID gains for Crazyflie CF2X", "cwd": "/Users/kenichi.matsuo/Personal/killallhumans"}
  ]
```

## Automated Recurring Iteration

Use Claude Code's cron scheduling to run benchmarks on a timer:

```
Tool: CronCreate
Args:
  cron: "*/10 * * * *"
  prompt: "Run the benchmark at ~/Personal/killallhumans with `python3 scripts/benchmark.py --mode full 2>/dev/null`, parse the JSON output, identify the single worst metric, make a targeted code change to improve it, then re-run the benchmark to verify improvement. If the change regresses other metrics, revert it."
  recurring: true
```

This creates a 10-minute iteration loop that autonomously improves the codebase.

## Example: Full Autonomous Session

Here's what an ideal autonomous iteration session looks like:

```
Agent: Run benchmark
→ JSON shows: gate_pass_rate=0.67, avg_tracking_error=1.45m, gates 9-12 missed

Agent: Analyze — trajectory ends before reaching gates 9-12 (helix section)
→ Root cause: trajectory optimizer allocates too little time for the helix

Agent: Edit planning/trajectory_optimizer.py
→ Increase time allocation for segments with high curvature

Agent: Re-run benchmark
→ JSON shows: gate_pass_rate=0.83, avg_tracking_error=1.8m (slightly worse tracking)

Agent: Analyze — gate-7 through gate-10 have high error (helix section)
→ Controller can't track tight turns at current speed

Agent: Edit planning/racing_line.py
→ Reduce speed profile for high-curvature sections

Agent: Re-run benchmark
→ JSON shows: gate_pass_rate=1.0, avg_tracking_error=0.9m
→ All thresholds met!

Agent: Tighten thresholds, continue optimizing...
```

## File Quick Reference

| File | Purpose | Agent should edit? |
|------|---------|-------------------|
| `scripts/benchmark.py` | Headless benchmark runner | Only to add new metrics |
| `scripts/visual_demo.py` | GUI visualization | No (human review only) |
| `scripts/smoke_test.py` | Legacy smoke test | No (use benchmark.py) |
| `CLAUDE.md` | Agent instructions | Only to update thresholds |
| `estimation/ekf.py` | State estimation | Yes — tune noise params |
| `planning/trajectory_optimizer.py` | Trajectory generation | Yes — tune constraints |
| `planning/racing_line.py` | Racing line + speed | Yes — tune offsets/speeds |
| `control/mpc_tracker.py` | Flight controller | Yes — tune gains |
| `gate_sequencing/sequencer.py` | Gate detection logic | Yes — tune margins |
| `race_pipeline.py` | Pipeline orchestrator | Yes — integration changes |
| `sim_pybullet/` | Physics simulation | **NO — treat as ground truth** |

## Metrics History

To track improvement across iterations, save benchmark results:

```bash
python3 scripts/benchmark.py --mode full --json-only 2>/dev/null \
  >> benchmark_history.jsonl
```

Each line is a complete benchmark run. An agent can read the last N lines to track trends.
