"""
Adversarial tests for the synthetic benchmark (iter-001 A2).

Pre-fix: synthetic bench at `scripts/benchmark.py` only terminates when
`pos[2] < 0.05` (ground) or `pos[2] > 20.0` (ceiling). A drone that clips
a gate frame is treated as PASS.

Pre-fix: bench has hard-coded `inflection_start = int(2.0/dt)`,
`inflection_end = int(4.4/dt)`, `helix_start = int(7.4/dt)` — wall-clock
constants tuned to race_01.json's helix.

Post-fix:
  A7: synthetic loop checks `seq.last_crash` and `seq.is_disqualified` and
      breaks immediately with `crashed=True` / `dq=True`.
  A9: section boundaries derived from trajectory curvature; race_01-specific
      magic numbers live behind an explicit override block (no longer in
      the default path).
"""
from __future__ import annotations

import pathlib
import re

import pytest

from competition.aigp_geometry import (
    AIGP_GATE_BORDER_M,
    AIGP_GATE_INTERIOR_M,
)
from gate_sequencing.sequencer import (
    GateSequencer,
    GateSpec,
    SequencerConfig,
)


# ---------------------------------------------------------------------------
# I-2: synthetic frame-strut crash is terminal
# ---------------------------------------------------------------------------

def test_sequencer_records_geometric_frame_strike():
    """A segment whose crossing falls in [opening, outer_frame] -> crash event."""
    gate = GateSpec(
        gate_id="g1",
        position=(5.0, 0.0, -2.0),
        yaw=0.0,
        sequence_index=0,
    )
    seq = GateSequencer([gate])
    seq.start()
    # AIGP opening half-width = 0.75 m; outer half = 1.35 m.
    # Cross the plane at y = 1.0 m — solidly inside the outer frame but
    # outside the opening.
    seq.update((4.5, 1.0, -2.0))
    seq.update((5.5, 1.0, -2.0))
    assert seq.last_crash is not None
    assert seq.last_crash[0] == "g1"


def test_synthetic_bench_reads_seq_crash_signal():
    """`run_synthetic_benchmark` must propagate seq.last_crash to its result.

    Skipped until A2 lands the injectable-config seam in
    `run_synthetic_benchmark`. Today the bench reads race_01.json from
    disk and can't be steered into a deliberate crash from a unit test.
    """
    pytest.skip("requires bench config-injection seam (lands with A7)")


def test_synthetic_bench_reads_seq_dq_signal():
    """`run_synthetic_benchmark` must propagate seq.is_disqualified to result.

    Skipped until A2/A7 land the seam.
    """
    pytest.skip("requires bench config-injection seam (lands with A7)")


# ---------------------------------------------------------------------------
# I-3: no race_01 wall-clock magic constants in the default path
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BENCH_SRC = _REPO_ROOT / "scripts" / "benchmark.py"

# These literal patterns are the smoking gun: each is a wall-clock window
# expressed in seconds-from-race-start that only makes sense for race_01.
_FORBIDDEN_PATTERNS = [
    (r"int\(\s*2\.0\s*/\s*dt\s*\)", "inflection_start (race_01 helix start)"),
    (r"int\(\s*4\.4\s*/\s*dt\s*\)", "inflection_end (race_01 helix mid)"),
    (r"int\(\s*7\.4\s*/\s*dt\s*\)", "helix_start (race_01 helix late)"),
]


def _strip_legacy_override_blocks(src: str) -> str:
    """Remove text inside `# race_01_legacy_override: begin/end` blocks.

    Per the iter-001 synthesis: race_01-specific tunings are allowed only
    inside an explicit, clearly-marked override block. The default code path
    must be course-agnostic.
    """
    return re.sub(
        r"# race_01_legacy_override: begin.*?# race_01_legacy_override: end",
        "",
        src,
        flags=re.DOTALL,
    )


def test_no_race01_wall_clock_constants_in_default_path():
    """Magic ILC time constants must not appear outside an override block."""
    assert _BENCH_SRC.exists(), f"missing {_BENCH_SRC}"
    src = _strip_legacy_override_blocks(_BENCH_SRC.read_text())
    hits = [
        label
        for pat, label in _FORBIDDEN_PATTERNS
        if re.search(pat, src)
    ]
    assert not hits, (
        "course-specific magic constants still in default path: "
        + ", ".join(hits)
        + "\nWrap them in `# race_01_legacy_override: begin/end` or replace "
          "with derived section boundaries (iter-001 A9)."
    )


# ---------------------------------------------------------------------------
# I-4: convergence threshold + momentum gamma must be config-loaded, not literal
# ---------------------------------------------------------------------------

_FORBIDDEN_HYPERPARAM_PATTERNS = [
    (r"convergence_threshold\s*=\s*0\.0005", "ILC convergence_threshold literal"),
    (r"momentum_gamma\s*=\s*0\.2", "ILC momentum_gamma literal"),
]


def test_ilc_hyperparameters_are_not_inline_literals():
    """Pre-fix: `convergence_threshold=0.0005`, `momentum_gamma=0.2` in source.

    Post-fix (A9): both load from `config/ilc_defaults.yaml`.
    """
    src = _strip_legacy_override_blocks(_BENCH_SRC.read_text())
    hits = [
        label
        for pat, label in _FORBIDDEN_HYPERPARAM_PATTERNS
        if re.search(pat, src)
    ]
    assert not hits, (
        "ILC hyperparameter literals still in default path: "
        + ", ".join(hits)
        + "\nMove to config/ilc_defaults.yaml (iter-001 A9)."
    )
