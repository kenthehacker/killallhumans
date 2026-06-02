"""
Iter-031 acceptance gate: the trained residual must actually help the
matrix (not just be net-neutral like iter-027).

This test runs the matrix twice — once with `use_residual=False`
(baseline) and once with `use_residual=True` pointing at the committed
weights file — and asserts:

  - **≥5 of 6** non-figure8 tracks IMPROVE (tracking err drops > 1e-4 m).
  - **No track REGRESSES by more than 1%** (err on / err base > 1.01).
  - **figure8 sim_passed remains True under residual ON** (the iter-028
    coplanar-DQ fix must not regress).

If the committed weights file is missing, the test is SKIPPED — so a
clean checkout without trained weights still passes the suite, but the
matrix-7/7 guarantee is upheld by `test_benchmark_matrix.py`.

Total runtime: ~30 s on M-class CPU (2× matrix at duration=20 s, 7
tracks each).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark import run_synthetic_benchmark
from scripts.benchmark_matrix import _list_configs

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEIGHTS = _REPO_ROOT / "control" / "residual_weights.npz"

# Tracks the residual is trained on (figure8 excluded — see
# scripts/collect_residual_dataset.py::_SKIP).
_TRAINING_TRACKS = (
    "race_01", "slalom", "grand_tour", "straight_hairpin",
    "vertical_cliff", "aigp_default",
)

# Acceptance thresholds.
# Iter-031 (post review):
#   - 1e-4 m absolute threshold is at sim noise floor (Opus M1 / Codex
#     MAJOR). Switched to relative % matching the regress tolerance.
#   - Trainer's CL scoring uses the same thresholds for selection (see
#     scripts/train_tracker_residual.py closed-loop block) so a checkpoint
#     that wins selection also passes this gate.
_MIN_IMPROVED = 5                  # ≥5/6 training tracks must improve
_IMPROVE_FRACTION = 0.01           # err drop > 1% (relative) counts as improvement
_REGRESS_TOLERANCE_FRAC = 1.01     # err_on > err_base * this counts as regression
_MATRIX_DURATION = 20.0            # per-track sim seconds


def _run_matrix(overrides: dict | None) -> dict:
    """Run every config and return {track_name: result_dict}."""
    results = {}
    for cfg_path in _list_configs():
        with open(cfg_path) as f:
            cfg = json.load(f)
        r = run_synthetic_benchmark(
            duration=_MATRIX_DURATION,
            config=cfg,
            tracker_config_overrides=overrides,
        )
        results[cfg_path.stem] = r
    return results


@pytest.mark.skipif(
    not _WEIGHTS.exists(),
    reason=(
        f"trained weights at {_WEIGHTS} not committed yet — run "
        "scripts/train_tracker_residual.py to produce them"
    ),
)
def test_residual_beats_baseline_on_five_of_six_tracks():
    """The committed residual weights must improve tracking on ≥5 of 6
    training tracks, regress no track by >1%, and not break figure8."""
    base = _run_matrix(None)
    on = _run_matrix({
        "use_residual": True,
        "residual_weights_path": str(_WEIGHTS),
    })

    improved: list[tuple[str, float, float, float]] = []
    regressed: list[tuple[str, float, float, float]] = []
    rows: list[str] = []
    for name in _TRAINING_TRACKS:
        b = base[name]
        o = on[name]
        if b.get("skipped") or o.get("skipped"):
            continue
        b_err = b.get("avg_tracking_error_m")
        o_err = o.get("avg_tracking_error_m")
        if b_err is None or o_err is None:
            continue
        ratio = o_err / max(b_err, 1e-9)
        diff_pct = 100.0 * (b_err - o_err) / max(b_err, 1e-9)
        rows.append(
            f"  {name:18s} base={b_err:.4f} on={o_err:.4f} diff={diff_pct:+.2f}%"
        )
        if ratio < 1.0 - _IMPROVE_FRACTION:
            improved.append((name, b_err, o_err, diff_pct))
        if ratio > _REGRESS_TOLERANCE_FRAC:
            regressed.append((name, b_err, o_err, diff_pct))

    debug_msg = "\n".join(rows)

    # Hard constraints.
    assert on["figure8"]["sim_passed"], (
        "figure8 must still pass under residual ON (iter-028 invariant)\n"
        f"{debug_msg}"
    )
    assert len(improved) >= _MIN_IMPROVED, (
        f"only {len(improved)}/6 tracks improved (need >= {_MIN_IMPROVED}):\n"
        f"{debug_msg}\nimproved: {[r[0] for r in improved]}"
    )
    assert not regressed, (
        f"tracks regressed > 1%: {[(r[0], f'{r[3]:.2f}%') for r in regressed]}\n"
        f"{debug_msg}"
    )

    # All 6 training tracks must still complete the sim (no crashes
    # introduced by the residual).
    for name in _TRAINING_TRACKS:
        assert on[name]["sim_passed"], (
            f"{name}: residual broke sim_passed\n{debug_msg}"
        )
