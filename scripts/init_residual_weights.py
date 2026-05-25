"""
Iter-022: produce a zero-init `TrackerResidualMLP` .npz file.

Closes the "connect-the-dots" gap on the ML pipeline scaffolding:
  - iter-001 A15  : TrackerResidualMLP shipped (zero-init constructor exists)
  - iter-014      : feature-trace hook on GeometricTracker
  - iter-015      : save/load utility for feature traces; PipelineConfig wiring
  - iter-022 (this): a runnable script that materialises a weights .npz
                    file callers can point `TrackerConfig.residual_weights_path`
                    at to test the load-and-use path end to end.

The produced weights are ZERO-INIT — calling forward() with them always
returns (0, 0, 0). Tracker behaviour with `use_residual=True +
residual_weights_path=<this file>` is therefore byte-identical to
baseline (modulo the extra branch's float-epsilon noise; see
`tests/test_tracker_residual.py::test_residual_off_is_baseline`).

This is intentional: the script is a pipeline-correctness smoke
artifact, NOT a trained model. Iter-023+ will replace the contents
with FEL or least-squares-fit weights once dataset collection works.

Usage:
    python scripts/init_residual_weights.py [--out PATH]

Default PATH: control/residual_weights.npz (gitignored, regenerated
on demand).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without `python -m`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from control.learned_residual import TrackerResidualMLP  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Materialise a zero-init residual-weights .npz file",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "control" / "residual_weights.npz"),
        help="Output path for the .npz file",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlp = TrackerResidualMLP.zero_init()
    mlp.to_npz(out_path)
    print(f"wrote zero-init residual weights to {out_path}")
    print("  W1.shape:", mlp.W1.shape)
    print("  b1.shape:", mlp.b1.shape)
    print("  W2.shape:", mlp.W2.shape)
    print("  b2.shape:", mlp.b2.shape)
    print()
    print("To use:")
    print("  from control.mpc_tracker import GeometricTracker, TrackerConfig")
    print(
        f"  tracker = GeometricTracker(TrackerConfig(use_residual=True, "
        f"residual_weights_path={str(out_path)!r}))"
    )
    print("Behavior will be identical to baseline (weights are zero).")


if __name__ == "__main__":
    main()
