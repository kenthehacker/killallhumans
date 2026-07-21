"""Offline-only L0 import-inventory audit entry point.

This module has no live or publication boundary.  Its sole invocation is an
exact ``-E -s -B -m`` process with no arguments and stdout captured by the L0
reviewer.  The six frozen seeds are first imported in ordinal array order;
the powered probe then imports its code-owned eager list and derives the
complete, handle-verified graph.
"""

from __future__ import annotations

import importlib
import sys


AUDIT_MODULE = "scripts.aigp_vq2_powered_import_audit"
IMPORT_SEEDS = (
    "scripts.aigp_vq2_powered_attempt",
    "scripts.aigp_vq2_powered_calibration_analysis",
    "scripts.aigp_vq2_powered_calibration_probe",
    "scripts.aigp_vq2_powered_cleanup",
    "scripts.aigp_vq2_powered_runtime",
    "scripts.aigp_vq2_run",
)


def _exact_invocation() -> bool:
    return (
        tuple(sys.version_info[:3]) == (3, 12, 2)
        and sys.implementation.name == "cpython"
        and (
            sys.flags.ignore_environment,
            sys.flags.no_user_site,
            sys.flags.dont_write_bytecode,
        )
        == (1, 1, 1)
        and list(getattr(sys, "orig_argv", ()))[1:]
        == ["-E", "-s", "-B", "-m", AUDIT_MODULE]
        and len(sys.argv) == 1
        and getattr(sys.stdout, "buffer", None) is not None
        and not sys.stdout.isatty()
        and getattr(getattr(sys.modules.get("__main__"), "__spec__", None), "name", None)
        == AUDIT_MODULE
    )


def main() -> int:
    if not _exact_invocation():
        print("initial import inventory audit refused: invocation", file=sys.stderr)
        return 2
    try:
        imported = {}
        for name in IMPORT_SEEDS:
            imported[name] = importlib.import_module(name)
        contract = imported["scripts.aigp_vq2_powered_attempt"]
        probe = imported["scripts.aigp_vq2_powered_calibration_probe"]
        if (
            tuple(contract.IMPORT_INVENTORY_SEEDS) != IMPORT_SEEDS
            or tuple(probe.POWERED_IMPORT_SEED_MODULES) != IMPORT_SEEDS
            or probe.IMPORT_AUDIT_MODULE != AUDIT_MODULE
        ):
            raise RuntimeError("code-owned import audit constants drifted")
        return probe.run_initial_import_inventory_audit(
            audit_module=AUDIT_MODULE,
            seed_modules=IMPORT_SEEDS,
        )
    except Exception as exc:
        print(
            f"initial import inventory audit refused: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover - isolated production audit
    raise SystemExit(main())
