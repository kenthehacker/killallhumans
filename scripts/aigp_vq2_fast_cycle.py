"""Minimal noninteractive front door for one bounded VQ2 powered cycle.

The legacy Package 2 wrapper remains available for historical evidence, but it
is deliberately not on this path.  A fast cycle writes one compact manifest,
takes a nonblocking host lock, and delegates flight to ``aigp_vq2_run``.  The
runner still owns reset/GO proof, fresh-stream checks, command pacing and
bounds, watchdogs, disarm/reset, and cleanup confirmation.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from scripts.aigp_live_lease import (
    LIVE_LEASE_MUTEX_NAME,
    LiveLeaseError,
    live_simulator_lease,
)
from scripts.aigp_vq2_controller_config import (
    ControllerConfigError,
    VQ2ControllerConfig,
    default_controller_config,
    validate_controller_config,
)
from scripts.aigp_vq2_visual_config import (
    VisualConfigError,
    VisualNavigationConfig,
    default_visual_config,
    validate_visual_config,
)


MANIFEST_SCHEMA = "aigp-vq2-fast-flight-cycle-manifest/2"
RESULT_SCHEMA = "aigp-vq2-fast-flight-cycle-result/2"
SIMULATOR_BUILD = 3385
SIMULATOR_MODE = "Training"
DEFAULT_ADDRESS = "udpin:127.0.0.1:14550"
ISOLATION_FLAGS = ("-E", "-s", "-B")
VISUAL_POWERED_STAGES = ("visual-shadow", "visual-align")
FAST_POWERED_STAGES = (
    "sign-id",
    "hover",
    "gate0",
    "gate0-observe",
    "gate1-recenter",
    *VISUAL_POWERED_STAGES,
    "calibration-excite",
)
_HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")

_RUNTIME_SOURCE_PATHS = (
    "scripts/aigp_live_lease.py",
    "scripts/aigp_vq2_controller_config.py",
    "scripts/aigp_vq2_visual_config.py",
    "scripts/aigp_vq2_visual_alignment_stage.py",
    "scripts/aigp_vq2_fast_cycle.py",
    "scripts/aigp_vq2_run.py",
    "scripts/aigp_vq2_powered_attempt.py",
    "competition/adapter.py",
    "competition/aigp_mavlink.py",
    "competition/aigp_messages.py",
    "competition/vq2_contracts.py",
    "competition/vq2_capture.py",
    "competition/vq2_visual_tracker.py",
    "competition/vq2_vision.py",
    "competition/vision_udp.py",
    "estimation/imu_attitude.py",
    "gate_detection/src/gate_detector.py",
    "gate_detection/src/vq2_detector.py",
    "planning/vq2_gate_graph.py",
    "planning/vq2_visual_approach.py",
    "planning/vq2_visual_alignment.py",
    "planning/vq2_visual_recovery.py",
    "planning/vq2_visual_servo.py",
)

ControllerConfigValue = VQ2ControllerConfig | VisualNavigationConfig


class FastCycleError(RuntimeError):
    """The compact admission or execution boundary failed closed."""


def _visual_replay_filename(stage: str) -> str | None:
    if stage == "visual-shadow":
        return "shadow.vq2replay"
    if stage == "visual-align":
        return "alignment.vq2replay"
    return None


class _CaptureLeaseRelease:
    """Preserve a terminal result if lease cleanup fails after a completed body."""

    def __init__(self, manager: Any, errors: list[BaseException]) -> None:
        self._manager = manager
        self._errors = errors

    def __enter__(self) -> Any:
        return self._manager.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is not None:
            return bool(self._manager.__exit__(exc_type, exc, traceback))
        try:
            return bool(self._manager.__exit__(None, None, None))
        except LiveLeaseError as cleanup_error:
            self._errors.append(cleanup_error)
            return False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("UTC timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _publish_create_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FastCycleError(f"could not create {path.name}") from exc


def _file_identity(path: Path) -> dict[str, Any]:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except OSError as exc:
        raise FastCycleError(f"could not identify required file {path}") from exc
    before_key = (before.st_size, before.st_mtime_ns, before.st_ino)
    after_key = (after.st_size, after.st_mtime_ns, after.st_ino)
    if before_key != after_key:
        raise FastCycleError(f"required file changed while hashing: {path}")
    return {
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def _optional_file_identity(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    return {"path": str(path), **_file_identity(path)}


def _runtime_source_identities(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in _RUNTIME_SOURCE_PATHS:
        identity = _file_identity(repo_root / Path(relative))
        rows.append({"path": relative, **identity})
    return rows


def _git_bytes(repo_root: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FastCycleError(f"git {' '.join(arguments)} failed") from exc
    return completed.stdout


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    commit = _git_bytes(repo_root, ("rev-parse", "HEAD")).decode("ascii").strip()
    tree = _git_bytes(repo_root, ("rev-parse", "HEAD^{tree}")).decode(
        "ascii"
    ).strip()
    if _HEX40_RE.fullmatch(commit) is None or _HEX40_RE.fullmatch(tree) is None:
        raise FastCycleError("Git returned an invalid commit or tree identity")
    status = _git_bytes(
        repo_root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    )
    tracked_diff = _git_bytes(repo_root, ("diff", "--binary", "HEAD", "--"))
    return {
        "head_commit": commit,
        "head_tree": tree,
        "worktree_state": "dirty" if status else "clean",
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _excitation_plan_identity(stage: str) -> Mapping[str, Any] | None:
    if stage != "calibration-excite":
        return None
    from scripts import aigp_vq2_powered_attempt as contract

    plan = contract.fast_excitation_plan()
    return {
        "plan_id": plan["plan_id"],
        "sha256": contract.canonical_object_sha256(plan),
        "tick_count": plan["tick_count"],
        "control_period_ns": plan["control_period_ns"],
    }


def _controller_evidence(
    config: ControllerConfigValue,
    *,
    candidate_commit: str,
) -> dict[str, Any]:
    if _HEX40_RE.fullmatch(candidate_commit) is None:
        raise FastCycleError("controller binding requires an exact Git commit")
    effective = config.to_effective_mapping()
    return {
        "git_commit": candidate_commit,
        "config_schema": config.schema,
        "controller_family": config.controller_family,
        "config_sha256": config.effective_config_sha256,
        "effective_parameters": {
            key: value
            for key, value in effective.items()
            if key not in {"schema", "controller_family"}
        },
    }


def _default_controller_config_for_stage(stage: str) -> ControllerConfigValue:
    if stage in VISUAL_POWERED_STAGES:
        return default_visual_config()
    return default_controller_config()


def _validate_controller_config_for_stage(
    stage: str,
    document: object,
) -> ControllerConfigValue:
    if stage in VISUAL_POWERED_STAGES:
        return validate_visual_config(document)
    return validate_controller_config(document)


def _load_controller_config(
    path: Path | None,
    *,
    stage: str = "gate1-recenter",
) -> ControllerConfigValue:
    if path is None:
        return _default_controller_config_for_stage(stage)
    resolved = path.expanduser().resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise FastCycleError(
            f"could not read controller configuration {resolved}"
        ) from exc
    if not payload or len(payload) > 64 * 1024:
        raise FastCycleError(
            "controller configuration must contain 1-65536 bytes"
        )

    def exact_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise FastCycleError(
                    f"controller configuration repeats field {key!r}"
                )
            value[key] = item
        return value

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=exact_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FastCycleError(
                    f"controller configuration contains invalid {value}"
                )
            ),
        )
        return _validate_controller_config_for_stage(stage, document)
    except UnicodeDecodeError as exc:
        raise FastCycleError(
            "controller configuration must be UTF-8 JSON"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FastCycleError(
            f"controller configuration is malformed JSON: {exc.msg}"
        ) from exc
    except (ControllerConfigError, VisualConfigError) as exc:
        raise FastCycleError(f"controller configuration refused: {exc}") from exc


def build_manifest(
    *,
    stage: str,
    run_id: str,
    created_at: datetime,
    repo_root: Path,
    run_directory: Path,
    git_snapshot: Mapping[str, Any],
    runtime_sources: Sequence[Mapping[str, Any]],
    target_config: Mapping[str, Any],
    development_lock: Mapping[str, Any],
    excitation_plan: Mapping[str, Any] | None,
    controller_config: ControllerConfigValue | None = None,
) -> dict[str, Any]:
    """Build the exact small manifest with one bounded controller identity."""

    if stage not in FAST_POWERED_STAGES:
        raise FastCycleError(f"unsupported fast powered stage: {stage}")
    effective_controller = (
        controller_config or _default_controller_config_for_stage(stage)
    )
    controller = _controller_evidence(
        effective_controller,
        candidate_commit=str(git_snapshot["head_commit"]),
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "created_at_utc": _format_utc(created_at),
        "authorization": {
            "basis": "caller_asserted_existing_scoped_user_authorization",
            "scope": {"stage": stage, "attempt_limit": 1},
            "interactive_confirmation": False,
            "expires": "process_exit",
        },
        "simulator": {
            "build": SIMULATOR_BUILD,
            "mode": SIMULATOR_MODE,
            "mode_basis": "configured_session_not_machine_readable",
        },
        "candidate": {
            "worktree": str(repo_root),
            **dict(git_snapshot),
            "runtime_sources": [dict(row) for row in runtime_sources],
        },
        "controller": controller,
        "runtime": {
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "isolation_flags": list(ISOLATION_FLAGS),
            "development_lock": dict(development_lock),
        },
        "inputs": {
            "target_config": dict(target_config),
            "excitation_plan": (
                None if excitation_plan is None else dict(excitation_plan)
            ),
        },
        "execution": {
            "stage": stage,
            "address": DEFAULT_ADDRESS,
            "separate_passive_preflight": False,
            "preflight_healthy_dwell_s": 0.0,
            "manual_training_challenge": False,
            "screenshot_capture": False,
            "full_environment_or_import_inventory": False,
            "live_lease_mutex": LIVE_LEASE_MUTEX_NAME,
        },
        "evidence": {
            "directory": str(run_directory),
            "manifest": str(run_directory / "run-manifest.json"),
            "trace": str(run_directory / "session.jsonl.gz"),
            "replay_bundle": (
                str(run_directory / replay_name)
                if (replay_name := _visual_replay_filename(stage))
                else None
            ),
            "result": str(run_directory / "result.json"),
            "live_lease": str(run_directory / "live-lease.json"),
        },
    }


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_external_path(path: Path, repo_root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == repo_root or _is_within(resolved, repo_root):
        raise FastCycleError(f"{label} must be outside the Git worktree")
    return resolved


def _default_evidence_root() -> Path:
    configured = os.environ.get("AIGP_EVIDENCE_ROOT")
    base = Path(configured) if configured else Path.home() / "aigp-evidence"
    return base / "fast-flight-cycles"


def _require_isolated_runtime() -> None:
    if (
        sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
        or sys.flags.dont_write_bytecode != 1
    ):
        raise FastCycleError("fast cycles require Python -E -s -B isolation")


def _new_run_id(stage: str, created_at: datetime) -> str:
    timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{stage}-{secrets.token_hex(4)}"


def _execute_fast_cycle(
    stage: str,
    *,
    evidence_root: Path | None = None,
    address: str = DEFAULT_ADDRESS,
    now: Callable[[], datetime] = _utc_now,
    load_runner: Callable[[], Any] | None = None,
    lease_factory: Callable[..., Any] = live_simulator_lease,
    controller_config: Mapping[str, Any] | ControllerConfigValue | None = None,
) -> tuple[int, dict[str, Any]]:
    """Injected implementation used by the production boundary and unit tests."""

    if stage not in FAST_POWERED_STAGES:
        raise FastCycleError(f"unsupported fast powered stage: {stage}")
    if address != DEFAULT_ADDRESS:
        raise FastCycleError("fast cycles use the fixed verified MAVLink address")
    try:
        effective_controller = (
            _default_controller_config_for_stage(stage)
            if controller_config is None
            else _validate_controller_config_for_stage(
                stage,
                (
                    controller_config.to_effective_mapping()
                    if isinstance(
                        controller_config,
                        (VQ2ControllerConfig, VisualNavigationConfig),
                    )
                    else controller_config
                ),
            )
        )
    except (ControllerConfigError, VisualConfigError) as exc:
        raise FastCycleError(f"controller configuration refused: {exc}") from exc
    if (
        stage not in VISUAL_POWERED_STAGES
        and stage != "gate1-recenter"
        and effective_controller.effective_config_sha256
        != default_controller_config().effective_config_sha256
    ):
        raise FastCycleError(
            "custom controller configurations are admitted only for "
            "gate1-recenter"
        )
    repo_root = Path(__file__).resolve().parents[1]
    root = _require_external_path(
        _default_evidence_root() if evidence_root is None else evidence_root,
        repo_root,
        label="evidence root",
    )
    root.mkdir(parents=True, exist_ok=True)
    created_at = now()
    run_id = _new_run_id(stage, created_at)
    run_directory = root / run_id
    try:
        run_directory.mkdir()
    except OSError as exc:
        raise FastCycleError("could not create the unique evidence directory") from exc

    lease_release_errors: list[BaseException] = []
    lease_manager = lease_factory(
        run_directory / "live-lease.json",
        initial_phase="fast_cycle",
    )
    with _CaptureLeaseRelease(lease_manager, lease_release_errors):
        runtime_before = _runtime_source_identities(repo_root)
        target_path = repo_root / "config/aigp_vq2_calibration_target_build3385.json"
        lockfile_path = repo_root / "requirements/development-test.lock.txt"
        target_config = {
            "path": str(target_path),
            **_file_identity(target_path),
        }
        development_lock = {
            "path": str(lockfile_path),
            **_file_identity(lockfile_path),
        }
        git_snapshot = _git_snapshot(repo_root)
        if (
            stage in VISUAL_POWERED_STAGES
            and git_snapshot["worktree_state"] != "clean"
        ):
            raise FastCycleError(
                "visual-navigation powered stages require a clean exact commit"
            )
        manifest = build_manifest(
            stage=stage,
            run_id=run_id,
            created_at=created_at,
            repo_root=repo_root,
            run_directory=run_directory,
            git_snapshot=git_snapshot,
            runtime_sources=runtime_before,
            target_config=target_config,
            development_lock=development_lock,
            excitation_plan=_excitation_plan_identity(stage),
            controller_config=effective_controller,
        )
        manifest_path = run_directory / "run-manifest.json"
        manifest_payload = _canonical_json_bytes(manifest)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        _publish_create_new(manifest_path, manifest)

        started_at = now()
        result_value: dict[str, Any]
        exit_code = 2
        try:
            if load_runner is None:
                from scripts import aigp_vq2_run as runner_module
            else:
                runner_module = load_runner()
            if _runtime_source_identities(repo_root) != runtime_before:
                raise FastCycleError("runtime sources changed before live execution")
            if _file_identity(target_path) != {
                key: target_config[key] for key in ("size_bytes", "sha256")
            }:
                raise FastCycleError("target configuration changed before live execution")
            if _file_identity(lockfile_path) != {
                key: development_lock[key] for key in ("size_bytes", "sha256")
            }:
                raise FastCycleError("development lock changed before live execution")
            observed_git = _git_snapshot(repo_root)
            expected_git = {
                key: manifest["candidate"][key]
                for key in (
                    "head_commit",
                    "head_tree",
                    "worktree_state",
                    "status_sha256",
                    "tracked_diff_sha256",
                )
            }
            if observed_git != expected_git:
                raise FastCycleError("candidate changed before live execution")
            result = asyncio.run(
                runner_module.run_live(
                    stage,
                    address,
                    str(run_directory / "session.jsonl.gz"),
                    preflight_before_powered_stage=False,
                    write_diagnostic_pngs=False,
                    run_manifest_sha256=manifest_sha256,
                    controller_config=effective_controller.to_effective_mapping(),
                    candidate_commit=manifest["candidate"]["head_commit"],
                    expected_controller_config_sha256=(
                        manifest["controller"]["config_sha256"]
                    ),
                    replay_bundle=(
                        str(run_directory / replay_name)
                        if (
                            replay_name
                            := _visual_replay_filename(stage)
                        )
                        else None
                    ),
                    recording_approved=(stage in VISUAL_POWERED_STAGES),
                )
            )
            result_value = {
                "schema": RESULT_SCHEMA,
                "run_id": run_id,
                "stage": stage,
                "started_at_utc": _format_utc(started_at),
                "finished_at_utc": _format_utc(now()),
                "success": bool(result.success),
                "reason": str(result.reason),
                "run_manifest_sha256": manifest_sha256,
                "controller": dict(manifest["controller"]),
                "trace": _optional_file_identity(
                    run_directory / "session.jsonl.gz"
                ),
                "runner_result": asdict(result),
            }
            exit_code = 0 if result.success else 2
        except Exception as exc:
            result_value = {
                "schema": RESULT_SCHEMA,
                "run_id": run_id,
                "stage": stage,
                "started_at_utc": _format_utc(started_at),
                "finished_at_utc": _format_utc(now()),
                "success": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "run_manifest_sha256": manifest_sha256,
                "controller": dict(manifest["controller"]),
                "trace": _optional_file_identity(
                    run_directory / "session.jsonl.gz"
                ),
                "runner_result": None,
            }
    if lease_release_errors:
        cleanup_error = lease_release_errors[0]
        result_value["success"] = False
        result_value["reason"] = (
            f"{result_value['reason']}; live lease cleanup failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
        result_value["finished_at_utc"] = _format_utc(now())
        exit_code = 2
    result_value["live_lease"] = _optional_file_identity(
        run_directory / "live-lease.json"
    )
    _publish_create_new(run_directory / "result.json", result_value)
    return exit_code, result_value


def execute_fast_cycle(
    stage: str,
    *,
    evidence_root: Path | None = None,
    controller_config_path: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Enforce production isolation/lease boundaries and run one attempt."""

    _require_isolated_runtime()
    return _execute_fast_cycle(
        stage,
        evidence_root=evidence_root,
        address=DEFAULT_ADDRESS,
        now=_utc_now,
        load_runner=None,
        lease_factory=live_simulator_lease,
        controller_config=_load_controller_config(
            controller_config_path,
            stage=stage,
        ),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one bounded build-3385 Training flight with a compact manifest "
            "and no interactive ceremony"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("stage", choices=FAST_POWERED_STAGES)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument(
        "--controller-config",
        type=Path,
        help=(
            "complete strict-schema controller JSON for the selected stage; "
            "omitted uses that stage family's default"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        exit_code, result = execute_fast_cycle(
            parsed.stage,
            evidence_root=parsed.evidence_root,
            controller_config_path=parsed.controller_config,
        )
    except (FastCycleError, LiveLeaseError) as exc:
        print(f"fast flight cycle refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
