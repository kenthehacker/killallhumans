"""Single-scheduler orchestration in isolated Git worktrees.

The ordinary scheduler is intentionally non-live and refuses T5.  Official
simulator campaigns use :mod:`aigp_loop.campaign`, whose authorization is tied
to one immutable campaign plan.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time
import uuid
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from ._util import (
    bounded_tail,
    canonical_json,
    environment_fingerprint,
    git_provenance,
    json_hash,
    run_checked,
    read_secure_regular_file,
    secure_directory,
    secure_relative_regular_file,
    secure_regular_file,
    sha256_bytes,
    sha256_file,
    strict_json_load,
)
from ._util import strict_json_loads
from .ledger import TrialLedger
from .promotion import (
    CandidateEvaluation,
    HardGates,
    PromotionLadder,
    QualityVector,
    Tier,
    TierEligibility,
    replay_promotion_policy_failures,
    validate_promotion_chain,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_ORCHESTRATION_LEASE = "orchestration"
_TRUSTED_MANIFEST_PATH = "config/promotion_trusted_files.json"
_TRUSTED_REPLAY_HOST_FILES = frozenset(
    {
        "aigp_loop/__init__.py",
        "aigp_loop/_util.py",
        "aigp_loop/evidence.py",
        "aigp_loop/ledger.py",
        "aigp_loop/promotion.py",
        "aigp_loop/replay.py",
        "scripts/aigp_replay.py",
    }
)
_TRUSTED_T1_PYTEST_FILES = frozenset(
    {
        "conftest.py",
        "pyproject.toml",
        "config/t1_pytest.ini",
        "config/t1_pytest_policy.json",
        "scripts/aigp_pytest.py",
        "competition/tests/test_adapter.py",
        "competition/tests/test_aigp_indi_wiring.py",
        "competition/tests/test_aigp_mavlink.py",
        "competition/tests/test_aigp_messages.py",
        "competition/tests/test_aigp_recorder.py",
        "competition/tests/test_gate_map_integrity.py",
        "competition/tests/test_sim_health.py",
        "competition/tests/test_track_data.py",
        "competition/tests/test_vq2_contracts.py",
        "competition/tests/test_vq2_vision.py",
        "estimation/tests/test_ekf.py",
        "estimation/tests/test_gate_pnp.py",
        "estimation/tests/test_imu_attitude.py",
        "gate_detection/tests/test_detection.py",
        "gate_detection/tests/test_vq2_detector.py",
        "tests/test_aigp_vq2_runner.py",
        "tests/test_vision_udp.py",
        "tests/test_vision_udp_listener.py",
    }
)
_TRUSTED_NONLIVE_FILES = frozenset(
    {
        "aigp_loop/__init__.py",
        "aigp_loop/_util.py",
        "aigp_loop/evidence.py",
        "aigp_loop/ledger.py",
        "aigp_loop/nonlive.py",
        "aigp_loop/promotion.py",
        "aigp_loop/scheduler.py",
        "planning/__init__.py",
        "planning/artifact_cache.py",
        "scripts/aigp_nonlive.py",
        "scripts/benchmark.py",
        "scripts/benchmark_matrix.py",
        "sim_pybullet/configs/aigp_default.json",
        "sim_pybullet/configs/figure8.json",
        "sim_pybullet/configs/grand_tour.json",
        "sim_pybullet/configs/race_01.json",
        "sim_pybullet/configs/slalom.json",
        "sim_pybullet/configs/straight_hairpin.json",
        "sim_pybullet/configs/vertical_cliff.json",
    }
)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class CommandStep:
    argv: tuple[str, ...]
    timeout_s: float
    metrics_from_stdout: bool = False
    require_hard_gates: Optional[bool] = None
    trusted_files_sha256: tuple[tuple[str, str], ...] = ()
    isolation_wrapper: Optional[str] = None
    isolation_wrapper_sha256: Optional[str] = None
    trusted_host: bool = False

    def __post_init__(self) -> None:
        if not self.argv or any(not isinstance(part, str) or not part for part in self.argv):
            raise ValueError("tier commands must be non-empty argv arrays")
        if type(self.timeout_s) not in {int, float}:
            raise TypeError("tier command timeout must be numeric and not bool")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("tier command timeout must be finite and positive")
        for name in ("metrics_from_stdout", "require_hard_gates"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be an exact bool or null")
        if type(self.trusted_host) is not bool:
            raise TypeError("trusted_host must be an exact bool")
        if "{python}" in self.argv[1:]:
            raise ValueError("{python} is allowed only as argv[0]")
        if self.argv[0] == "{config}":
            raise ValueError("{config} cannot be an executable")
        raw_trusted = self.trusted_files_sha256
        if isinstance(raw_trusted, Mapping):
            raw_trusted = tuple(raw_trusted.items())
        if type(raw_trusted) is not tuple:
            raise TypeError("trusted_files_sha256 must be an exact object")
        normalized: list[tuple[str, str]] = []
        for item in raw_trusted:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("trusted file entries must be path/hash pairs")
            name, digest = item
            path = Path(name) if type(name) is str else None
            if (
                path is None
                or not name
                or path.is_absolute()
                or ".." in path.parts
                or type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    "trusted files require safe relative paths and SHA-256"
                )
            normalized.append((path.as_posix(), digest))
        if len({name for name, _digest in normalized}) != len(normalized):
            raise ValueError("duplicate trusted file path")
        object.__setattr__(
            self, "trusted_files_sha256", tuple(sorted(normalized))
        )
        if (self.isolation_wrapper is None) != (
            self.isolation_wrapper_sha256 is None
        ):
            raise ValueError("isolation wrapper path and SHA-256 must be paired")
        if self.isolation_wrapper is not None:
            if (
                type(self.isolation_wrapper) is not str
                or not self.isolation_wrapper.strip()
                or not _is_sha256(self.isolation_wrapper_sha256)
            ):
                raise ValueError(
                    "isolation wrapper requires a non-empty path and exact SHA-256"
                )

    def resolved_argv(
        self,
        config_path: Path,
        *,
        trusted_repository: Optional[Path] = None,
    ) -> tuple[str, ...]:
        argv = tuple(
            (
                str(config_path)
                if part == "{config}"
                else str(trusted_repository / "scripts" / "aigp_replay.py")
                if part == "{trusted_replay}" and trusted_repository is not None
                else part
            )
            for part in self.argv
        )
        if "{trusted_replay}" in argv:
            raise ValueError("trusted replay path requires a trusted repository")
        if argv[0] == "{python}":
            argv = (sys.executable, *argv[1:])
        if self.isolation_wrapper is not None:
            return (self.isolation_wrapper, "--", *argv)
        return argv


@dataclass(frozen=True)
class TierCommand:
    tier: Tier
    argv: tuple[str, ...] = ()
    timeout_s: float = 1.0
    metrics_from_stdout: bool = False
    require_hard_gates: Optional[bool] = None
    trusted_files_sha256: tuple[tuple[str, str], ...] = ()
    steps: tuple[CommandStep, ...] = ()

    def __post_init__(self) -> None:
        if self.tier is Tier.T5_AUTHORIZED_LIVE:
            raise ValueError("the non-live scheduler cannot contain a T5 command")
        if self.steps and self.argv:
            raise ValueError("use either a legacy argv or ordered steps, not both")
        if not self.steps:
            step = CommandStep(
                self.argv,
                self.timeout_s,
                self.metrics_from_stdout,
                self.require_hard_gates,
                self.trusted_files_sha256,
            )
            if step.require_hard_gates is None:
                step = dataclasses.replace(
                    step,
                    require_hard_gates=bool(
                        step.metrics_from_stdout and self.tier >= Tier.T1_VQ2_REPLAY
                    ),
                )
            object.__setattr__(self, "steps", (step,))
        else:
            normalized = []
            for step in self.steps:
                if step.require_hard_gates is None:
                    step = dataclasses.replace(
                        step,
                        require_hard_gates=bool(
                            step.metrics_from_stdout and self.tier >= Tier.T1_VQ2_REPLAY
                        ),
                    )
                normalized.append(step)
            object.__setattr__(self, "steps", tuple(normalized))


def _trusted_manifest_files(
    command_document: Path, raw_path: Any
) -> Mapping[str, str]:
    if raw_path != _TRUSTED_MANIFEST_PATH:
        raise ValueError(
            "trusted_manifest must name the canonical repository trust manifest"
        )
    repository = Path(
        run_checked(
            ["git", "rev-parse", "--show-toplevel"], cwd=command_document.parent
        )
    ).resolve()
    try:
        manifest_path = secure_relative_regular_file(repository, raw_path)
        manifest_payload = read_secure_regular_file(manifest_path)
        manifest = strict_json_loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "trusted manifest is missing, unsafe, or outside the repository"
        ) from exc
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "files"}
        or manifest.get("schema") != "aigp-trusted-evaluator-files/1"
        or type(manifest.get("files")) is not dict
        or not manifest["files"]
    ):
        raise ValueError("trusted evaluator manifest has an invalid exact schema")
    result = dict(manifest["files"])
    result[Path(raw_path).as_posix()] = sha256_bytes(manifest_payload)
    return result


def _argv_option_equals(
    argv: Sequence[str], option: str, expected: str
) -> bool:
    """Return true only for one exact two-token option/value pair."""

    if argv.count(option) != 1:
        return False
    index = argv.index(option)
    return index + 1 < len(argv) and argv[index + 1] == expected


def _t1_step_is_trusted_replay_host(step: CommandStep) -> bool:
    """Recognize the sole metrics-bearing T1 evaluator shape."""

    argv = step.argv
    return bool(
        step.trusted_host is True
        and len(argv) >= 6
        and argv[:4] == ("{python}", "-I", "{trusted_replay}", "corpus")
        and "--processor" in argv
        and "--isolation-wrapper" in argv
        and "--isolation-wrapper-sha256" in argv
        and _argv_option_equals(
            argv, "--trusted-manifest", _TRUSTED_MANIFEST_PATH
        )
        and _TRUSTED_MANIFEST_PATH
        in {name for name, _digest in step.trusted_files_sha256}
        and _TRUSTED_REPLAY_HOST_FILES
        <= {name for name, _digest in step.trusted_files_sha256}
    )


def _t1_step_has_candidate_isolation(step: CommandStep) -> bool:
    """Recognize the two reviewed T1 candidate-execution boundaries."""

    return bool(
        step.isolation_wrapper is not None
        or _t1_step_is_trusted_replay_host(step)
    )


def _t1_step_is_trusted_pytest(step: CommandStep) -> bool:
    return bool(
        step.trusted_host is False
        and step.metrics_from_stdout is False
        and step.argv == ("{python}", "-I", "scripts/aigp_pytest.py", "vq2")
        and step.isolation_wrapper is not None
        and _TRUSTED_T1_PYTEST_FILES
        <= {name for name, _digest in step.trusted_files_sha256}
    )


def _validate_t1_command(command: TierCommand) -> None:
    steps = command.steps
    if any(not step.trusted_files_sha256 for step in steps):
        raise ValueError("every T1 step requires trusted evaluator file binding")
    metrics_steps = [step for step in steps if step.metrics_from_stdout]
    if len(metrics_steps) != 1 or not _t1_step_is_trusted_replay_host(
        metrics_steps[0]
    ):
        raise ValueError(
            "T1 requires exactly one trusted-host replay metrics step"
        )
    pytest_steps = [step for step in steps if _t1_step_is_trusted_pytest(step)]
    if len(steps) != 2 or len(pytest_steps) != 1:
        raise ValueError(
            "T1 requires exactly one hash-pinned isolated pytest bootstrap"
        )
    for step in steps:
        if not _t1_step_has_candidate_isolation(step):
            raise ValueError(
                "every T1 candidate-executing step requires pinned OS isolation"
            )
        if step is not metrics_steps[0] and (
            step.metrics_from_stdout
            or step.trusted_host
            or step.isolation_wrapper is None
        ):
            raise ValueError(
                "non-replay T1 steps must be non-metrics candidate-wrapped commands"
            )


def _validate_t0_command(command: TierCommand) -> None:
    if len(command.steps) != 1:
        raise ValueError("T0 requires one trusted affected-test step")
    step = command.steps[0]
    if (
        step.argv[:4]
        != ("{python}", "-I", "scripts/aigp_pytest.py", "affected")
        or len(step.argv) < 5
        or step.metrics_from_stdout is not False
        or step.require_hard_gates is not False
        or step.trusted_host is not False
        or step.isolation_wrapper is not None
        or not {
            "scripts/aigp_pytest.py",
            "config/t1_pytest.ini",
            "config/t1_pytest_policy.json",
        }
        <= {name for name, _digest in step.trusted_files_sha256}
    ):
        raise ValueError(
            "T0 requires the hash-pinned isolated affected-test bootstrap"
        )


def _validate_nonlive_command(command: TierCommand) -> None:
    if len(command.steps) != 1:
        raise ValueError("T2-T4 require exactly one trusted non-live step")
    step = command.steps[0]
    argv = step.argv
    tier_text = str(int(command.tier))
    if (
        step.trusted_host is not False
        or step.metrics_from_stdout is not True
        or step.require_hard_gates is not True
        or step.isolation_wrapper is not None
        or len(argv) < 6
        or argv[:3] != ("{python}", "-I", "scripts/aigp_nonlive.py")
        or not _argv_option_equals(argv, "--tier", tier_text)
        or not _argv_option_equals(
            argv, "--trusted-manifest", _TRUSTED_MANIFEST_PATH
        )
        or _TRUSTED_MANIFEST_PATH
        not in {name for name, _digest in step.trusted_files_sha256}
        or _TRUSTED_NONLIVE_FILES
        > {name for name, _digest in step.trusted_files_sha256}
    ):
        raise ValueError(
            "T2-T4 require the hash-pinned isolated non-live script bootstrap"
        )


def load_tier_commands(path: Path | str) -> Dict[Tier, TierCommand]:
    command_document = secure_regular_file(path)
    command_payload = read_secure_regular_file(command_document)
    try:
        payload = strict_json_loads(command_payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("promotion command document must be UTF-8") from exc
    if type(payload) is not dict or set(payload) != {"schema", "tiers"}:
        raise ValueError("promotion command document has missing/unknown keys")
    if payload.get("schema") != "aigp-promotion-commands/1":
        raise ValueError("unsupported promotion command schema")
    if type(payload["tiers"]) is not list:
        raise TypeError("promotion tiers must be a list")
    result: Dict[Tier, TierCommand] = {}
    for raw in payload.get("tiers", []):
        if type(raw) is not dict:
            raise TypeError("tier entry must be an object")
        if type(raw.get("tier")) is not int:
            raise TypeError("tier must be an exact integer")
        tier = Tier(raw["tier"])
        if "steps" in raw:
            if set(raw) != {"tier", "steps"}:
                raise ValueError("step tier entry has missing/unknown keys")
            if "argv" in raw:
                raise ValueError("tier cannot contain both argv and steps")
            if type(raw["steps"]) is not list or not raw["steps"]:
                raise ValueError("steps must be a non-empty list")
            steps = []
            for step in raw["steps"]:
                allowed = {
                    "argv",
                    "timeout_s",
                    "metrics_from_stdout",
                    "require_hard_gates",
                    "trusted_files_sha256",
                    "trusted_manifest",
                    "isolation_wrapper",
                    "isolation_wrapper_sha256",
                    "trusted_host",
                }
                if type(step) is not dict or not {"argv", "timeout_s"} <= set(step) or set(step) - allowed:
                    raise ValueError("command step has missing/unknown keys")
                if type(step.get("argv")) is not list:
                    raise TypeError("commands must be exact argv arrays; shell strings and objects are forbidden")
                raw_trusted = step.get("trusted_files_sha256", {})
                if type(raw_trusted) is not dict:
                    raise TypeError("trusted_files_sha256 must be an exact object")
                trusted = dict(raw_trusted)
                if "trusted_manifest" in step:
                    for name, digest in _trusted_manifest_files(
                        command_document, step["trusted_manifest"]
                    ).items():
                        if name in trusted and trusted[name] != digest:
                            raise ValueError("conflicting trusted evaluator hashes")
                        trusted[name] = digest
                steps.append(
                    CommandStep(
                        argv=tuple(step["argv"]),
                        timeout_s=step["timeout_s"],
                        metrics_from_stdout=step.get("metrics_from_stdout", False),
                        require_hard_gates=step.get("require_hard_gates"),
                        trusted_files_sha256=trusted,
                        isolation_wrapper=step.get("isolation_wrapper"),
                        isolation_wrapper_sha256=step.get(
                            "isolation_wrapper_sha256"
                        ),
                        trusted_host=step.get("trusted_host", False),
                    )
                )
            command = TierCommand(tier=tier, steps=tuple(steps))
        else:
            allowed = {
                "tier",
                "argv",
                "timeout_s",
                "metrics_from_stdout",
                "require_hard_gates",
                "trusted_files_sha256",
                "trusted_manifest",
            }
            if not {"tier", "argv", "timeout_s"} <= set(raw) or set(raw) - allowed:
                raise ValueError("tier command has missing/unknown keys")
            if type(raw.get("argv")) is not list:
                raise TypeError("commands must be exact argv arrays; shell strings and objects are forbidden")
            raw_trusted = raw.get("trusted_files_sha256", {})
            if type(raw_trusted) is not dict:
                raise TypeError("trusted_files_sha256 must be an exact object")
            trusted = dict(raw_trusted)
            if "trusted_manifest" in raw:
                trusted.update(
                    _trusted_manifest_files(command_document, raw["trusted_manifest"])
                )
            command = TierCommand(
                tier=tier,
                argv=tuple(raw["argv"]),
                timeout_s=raw["timeout_s"],
                metrics_from_stdout=raw.get("metrics_from_stdout", False),
                require_hard_gates=raw.get("require_hard_gates"),
                trusted_files_sha256=trusted,
            )
        if command.tier in result:
            raise ValueError(f"duplicate tier command: {command.tier.name}")
        if command.tier is Tier.T0_AFFECTED:
            _validate_t0_command(command)
        elif command.tier is Tier.T1_VQ2_REPLAY:
            _validate_t1_command(command)
        elif Tier.T2_WARM_SIM <= command.tier <= Tier.T4_FULL_NON_LIVE:
            _validate_nonlive_command(command)
        result[command.tier] = command
    return result


class GitWorktreePool:
    def __init__(self, repository: Path | str, root: Path | str) -> None:
        self.repository = secure_directory(repository)
        lexical_root = Path(root)
        if not lexical_root.is_absolute():
            lexical_root = Path.cwd() / lexical_root
        lexical_root = Path(os.path.abspath(lexical_root))
        lexical_root.mkdir(parents=True, exist_ok=True)
        self.root = secure_directory(lexical_root)
        top = Path(run_checked(["git", "rev-parse", "--show-toplevel"], cwd=self.repository)).resolve()
        if top != self.repository:
            raise ValueError("repository must be the Git top level")
        if self.root == self.repository or self.repository in self.root.parents:
            # Nested worktrees are possible but create indexing/context noise
            # and can be accidentally swept by broad repository tools.
            raise ValueError("isolated worktree root must be outside the repository")

    def path_for(self, trial_id: str) -> Path:
        if not _SAFE_ID.fullmatch(trial_id):
            raise ValueError("trial id contains unsafe path characters")
        # Preserve the lexical leaf until ``ensure`` can reject any existing
        # link/junction.  Resolving here would turn two trial IDs into aliases
        # of one shared checkout before the security check sees the indirection.
        path = Path(os.path.abspath(self.root / trial_id))
        if path.parent != self.root:
            raise ValueError("worktree path escaped pool root")
        return path

    def _assert_pristine_checkout(self, path: Path, commit_hash: str) -> None:
        """Reject every object not represented by the immutable Git tree.

        Git status/provenance intentionally ignores ignored files.  That makes
        it insufficient at a crash/resume boundary: a killed candidate could
        leave an ignored Python module or model behind for the next process.
        Compare the lexical checkout to the tree object itself, allowing only
        the linked-worktree ``.git`` indirection in addition to tracked files.
        """

        root = secure_directory(path)
        try:
            listing = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", "-z", commit_hash],
                cwd=str(self.repository),
                check=True,
                capture_output=True,
                timeout=30.0,
                shell=False,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("could not derive pristine checkout inventory") from exc
        expected_files = {".git"}
        expected_directories: set[str] = set()
        for raw_name in listing.split(b"\0"):
            if not raw_name:
                continue
            name = os.fsdecode(raw_name)
            relative = Path(name)
            if (
                relative.is_absolute()
                or relative.drive
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise RuntimeError("Git tree contains an unsafe checkout path")
            normalized = relative.as_posix()
            expected_files.add(normalized)
            parent = relative.parent
            while parent != Path("."):
                expected_directories.add(parent.as_posix())
                parent = parent.parent

        git_file = root / ".git"
        try:
            git_payload = read_secure_regular_file(git_file)
            git_text = git_payload.decode("utf-8").rstrip("\r\n")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("worktree .git indirection is missing or unsafe") from exc
        if (
            not git_text.startswith("gitdir: ")
            or "\n" in git_text
            or "\r" in git_text
            or not git_text[8:]
        ):
            raise RuntimeError("worktree .git indirection is malformed")
        administrative = Path(git_text[8:])
        if not administrative.is_absolute():
            administrative = root / administrative
        try:
            administrative = secure_directory(administrative)
            common_raw = run_checked(
                ["git", "rev-parse", "--git-common-dir"], cwd=self.repository
            )
            common = Path(common_raw)
            if not common.is_absolute():
                common = self.repository / common
            worktree_admin_root = secure_directory(common) / "worktrees"
            administrative.relative_to(worktree_admin_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise RuntimeError("worktree .git indirection escaped Git administration") from exc

        observed_files: set[str] = set()
        observed_directories: set[str] = set()
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        for directory, names, files in os.walk(root, topdown=True, followlinks=False):
            base = Path(directory)
            names.sort()
            files.sort()
            for name in names:
                target = base / name
                info = target.lstat()
                if stat.S_ISLNK(info.st_mode) or (
                    getattr(info, "st_file_attributes", 0) & reparse_flag
                ):
                    raise RuntimeError("worktree contains a symlink/reparse directory")
                observed_directories.add(target.relative_to(root).as_posix())
            for name in files:
                target = base / name
                try:
                    secure_regular_file(target)
                except ValueError as exc:
                    raise RuntimeError("worktree contains an unsafe file") from exc
                observed_files.add(target.relative_to(root).as_posix())
        if observed_files != expected_files or observed_directories != expected_directories:
            extras = sorted(
                (observed_files - expected_files)
                | (observed_directories - expected_directories)
            )
            missing = sorted(
                (expected_files - observed_files)
                | (expected_directories - observed_directories)
            )
            detail = []
            if extras:
                detail.append("extra=" + ",".join(extras[:8]))
            if missing:
                detail.append("missing=" + ",".join(missing[:8]))
            raise RuntimeError(
                "worktree is not an exact pristine checkout"
                + (": " + "; ".join(detail) if detail else "")
            )

    def ensure(self, trial_id: str, commit_hash: str) -> Path:
        path = self.path_for(trial_id)
        # ``Path.exists()`` follows links and therefore reports ``False`` for
        # a dangling symlink.  Treat every lexical directory entry as an
        # existing target so indirection is rejected before Git is allowed to
        # create or populate a worktree at that name.
        try:
            path.lstat()
        except FileNotFoundError:
            path_exists = False
        else:
            path_exists = True
        if path_exists:
            path = secure_directory(path)
            self._assert_pristine_checkout(path, commit_hash)
            actual = run_checked(["git", "rev-parse", "HEAD"], cwd=path)
            if actual != commit_hash:
                raise RuntimeError(
                    f"existing trial worktree points to {actual}, expected {commit_hash}"
                )
            return path
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(path), commit_hash],
            cwd=str(self.repository),
            check=True,
            capture_output=True,
            text=True,
            timeout=60.0,
            shell=False,
        )
        path = secure_directory(path)
        self._assert_pristine_checkout(path, commit_hash)
        return path


@dataclass(frozen=True)
class CommandOutcome:
    returncode: int
    elapsed_s: float
    stdout: str
    stderr: str
    metrics: Mapping[str, Any]
    timed_out: bool = False
    failure_reason: Optional[str] = None


class _BoundedOutputCapture:
    """Continuously drain child pipes while retaining only bounded byte tails."""

    def __init__(self, *, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._buffers = {"stdout": bytearray(), "stderr": bytearray()}
        self._truncated = {"stdout": False, "stderr": False}
        self._errors: list[str] = []
        self._streams: list[Any] = []
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> "_BoundedOutputCapture":
        return self

    def start(self, process: subprocess.Popen[Any]) -> None:
        if self._threads or process.stdout is None or process.stderr is None:
            raise RuntimeError("bounded output capture was not given two fresh pipes")
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            self._streams.append(stream)
            thread = threading.Thread(
                target=self._drain,
                args=(name, stream),
                name=f"aigp-{name}-drain",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _drain(self, name: str, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                buffer = self._buffers[name]
                buffer.extend(chunk)
                overflow = len(buffer) - self.max_bytes
                if overflow > 0:
                    del buffer[:overflow]
                    self._truncated[name] = True
        except (OSError, ValueError) as exc:
            self._errors.append(f"{name} drain failed: {type(exc).__name__}: {exc}")

    def finish(self) -> tuple[str, str, Optional[str]]:
        for thread in self._threads:
            thread.join(timeout=3.0)
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        if alive:
            for stream in self._streams:
                try:
                    stream.close()
                except OSError:
                    pass
            for thread in self._threads:
                thread.join(timeout=1.0)
            self._errors.append("output drain threads did not stop: " + ", ".join(alive))
        truncation_errors: list[str] = []
        for name, truncated in self._truncated.items():
            if truncated:
                truncation_errors.append(
                    f"{name} exceeded {self.max_bytes} bytes; only its bounded tail was retained"
                )
        stderr_notes = [*self._errors, *truncation_errors]
        stdout = bytes(self._buffers["stdout"]).decode("utf-8", errors="replace")
        stderr = bytes(self._buffers["stderr"]).decode("utf-8", errors="replace")
        if stderr_notes:
            stderr = bounded_tail([stderr, *stderr_notes])
        # A retained tail is diagnostic output, not the complete stdout JSON
        # document.  Never parse or promote metrics after either pipe exceeded
        # the declared evidence bound.
        failures = [*self._errors, *truncation_errors]
        failure = "; ".join(failures) if failures else None
        return stdout, stderr, failure

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for stream in self._streams:
            try:
                stream.close()
            except OSError:
                pass


class _WindowsJobContainment:
    """Fail-closed Windows Job Object with kill-on-close descendant cleanup."""

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows Job containment is Windows-only")
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class BasicAccountingInformation(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_int64),
                ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._accounting_type = BasicAccountingInformation
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        self._kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        self._kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        self._ntdll.NtResumeProcess.restype = ctypes.c_long

        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle = handle
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self._kernel32.SetInformationJobObject(
            handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach_and_resume(self, process: subprocess.Popen[Any]) -> None:
        process_handle = self._wintypes.HANDLE(int(process._handle))
        if not self._kernel32.AssignProcessToJobObject(
            self._handle, process_handle
        ):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        status = int(self._ntdll.NtResumeProcess(process_handle))
        if status != 0:
            raise RuntimeError(f"NtResumeProcess failed with NTSTATUS 0x{status & 0xffffffff:08x}")

    def _active_processes(self) -> int:
        information = self._accounting_type()
        if not self._kernel32.QueryInformationJobObject(
            self._handle,
            self._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            self._ctypes.byref(information),
            self._ctypes.sizeof(information),
            None,
        ):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return int(information.ActiveProcesses)

    def terminate_and_prove(self, process: subprocess.Popen[Any]) -> None:
        error: Optional[BaseException] = None
        try:
            if not self._kernel32.TerminateJobObject(self._handle, 125):
                raise self._ctypes.WinError(self._ctypes.get_last_error())
            deadline = time.monotonic() + 5.0
            while self._active_processes() != 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            if self._active_processes() != 0:
                raise RuntimeError("Windows Job still contains active descendants")
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("contained parent did not terminate") from exc
        except BaseException as exc:
            error = exc
        finally:
            self.close()
        if error is not None:
            raise RuntimeError(f"Windows Job cleanup was not proved: {error}") from error

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._handle = None
            self._kernel32.CloseHandle(handle)


class TrialScheduler:
    """Lease one pending trial and resume it from the first missing tier."""

    def __init__(
        self,
        ledger: TrialLedger,
        worktrees: GitWorktreePool,
        commands: Mapping[Tier, TierCommand],
        *,
        owner: Optional[str] = None,
        lease_ttl_s: float = 30.0,
    ) -> None:
        self.ledger = ledger
        self.worktrees = worktrees
        self.commands = dict(commands)
        self.owner = owner or f"scheduler-{uuid.uuid4().hex}"
        if not math.isfinite(lease_ttl_s) or lease_ttl_s < 3.0:
            raise ValueError("lease_ttl_s must be finite and >=3 seconds")
        self.lease_ttl_s = float(lease_ttl_s)
        self.runtime_environment_fingerprint = environment_fingerprint()
        if Tier.T5_AUTHORIZED_LIVE in self.commands:
            raise ValueError("T5 is forbidden in the ordinary scheduler")
        t1_command = self.commands.get(Tier.T1_VQ2_REPLAY)
        if t1_command is not None:
            _validate_t1_command(t1_command)
        for tier in (
            Tier.T2_WARM_SIM,
            Tier.T3_DOMAIN_TRACKS,
            Tier.T4_FULL_NON_LIVE,
        ):
            command = self.commands.get(tier)
            if command is not None and any(
                not step.trusted_files_sha256 for step in command.steps
            ):
                raise ValueError(
                    f"{tier.name} requires trusted evaluator file binding on every step"
                )

    def _ensure_worktree(self, trial_id: str, commit_hash: str) -> Path:
        """Keep trial/global leases alive during potentially slow Git setup."""

        stop = threading.Event()
        failures: list[str] = []
        heartbeat_period = min(5.0, self.lease_ttl_s / 3.0)

        def renew() -> None:
            while not stop.wait(heartbeat_period):
                try:
                    self.ledger.heartbeat(
                        trial_id, self.owner, ttl_s=self.lease_ttl_s
                    )
                    if stop.is_set():
                        return
                    if not self.ledger.acquire_global_lease(
                        _ORCHESTRATION_LEASE, self.owner, ttl_s=self.lease_ttl_s
                    ):
                        raise RuntimeError("lost singleton lease during worktree setup")
                except Exception as exc:
                    failures.append(f"{type(exc).__name__}: {exc}")
                    return

        thread = threading.Thread(
            target=renew,
            name="aigp-scheduler-setup-heartbeat",
            daemon=True,
        )
        thread.start()
        try:
            worktree = self.worktrees.ensure(trial_id, commit_hash)
        finally:
            stop.set()
            thread.join(timeout=max(1.0, heartbeat_period * 2.0))
        if thread.is_alive():
            raise RuntimeError("worktree heartbeat thread did not stop")
        if failures:
            raise RuntimeError(f"worktree lease heartbeat failed: {failures[0]}")
        return worktree

    def _checkpoint_phase_timings(self, trial_id: str) -> Dict[str, float]:
        timings: Dict[str, float] = {}
        for tier in Tier:
            checkpoint = self.ledger.get_checkpoint(trial_id, int(tier))
            if checkpoint is None:
                continue
            elapsed = checkpoint.get("elapsed_s")
            if type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed >= 0:
                timings[tier.name] = float(elapsed)
        return timings

    @staticmethod
    def _outcome_artifacts(metrics: Mapping[str, Any]) -> Dict[str, str]:
        artifacts = {"metrics_sha256": json_hash(metrics)}
        declared = metrics.get("artifact_hashes")
        if declared is not None:
            if type(declared) is not dict:
                raise ValueError("artifact_hashes evidence must be an exact object")
            for name, digest in declared.items():
                if (
                    type(name) is not str
                    or not name
                    or type(digest) is not str
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError("artifact hashes must be named SHA-256 digests")
                artifacts[f"declared.{name}"] = digest
        return artifacts

    def _checkpoint_artifacts(self, trial_id: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for tier in Tier:
            checkpoint = self.ledger.get_checkpoint(trial_id, int(tier))
            if checkpoint is None:
                continue
            for name, digest in checkpoint["artifact_hashes"].items():
                result[f"{tier.name}.{name}"] = digest
        return result

    def _finalize_durable_failed_checkpoint(
        self, trial: Mapping[str, Any], *, through: Tier
    ) -> bool:
        """Reconcile a checkpoint committed just before a worker crash."""

        trial_id = str(trial["trial_id"])
        checkpoints = [
            self.ledger.get_checkpoint(trial_id, tier)
            for tier in range(int(through) + 1)
        ]
        failed = next(
            (item for item in checkpoints if item is not None and item["status"] == "failed"),
            None,
        )
        if failed is None:
            return False
        terminal_tier = Tier(int(failed["tier"]))
        tier_metrics = {
            Tier(int(item["tier"])).name: item["metrics"]
            for item in checkpoints
            if item is not None
        }
        self.ledger.finish_trial(
            trial_id,
            self.owner,
            success=False,
            phase_timings=self._checkpoint_phase_timings(trial_id),
            safety_and_completion_metrics={
                "tiers": tier_metrics,
                "terminal_tier": terminal_tier.name,
                "terminal_metrics": failed["metrics"],
                "reconciled_from_durable_checkpoint": True,
            },
            artifact_hashes=self._checkpoint_artifacts(trial_id),
            failure_reason=f"{terminal_tier.name} has a durable failed checkpoint",
            stdout_stderr_tail=failed.get("stdout_stderr_tail"),
            worktree_path=trial.get("worktree_path"),
        )
        return True

    @staticmethod
    def _tier_identity_hash(
        trial: Mapping[str, Any], tier: Tier, *, required: bool
    ) -> Optional[str]:
        config = trial.get("resolved_config")
        manifest = (
            config.get("promotion_ladder_manifest")
            if isinstance(config, Mapping)
            else None
        )
        if manifest is None:
            if required:
                raise RuntimeError(
                    "trusted promotion commands require a full-ladder manifest"
                )
            return None
        if (
            type(manifest) is not dict
            or set(manifest) != {"schema", "tiers"}
            or manifest.get("schema") != "aigp-promotion-ladder-manifest/2"
            or type(manifest.get("tiers")) is not list
            or len(manifest["tiers"]) != 5
        ):
            raise RuntimeError("promotion ladder manifest has an invalid exact schema")
        identities: Dict[int, Mapping[str, Any]] = {}
        expected_fields = {
            "tier",
            "dataset_hash",
            "config_hash",
            "seed",
            "repetitions",
            "evaluator_version",
            "command_plan_sha256",
        }
        for identity in manifest["tiers"]:
            if type(identity) is not dict or set(identity) != expected_fields:
                raise RuntimeError("promotion tier identity is incomplete or has unknown keys")
            number = identity["tier"]
            if type(number) is not int or number not in range(5) or number in identities:
                raise RuntimeError("promotion tier identity numbers must be unique 0..4")
            for hash_name in ("dataset_hash", "config_hash"):
                digest = identity[hash_name]
                if (
                    type(digest) is not str
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise RuntimeError(f"promotion {hash_name} must be SHA-256")
            if type(identity["seed"]) is not int:
                raise RuntimeError("promotion seed must be an exact integer")
            if type(identity["repetitions"]) is not int or identity["repetitions"] < 1:
                raise RuntimeError("promotion repetitions must be a positive exact integer")
            if (
                type(identity["evaluator_version"]) is not str
                or not identity["evaluator_version"].strip()
            ):
                raise RuntimeError("promotion evaluator version must be non-empty")
            command_plan = identity["command_plan_sha256"]
            if (
                type(command_plan) is not str
                or len(command_plan) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in command_plan
                )
            ):
                raise RuntimeError("promotion command plan must be SHA-256")
            identities[number] = identity
        if set(identities) != set(range(5)):
            raise RuntimeError("promotion ladder manifest must bind T0 through T4")
        manifest_hash = json_hash(manifest)
        if trial.get("dataset_hash") != manifest_hash:
            raise RuntimeError("trial dataset_hash does not bind the full-ladder manifest")
        if trial.get("evaluator_version") != f"aigp-ladder/2:{manifest_hash}":
            raise RuntimeError(
                "trial evaluator_version does not bind the full-ladder manifest"
            )
        return json_hash(identities[int(tier)])

    @staticmethod
    def _frozen_command_plan_hash(
        trial: Mapping[str, Any], tier: Tier, *, required: bool
    ) -> Optional[str]:
        identity_hash = TrialScheduler._tier_identity_hash(
            trial, tier, required=required
        )
        if identity_hash is None:
            return None
        manifest = trial["resolved_config"]["promotion_ladder_manifest"]
        identity = next(
            item for item in manifest["tiers"] if item["tier"] == int(tier)
        )
        return str(identity["command_plan_sha256"])

    def _artifacts_for_tier(
        self,
        trial: Mapping[str, Any],
        tier: Tier,
        metrics: Mapping[str, Any],
        *,
        trusted_required: bool,
    ) -> Dict[str, str]:
        artifacts = self._outcome_artifacts(metrics)
        base_identity_hash = self._tier_identity_hash(
            trial, tier, required=trusted_required
        )
        if base_identity_hash is not None:
            command = self.commands.get(tier)
            assert command is not None
            artifacts["manifest_tier_identity_sha256"] = base_identity_hash
            command_plan_hash = json_hash(dataclasses.asdict(command))
            frozen_command_plan_hash = self._frozen_command_plan_hash(
                trial, tier, required=True
            )
            if command_plan_hash != frozen_command_plan_hash:
                raise RuntimeError(
                    f"{tier.name} command plan does not match the frozen TrialKey"
                )
            artifacts["command_plan_sha256"] = command_plan_hash
            artifacts["tier_identity_sha256"] = json_hash(
                {
                    "manifest_tier_identity_sha256": base_identity_hash,
                    "command_plan_sha256": command_plan_hash,
                }
            )
            if tier >= Tier.T2_WARM_SIM:
                trusted = dict(
                    pair
                    for step in command.steps
                    for pair in step.trusted_files_sha256
                )
                if not trusted:
                    raise RuntimeError(
                        f"{tier.name} cannot checkpoint without trusted evaluator files"
                    )
                artifacts["trusted_evaluator_files_sha256"] = json_hash(trusted)
        return artifacts

    def _verify_completed_checkpoint_identities(
        self, trial: Mapping[str, Any]
    ) -> None:
        config = trial.get("resolved_config")
        has_manifest = isinstance(config, Mapping) and (
            "promotion_ladder_manifest" in config
        )
        for completed_tier in self.ledger.completed_tiers(str(trial["trial_id"])):
            tier = Tier(completed_tier)
            command = self.commands.get(tier)
            trusted = bool(
                command
                and any(step.trusted_files_sha256 for step in command.steps)
            )
            base_expected = self._tier_identity_hash(
                trial, tier, required=trusted or has_manifest
            )
            frozen_command = self._frozen_command_plan_hash(
                trial, tier, required=trusted or has_manifest
            )
            actual_command = (
                json_hash(dataclasses.asdict(command))
                if command is not None
                else None
            )
            if frozen_command is not None and actual_command != frozen_command:
                raise RuntimeError(
                    f"configured {tier.name} command plan differs from the frozen TrialKey"
                )
            expected = (
                json_hash(
                    {
                        "manifest_tier_identity_sha256": base_expected,
                        "command_plan_sha256": frozen_command,
                    }
                )
                if base_expected is not None and command is not None
                else base_expected
            )
            checkpoint = self.ledger.get_checkpoint(
                str(trial["trial_id"]), completed_tier
            )
            if checkpoint is None:
                raise RuntimeError(f"completed {tier.name} checkpoint is missing")
            if checkpoint["artifact_hashes"].get("metrics_sha256") != json_hash(
                checkpoint["metrics"]
            ):
                raise RuntimeError(
                    f"completed {tier.name} checkpoint has stale metrics identity"
                )
            binding_failure = self._tier_evidence_binding_failure(
                trial, tier, checkpoint["metrics"]
            )
            if binding_failure is not None:
                raise RuntimeError(
                    f"completed {tier.name} checkpoint evidence is invalid: "
                    f"{binding_failure}"
                )
            if expected is not None and (
                checkpoint["artifact_hashes"].get(
                    "manifest_tier_identity_sha256"
                )
                != base_expected
                or checkpoint["artifact_hashes"].get("tier_identity_sha256")
                != expected
                or (
                    command is not None
                    and checkpoint["artifact_hashes"].get("command_plan_sha256")
                    != frozen_command
                )
            ):
                raise RuntimeError(
                    f"completed {tier.name} checkpoint has stale tier identity"
                )

    @staticmethod
    def _find_schema_evidence(
        metrics: Mapping[str, Any], schemas: set[str]
    ) -> Optional[Mapping[str, Any]]:
        from .evidence import find_unique_schema_evidence

        return find_unique_schema_evidence(metrics, schemas)

    def _tier_evidence_binding_failure(
        self,
        trial: Mapping[str, Any],
        tier: Tier,
        metrics: Mapping[str, Any],
    ) -> Optional[str]:
        # Domain-scope validation is independent of a frozen ladder manifest.
        # In particular, T0 must not bypass the non-flight claim boundary via
        # the manifest/T0 early return below.
        from .evidence import validate_tier_evidence

        try:
            evidence = validate_tier_evidence(tier, metrics)
        except (TypeError, ValueError) as exc:
            return f"{tier.name} tier evidence scope is invalid: {exc}"
        config = trial.get("resolved_config")
        manifest = (
            config.get("promotion_ladder_manifest")
            if isinstance(config, Mapping)
            else None
        )
        if manifest is None or tier is Tier.T0_AFFECTED:
            return None
        # Full validation, including TrialKey binding, happens here before
        # evidence can be checkpointed as completed.
        self._tier_identity_hash(trial, tier, required=True)
        identity = next(
            item for item in manifest["tiers"] if item["tier"] == int(tier)
        )
        if tier >= Tier.T2_WARM_SIM:
            from .nonlive import DOMAIN_TRACK_SET, FULL_TRACK_SET

            required_tracks = {
                Tier.T2_WARM_SIM: ("race_01",),
                Tier.T3_DOMAIN_TRACKS: DOMAIN_TRACK_SET,
                Tier.T4_FULL_NON_LIVE: FULL_TRACK_SET,
            }[tier]
            observed_tracks = evidence.get("track_identity")
            if (
                evidence.get("tier") != int(tier)
                or type(observed_tracks) is not list
                or any(type(name) is not str for name in observed_tracks)
                or tuple(sorted(observed_tracks))
                != tuple(sorted(required_tracks))
            ):
                return f"{tier.name} evidence has the wrong tier/track identity"
        observed = {
            "dataset_hash": evidence.get(
                "evaluation_input_hash", evidence.get("evaluation_evidence_hash")
            ),
            "config_hash": evidence.get("evaluation_config_sha256"),
            "seed": evidence.get("seed"),
            "repetitions": evidence.get("repetitions"),
            "evaluator_version": evidence.get("evaluator_version"),
        }
        expected = {
            name: identity[name]
            for name in (
                "dataset_hash",
                "config_hash",
                "seed",
                "repetitions",
                "evaluator_version",
            )
        }
        if observed != expected:
            return (
                f"{tier.name} evaluator evidence identity mismatch: "
                f"expected {expected!r}, observed {observed!r}"
            )
        if (
            tier is Tier.T1_VQ2_REPLAY
            and evidence.get("processor_code_sha256") != trial.get("code_hash")
        ):
            return "T1 processor code hash does not match the candidate TrialKey"
        if tier >= Tier.T2_WARM_SIM:
            declared_sources = evidence.get("evaluator_identity")
            declared_sources = (
                declared_sources.get("source_sha256")
                if isinstance(declared_sources, Mapping)
                else None
            )
            command = self.commands[tier]
            trusted = dict(
                pair
                for step in command.steps
                for pair in step.trusted_files_sha256
            )
            if not trusted:
                return f"{tier.name} command lacks trusted evaluator file hashes"
            if (
                type(declared_sources) is not dict
                or any(declared_sources.get(name) != digest for name, digest in trusted.items())
            ):
                return f"{tier.name} evidence does not match trusted evaluator file hashes"
        return None

    def _materialize_config(self, trial: Mapping[str, Any]) -> Path:
        config = trial["resolved_config"]
        if json_hash(config) != trial["config_hash"]:
            raise RuntimeError("ledger resolved_config no longer matches config_hash")
        root = getattr(self.worktrees, "root", None)
        if root is None:
            raise RuntimeError("worktree pool does not expose an external config root")
        directory = Path(root).resolve() / ".trial-configs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{trial['trial_id']}-{trial['config_hash']}.json"
        encoded = (canonical_json(config) + "\n").encode("utf-8")
        if path.exists():
            if path.read_bytes() != encoded:
                raise RuntimeError("materialized config content mismatch")
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            return path
        fd, raw_temp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=directory
        )
        temp = Path(raw_temp)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return path

    @staticmethod
    def _verify_trial_inputs(
        trial: Mapping[str, Any], worktree: Path, config_path: Path
    ) -> None:
        expected_config = (canonical_json(trial["resolved_config"]) + "\n").encode(
            "utf-8"
        )
        if (
            config_path.is_symlink()
            or not config_path.is_file()
            or config_path.read_bytes() != expected_config
            or json_hash(trial["resolved_config"]) != trial["config_hash"]
        ):
            raise RuntimeError("materialized config integrity drifted")
        actual_commit, actual_dirty, actual_code = git_provenance(worktree)
        if (
            actual_commit != trial["commit_hash"]
            or actual_dirty != trial["dirty_diff_hash"]
            or actual_code != trial["code_hash"]
        ):
            raise RuntimeError("isolated worktree provenance drifted")

    @staticmethod
    def _worktree_inventory_hash(worktree: Path) -> str:
        """Bind tracked, untracked, ignored, and empty-directory state."""

        root = Path(worktree).resolve(strict=True)
        digest = hashlib.sha256(b"aigp-worktree-inventory/1\0")
        for directory, names, files in os.walk(root, topdown=True, followlinks=False):
            base = Path(directory)
            names.sort()
            files.sort()
            for name in names:
                target = base / name
                info = target.lstat()
                if stat.S_ISLNK(info.st_mode) or (
                    getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                ):
                    raise RuntimeError("worktree contains a symlink/reparse directory")
                relative = target.relative_to(root).as_posix().encode("utf-8")
                digest.update(b"D\0" + len(relative).to_bytes(8, "big") + relative)
            for name in files:
                target = base / name
                try:
                    payload = read_secure_regular_file(target)
                except ValueError as exc:
                    raise RuntimeError(
                        "worktree contains an unsafe or unstable file"
                    ) from exc
                relative = target.relative_to(root).as_posix().encode("utf-8")
                digest.update(b"F\0" + len(relative).to_bytes(8, "big") + relative)
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
        return digest.hexdigest()

    def _prepare_trial(self, trial: Mapping[str, Any]) -> tuple[Path, Path]:
        trial_id = str(trial["trial_id"])
        worktree = self._ensure_worktree(trial_id, str(trial["commit_hash"]))
        actual_commit, actual_dirty, actual_code = git_provenance(worktree)
        if (
            actual_commit != trial["commit_hash"]
            or actual_dirty != trial["dirty_diff_hash"]
            or actual_code != trial["code_hash"]
        ):
            raise RuntimeError(
                "isolated worktree provenance does not match ledger candidate"
            )
        if trial["environment_fingerprint"] != self.runtime_environment_fingerprint:
            raise RuntimeError("runtime environment fingerprint drifted since enqueue")
        config = trial.get("resolved_config")
        has_manifest = isinstance(config, Mapping) and (
            "promotion_ladder_manifest" in config
        )
        for tier, command in self.commands.items():
            trusted = any(step.trusted_files_sha256 for step in command.steps)
            frozen_command = self._frozen_command_plan_hash(
                trial, tier, required=trusted or has_manifest
            )
            if (
                frozen_command is not None
                and frozen_command != json_hash(dataclasses.asdict(command))
            ):
                raise RuntimeError(
                    f"configured {tier.name} command plan differs from the frozen TrialKey"
                )
        self._verify_completed_checkpoint_identities(trial)
        config_path = self._materialize_config(trial)
        self._verify_trial_inputs(trial, worktree, config_path)
        return worktree, config_path

    @staticmethod
    def _verify_trusted_files(worktree: Path, command: CommandStep) -> None:
        root = worktree.resolve()
        for relative, expected in command.trusted_files_sha256:
            try:
                target = secure_relative_regular_file(root, relative)
            except ValueError as exc:
                raise RuntimeError(
                    f"trusted evaluator file is missing or unsafe: {relative}"
                ) from exc

            actual = sha256_bytes(read_secure_regular_file(target))
            if actual != expected:
                raise RuntimeError(
                    f"trusted evaluator hash mismatch for {relative}: "
                    f"expected {expected}, observed {actual}"
                )

    @staticmethod
    def _verify_isolation_wrapper(
        command: CommandStep, *, attest: bool = True
    ) -> None:
        if command.isolation_wrapper is None:
            return
        wrapper = Path(command.isolation_wrapper)
        try:
            resolved_wrapper = secure_regular_file(wrapper)
            digest = sha256_bytes(read_secure_regular_file(resolved_wrapper))
        except ValueError as exc:
            raise RuntimeError(
                "OS isolation wrapper path/hash is missing or mismatched"
            ) from exc
        if not wrapper.is_absolute() or digest != command.isolation_wrapper_sha256:
            raise RuntimeError("OS isolation wrapper path/hash is missing or mismatched")
        if not attest:
            return
        try:
            run = subprocess.run(
                [str(resolved_wrapper), "--attest"],
                capture_output=True,
                text=True,
                timeout=5.0,
                shell=False,
            )
            attestation = strict_json_loads(run.stdout)
        except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
            raise RuntimeError("OS isolation wrapper attestation failed") from exc
        expected = {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "denied",
            "filesystem": "readonly-worktree-only",
            "non_interactive": True,
            "process_tree_containment": "kill-on-wrapper-exit",
            "host_process_access": "denied",
        }
        if run.returncode != 0 or type(attestation) is not dict or attestation != expected:
            raise RuntimeError("OS isolation wrapper attestation is insufficient")

    def _run_command(
        self,
        trial_id: str,
        command: CommandStep,
        worktree: Path,
        *,
        tier: Tier,
        config_path: Path,
        config_hash: str,
    ) -> CommandOutcome:
        started = time.monotonic()
        try:
            trusted_repository = Path(self.worktrees.repository).resolve()
            verification_root = (
                trusted_repository if command.trusted_host else worktree
            )
            self._verify_trusted_files(verification_root, command)
            self._verify_isolation_wrapper(command)
        except Exception as exc:
            return CommandOutcome(
                returncode=126,
                elapsed_s=time.monotonic() - started,
                stdout="",
                stderr=f"trusted evaluator verification failed: {type(exc).__name__}: {exc}",
                metrics={},
                failure_reason="trusted evaluator verification failed",
            )
        # Candidate processes receive only runtime essentials and explicit
        # trial inputs.  API/cloud credentials inherited by the scheduler are
        # not exposed to an offline evaluation subprocess.
        allowed_environment = {
            "PATH",
            "SystemRoot",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
            "LANG",
            "LC_ALL",
        }
        allowed_environment_folded = {
            name.casefold() for name in allowed_environment
        }
        environment = {
            name: value
            for name, value in os.environ.items()
            if name.casefold() in allowed_environment_folded
        }
        environment.update(
            {
                "AIGP_TRIAL_ID": trial_id,
                "AIGP_PROMOTION_TIER": str(int(tier)),
                "AIGP_TRIAL_OFFLINE": "1",
                "AIGP_RESOLVED_CONFIG": str(config_path),
                "AIGP_CONFIG_HASH": config_hash,
                "AIGP_CANDIDATE_WORKTREE": str(worktree.resolve()),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                # A shared external content-addressed cache makes T2 genuinely
                # warm across isolated worktrees.  Cache keys bind source,
                # resolved inputs, evaluator, and numeric dependencies; the
                # worktree pool root keeps independent scheduler instances
                # isolated from one another.
                "AIGP_CACHE_ROOT": str(
                    (Path(self.worktrees.root).resolve() / ".artifact-cache")
                ),
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        # Dedicated readers continuously drain both pipes to avoid child
        # backpressure, but retain only fixed-size tails in scheduler memory.
        # This bounds both RAM and disk use even for hostile candidate output.
        with _BoundedOutputCapture(max_bytes=1_000_000) as output_capture:
            launch_options: Dict[str, Any] = {}
            containment: Optional[_WindowsJobContainment] = None
            if os.name == "nt":
                try:
                    containment = _WindowsJobContainment()
                except Exception as exc:
                    return CommandOutcome(
                        returncode=125,
                        elapsed_s=time.monotonic() - started,
                        stdout="",
                        stderr=(
                            "Windows Job containment setup failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        metrics={},
                        failure_reason="process containment unavailable",
                    )
                launch_options["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | 0x00000004  # CREATE_SUSPENDED
                )
            else:
                launch_options["start_new_session"] = True
            try:
                # The attested wrapper is invoked again below; bind the final
                # path and bytes immediately before process creation.
                self._verify_isolation_wrapper(command, attest=False)
                process = subprocess.Popen(
                    list(
                        command.resolved_argv(
                            config_path,
                            trusted_repository=trusted_repository,
                        )
                    ),
                    cwd=str(
                        trusted_repository if command.trusted_host else worktree
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    env=environment,
                    **launch_options,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                if containment is not None:
                    containment.close()
                return CommandOutcome(
                    returncode=127,
                    elapsed_s=time.monotonic() - started,
                    stdout="",
                    stderr=f"command launch failed: {type(exc).__name__}: {exc}",
                    metrics={},
                    failure_reason="command launch failed",
                )
            output_capture.start(process)
            if containment is not None:
                try:
                    containment.attach_and_resume(process)
                except Exception as exc:
                    # The process was created suspended, so killing this exact
                    # parent is sufficient if assignment itself failed; it had
                    # no opportunity to create an uncontained descendant.
                    try:
                        process.kill()
                        process.wait(timeout=3.0)
                    finally:
                        containment.close()
                    return CommandOutcome(
                        returncode=125,
                        elapsed_s=time.monotonic() - started,
                        stdout="",
                        stderr=(
                            "Windows Job assignment failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        metrics={},
                        failure_reason="process containment unavailable",
                    )
            timed_out = False
            heartbeat_period = min(5.0, self.lease_ttl_s / 3.0)
            next_heartbeat = time.monotonic() + heartbeat_period
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if now - started > command.timeout_s:
                        timed_out = True
                        break
                    if now >= next_heartbeat:
                        self.ledger.heartbeat(
                            trial_id, self.owner, ttl_s=self.lease_ttl_s
                        )
                        if not self.ledger.acquire_global_lease(
                            _ORCHESTRATION_LEASE,
                            self.owner,
                            ttl_s=self.lease_ttl_s,
                        ):
                            raise RuntimeError("lost the singleton scheduler lease")
                        next_heartbeat = now + heartbeat_period
                    time.sleep(0.05)
            except BaseException as exc:
                try:
                    if containment is not None:
                        containment.terminate_and_prove(process)
                    else:
                        self._terminate_process_tree(process)
                except Exception as cleanup_exc:
                    raise RuntimeError(
                        "process containment cleanup failed during scheduler error"
                    ) from cleanup_exc
                raise exc
            containment_failure: Optional[str] = None
            try:
                if containment is not None:
                    containment.terminate_and_prove(process)
                else:
                    self._terminate_process_tree(process)
            except Exception as exc:
                containment_failure = f"{type(exc).__name__}: {exc}"

            stdout, stderr, capture_failure = output_capture.finish()
            if capture_failure is not None:
                stderr = bounded_tail(
                    [stderr, f"output capture integrity failed: {capture_failure}"]
                )
            if containment_failure is not None:
                stderr = bounded_tail(
                    [
                        stderr,
                        "process containment cleanup failed: "
                        f"{containment_failure}",
                    ]
                )
        elapsed = time.monotonic() - started
        returncode = process.returncode if process.returncode is not None else -9
        metrics: Mapping[str, Any] = {}
        if command.metrics_from_stdout and capture_failure is None:
            try:
                if not stdout.strip():
                    raise ValueError("metrics stdout is empty")
                decoded = strict_json_loads(stdout)
                if type(decoded) is dict and decoded:
                    # Validate nested values before the ledger checkpoint so
                    # malformed numeric evidence cannot strand a leased row.
                    canonical_json(decoded)
                    metrics = decoded
                else:
                    raise ValueError("metrics stdout is not a non-empty JSON object")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                stderr = bounded_tail([stderr, f"metrics parse error: {exc}"])
                returncode = returncode or 3
        return CommandOutcome(
            returncode=returncode,
            elapsed_s=elapsed,
            stdout=stdout,
            stderr=stderr,
            metrics=metrics,
            timed_out=timed_out,
            failure_reason=(
                "process containment cleanup failed"
                if containment_failure is not None
                else "output capture integrity failed"
                if capture_failure is not None
                else None
            ),
        )

    def _run_tier_command(
        self,
        trial_id: str,
        trial: Mapping[str, Any],
        command: TierCommand,
        worktree: Path,
        config_path: Path,
        config_hash: str,
    ) -> CommandOutcome:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        elapsed = 0.0
        merged_metrics: Dict[str, Any] = {}
        for index, step in enumerate(command.steps):
            try:
                self._verify_trial_inputs(trial, worktree, config_path)
                inventory_before = self._worktree_inventory_hash(worktree)
            except Exception as exc:
                return CommandOutcome(
                    125,
                    elapsed,
                    bounded_tail(stdout_parts),
                    bounded_tail(
                        stderr_parts
                        + [
                            "trial input verification failed before step "
                            f"{index}: {type(exc).__name__}: {exc}"
                        ]
                    ),
                    merged_metrics,
                    False,
                    "trial input integrity failed",
                )
            outcome = self._run_command(
                trial_id,
                step,
                worktree,
                tier=command.tier,
                config_path=config_path,
                config_hash=config_hash,
            )
            elapsed += outcome.elapsed_s
            stdout_parts.append(f"[step {index}]\n{outcome.stdout}")
            stderr_parts.append(f"[step {index}]\n{outcome.stderr}")
            try:
                self._verify_trial_inputs(trial, worktree, config_path)
                if self._worktree_inventory_hash(worktree) != inventory_before:
                    raise RuntimeError(
                        "candidate step mutated its isolated worktree inventory"
                    )
            except Exception as exc:
                return CommandOutcome(
                    125,
                    elapsed,
                    bounded_tail(stdout_parts),
                    bounded_tail(
                        stderr_parts
                        + [
                            "trial input verification failed after step "
                            f"{index}: {type(exc).__name__}: {exc}"
                        ]
                    ),
                    merged_metrics or outcome.metrics,
                    False,
                    "trial input integrity failed",
                )
            if outcome.metrics:
                merged_metrics[f"step_{index}"] = outcome.metrics
            gate_failure = self._hard_gate_failure(
                outcome.metrics, required=bool(step.require_hard_gates)
            )
            if (
                outcome.returncode != 0
                or outcome.timed_out
                or gate_failure
                or outcome.failure_reason is not None
            ):
                reason = outcome.failure_reason or gate_failure or (
                    f"step {index} timed out"
                    if outcome.timed_out
                    else f"step {index} exited {outcome.returncode}"
                )
                return CommandOutcome(
                    outcome.returncode,
                    elapsed,
                    bounded_tail(stdout_parts),
                    bounded_tail(stderr_parts),
                    merged_metrics or outcome.metrics,
                    outcome.timed_out,
                    reason,
                )
        final_metrics: Mapping[str, Any]
        if len(command.steps) == 1:
            # Preserve the command's exact evidence envelope for the common
            # one-step case.  Multi-step tiers are namespaced so independent
            # JSON documents cannot silently overwrite one another.
            final_metrics = next(iter(merged_metrics.values()), {})
        else:
            final_metrics = merged_metrics
        return CommandOutcome(
            0,
            elapsed,
            bounded_tail(stdout_parts),
            bounded_tail(stderr_parts),
            final_metrics,
            False,
            None,
        )

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
        """Boundedly terminate the exact trial process and its descendants."""

        if os.name == "nt":
            if process.poll() is not None:
                raise RuntimeError(
                    "cannot prove descendants absent after an uncontained parent exited"
                )
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=5.0,
                shell=False,
            )
            if result.returncode != 0:
                try:
                    process.kill()
                    process.wait(timeout=2.0)
                finally:
                    raise RuntimeError(
                        "taskkill did not prove descendant-tree termination: "
                        f"exit {result.returncode}"
                    )
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("taskkill returned but parent remains alive") from exc
            return
        group_id = process.pid

        def group_exists() -> bool:
            try:
                os.killpg(group_id, 0)
            except ProcessLookupError:
                return False
            return True

        def reap_parent() -> None:
            # ``poll`` performs a non-blocking waitpid on POSIX.  Without it,
            # an exited group leader remains a zombie and can keep
            # ``killpg(..., 0)`` reporting the group as present forever.
            process.poll()

        def prove_gone(deadline: float) -> bool:
            while time.monotonic() < deadline:
                reap_parent()
                if not group_exists():
                    if process.returncode is None:
                        try:
                            process.wait(
                                timeout=max(0.0, deadline - time.monotonic())
                            )
                        except subprocess.TimeoutExpired:
                            return False
                    return process.returncode is not None
                time.sleep(0.01)
            # One final reap/probe closes the boundary race at the deadline.
            reap_parent()
            return not group_exists() and process.returncode is not None

        try:
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            if process.returncode is None:
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        "POSIX process group vanished but parent remains alive"
                    ) from exc
            return
        if prove_gone(time.monotonic() + 1.0):
            return
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            reap_parent()
            if process.returncode is not None:
                return
        if prove_gone(time.monotonic() + 3.0):
            return
        raise RuntimeError("POSIX process-group descendants remain after SIGKILL")

    @staticmethod
    def _hard_gate_failure(
        metrics: Mapping[str, Any], *, required: bool
    ) -> Optional[str]:
        safety = metrics.get("safety_and_completion_metrics", metrics)
        if not isinstance(safety, Mapping):
            return "invalid safety/completion metrics"
        recognized = {
            "valid",
            "completed",
            "correct_gate_sequence",
            "cleanup_confirmed",
            "no_collision",
            "no_disqualification",
            "no_stale_stream_flight",
            "collision",
            "disqualified",
            "stale_stream_flight",
            "cleanup_failure",
        }
        non_boolean = sorted(
            name for name in recognized if name in safety and type(safety[name]) is not bool
        )
        if non_boolean:
            return "non-boolean hard-gate evidence: " + ", ".join(non_boolean)
        positive_flags = (
            "valid",
            "completed",
            "correct_gate_sequence",
            "cleanup_confirmed",
            "no_collision",
            "no_disqualification",
            "no_stale_stream_flight",
        )
        false_positive_flags = [
            name for name in positive_flags if name in safety and safety[name] is False
        ]
        if false_positive_flags:
            return "failed hard gates: " + ", ".join(false_positive_flags)
        if required:
            required_flags = positive_flags
            missing = [name for name in required_flags if name not in safety]
            if missing:
                return "missing hard-gate evidence: " + ", ".join(missing)
            failed_positive = [name for name in required_flags if safety[name] is not True]
            if failed_positive:
                return "failed hard gates: " + ", ".join(failed_positive)
        negative_flags = (
            "collision",
            "disqualified",
            "stale_stream_flight",
            "cleanup_failure",
        )
        failures = [name for name in negative_flags if safety.get(name) is True]
        required_true = (
            "valid",
            "correct_gate_sequence",
            "completed",
            "cleanup_confirmed",
        )
        failures.extend(name for name in required_true if name in safety and safety[name] is not True)
        return ", ".join(failures) if failures else None

    def run_once(self, *, through: Tier = Tier.T4_FULL_NON_LIVE) -> Optional[str]:
        if through is Tier.T5_AUTHORIZED_LIVE:
            raise PermissionError("T5 requires the explicitly authorized campaign runner")
        if not self.ledger.acquire_global_lease(
            _ORCHESTRATION_LEASE, self.owner, ttl_s=self.lease_ttl_s
        ):
            raise RuntimeError("another scheduler owns the active lease")
        trial_id: Optional[str] = None
        try:
            # Failed checkpoints are durable terminal outcomes.  They are not
            # counted by ``completed_tiers``, so reconcile one before the
            # ordinary next-tier query can accidentally schedule it again.
            for candidate in self.ledger.list_trials():
                if candidate["status"] not in {"pending", "running"}:
                    continue
                if not any(
                    (checkpoint := self.ledger.get_checkpoint(
                        candidate["trial_id"], tier
                    ))
                    is not None
                    and checkpoint["status"] == "failed"
                    for tier in range(int(through) + 1)
                ):
                    continue
                if self.ledger.lease_trial(
                    candidate["trial_id"], self.owner, ttl_s=self.lease_ttl_s
                ):
                    trial_id = candidate["trial_id"]
                    break
            if trial_id is None:
                trial_id = self.ledger.claim_next(
                    self.owner, ttl_s=self.lease_ttl_s, through=int(through)
                )
            if trial_id is None:
                return None
            trial = self.ledger.get_trial(trial_id)
            if self._finalize_durable_failed_checkpoint(trial, through=through):
                return trial_id
            try:
                worktree, config_path = self._prepare_trial(trial)
            except Exception as exc:
                reason = f"trial setup failed: {type(exc).__name__}: {exc}"
                self.ledger.finish_trial(
                    trial_id,
                    self.owner,
                    success=False,
                    failure_reason=reason,
                )
                return trial_id
            phase_timings = self._checkpoint_phase_timings(trial_id)
            tier_metrics: Dict[str, Mapping[str, Any]] = {}
            for completed_tier in self.ledger.completed_tiers(trial_id):
                checkpoint = self.ledger.get_checkpoint(trial_id, completed_tier)
                if checkpoint is not None:
                    tier_metrics[Tier(completed_tier).name] = checkpoint["metrics"]
            combined_tail: list[str] = []
            last_metrics: Mapping[str, Any] = {}
            next_tier = self.ledger.next_tier(trial_id, through=int(through))
            while next_tier is not None:
                tier = Tier(next_tier)
                command = self.commands.get(tier)
                if command is None:
                    reason = f"no command configured for {tier.name}"
                    self.ledger.finish_trial(
                        trial_id,
                        self.owner,
                        success=False,
                        phase_timings=phase_timings,
                        failure_reason=reason,
                        stdout_stderr_tail=bounded_tail(combined_tail),
                        worktree_path=str(worktree),
                    )
                    return trial_id
                self.ledger.checkpoint(
                    trial_id, int(tier), owner=self.owner, status="running"
                )
                outcome = self._run_tier_command(
                    trial_id,
                    trial,
                    command,
                    worktree,
                    config_path,
                    trial["config_hash"],
                )
                binding_failure = self._tier_evidence_binding_failure(
                    trial, tier, outcome.metrics
                )
                if binding_failure is not None and outcome.failure_reason is None:
                    outcome = dataclasses.replace(
                        outcome, failure_reason=binding_failure
                    )
                if tier is Tier.T1_VQ2_REPLAY and outcome.failure_reason is None:
                    eligibility = self._promotion_evaluation(
                        trial_id, tier, outcome.metrics
                    ).eligibility
                    assert eligibility is not None
                    if eligibility.passed is not True:
                        outcome = dataclasses.replace(
                            outcome,
                            failure_reason=(
                                "T1 replay eligibility failed: "
                                + ", ".join(eligibility.failures)
                            ),
                        )
                phase_timings[tier.name] = outcome.elapsed_s
                combined_tail.extend((outcome.stdout, outcome.stderr))
                tail = bounded_tail((outcome.stdout, outcome.stderr))
                gate_failure = outcome.failure_reason
                passed = outcome.returncode == 0 and not outcome.timed_out and gate_failure is None
                self.ledger.checkpoint(
                    trial_id,
                    int(tier),
                    owner=self.owner,
                    status="completed" if passed else "failed",
                    metrics=outcome.metrics,
                    artifact_hashes=self._artifacts_for_tier(
                        trial,
                        tier,
                        outcome.metrics,
                        trusted_required=any(
                            step.trusted_files_sha256 for step in command.steps
                        ),
                    ),
                    stdout_stderr_tail=tail,
                    elapsed_s=outcome.elapsed_s,
                )
                last_metrics = outcome.metrics
                tier_metrics[tier.name] = outcome.metrics
                if not passed:
                    reason = gate_failure or (
                        f"{tier.name} timed out" if outcome.timed_out
                        else f"{tier.name} exited {outcome.returncode}"
                    )
                    self.ledger.finish_trial(
                        trial_id,
                        self.owner,
                        success=False,
                        phase_timings=phase_timings,
                        safety_and_completion_metrics={
                            "tiers": tier_metrics,
                            "terminal_tier": tier.name,
                            "terminal_metrics": outcome.metrics,
                        },
                        artifact_hashes=self._checkpoint_artifacts(trial_id),
                        failure_reason=reason,
                        stdout_stderr_tail=bounded_tail(combined_tail),
                        worktree_path=str(worktree),
                    )
                    return trial_id
                next_tier = self.ledger.next_tier(trial_id, through=int(through))
            if through < Tier.T4_FULL_NON_LIVE:
                self.ledger.yield_trial(trial_id, self.owner)
            else:
                self.ledger.finish_trial(
                    trial_id,
                    self.owner,
                    success=True,
                    phase_timings=phase_timings,
                    safety_and_completion_metrics={
                        "tiers": tier_metrics,
                        "terminal_tier": Tier(int(through)).name,
                        "terminal_metrics": last_metrics,
                    },
                    artifact_hashes=self._checkpoint_artifacts(trial_id),
                    stdout_stderr_tail=bounded_tail(combined_tail),
                    worktree_path=str(worktree),
                )
            return trial_id
        finally:
            self.ledger.release_global_lease(_ORCHESTRATION_LEASE, self.owner)

    @staticmethod
    def _promotion_evaluation(
        trial_id: str,
        tier: Tier,
        metrics: Mapping[str, Any],
    ) -> CandidateEvaluation:
        safe = HardGates(True, True, True, True, True, True, True)
        if tier is Tier.T0_AFFECTED:
            return CandidateEvaluation(
                trial_id,
                tier,
                quality=QualityVector(),
                metrics=dict(metrics),
                eligibility=TierEligibility("affected-tests", True),
            )

        if tier is Tier.T1_VQ2_REPLAY:
            replay_score = TrialScheduler._find_schema_evidence(
                metrics,
                {
                    "aigp-vq2-replay-score/1",
                    "aigp-vq2-replay-corpus-score/1",
                },
            )

            def exact_metric(value: Any, *, fallback: float) -> float:
                if (
                    type(value) in {int, float}
                    and math.isfinite(value)
                ):
                    return float(value)
                return fallback

            is_corpus = bool(
                isinstance(replay_score, Mapping)
                and replay_score.get("schema")
                == "aigp-vq2-replay-corpus-score/1"
            )
            perception = (
                replay_score.get("aggregate" if is_corpus else "perception", {})
                if isinstance(replay_score, Mapping)
                else {}
            )
            if not isinstance(perception, Mapping):
                perception = {}
            failures: list[str] = []
            policy = replay_score.get("policy") if isinstance(replay_score, Mapping) else None
            if type(policy) is not dict or policy.get("passed") is not True:
                failures.append("golden replay policy did not pass exactly")
            if isinstance(replay_score, Mapping):
                failures.extend(replay_promotion_policy_failures(replay_score))
            evidence_hash = (
                replay_score.get(
                    "evaluation_input_hash",
                    replay_score.get("evaluation_evidence_hash"),
                )
                if isinstance(replay_score, Mapping)
                else None
            )
            if (
                type(evidence_hash) is not str
                or len(evidence_hash) != 64
                or any(character not in "0123456789abcdef" for character in evidence_hash)
            ):
                evidence_hash = None
                failures.append("golden replay evaluation evidence hash is missing")
            if (
                not isinstance(replay_score, Mapping)
                or replay_score.get("processor") == "recorded"
                or not _is_sha256(replay_score.get("processor_code_sha256"))
                or not isinstance(replay_score.get("domain_provenance"), Mapping)
                or replay_score["domain_provenance"].get("perception")
                != "candidate_detector_on_all_decoded_frames"
                or replay_score["domain_provenance"].get("estimator")
                != "candidate_estimator_on_ordered_sanitized_stream"
                or replay_score["domain_provenance"].get("open_loop_commands")
                != "candidate_generator_on_ordered_sanitized_stream"
            ):
                failures.append("golden replay was not candidate-processor-derived")
            isolation = (
                replay_score.get("candidate_isolation")
                if isinstance(replay_score, Mapping)
                else None
            )
            if (
                type(isolation) is not dict
                or isolation.get("schema")
                != "aigp-replay-isolation-attestation/1"
                or isolation.get("network") != "denied"
                or isolation.get("filesystem") != "readonly-worktree-only"
                or isolation.get("non_interactive") is not True
                or isolation.get("process_tree_containment")
                != "kill-on-wrapper-exit"
                or isolation.get("host_process_access") != "denied"
                or not _is_sha256(isolation.get("wrapper_sha256"))
            ):
                failures.append("candidate replay did not prove OS isolation")
            if is_corpus:
                sessions = replay_score.get("sessions")
                if type(sessions) is not list or not sessions:
                    failures.append("golden replay corpus contains no retained sessions")
                else:
                    for session in sessions:
                        if (
                            type(session) is not dict
                            or type(session.get("evaluation_input_hash")) is not str
                            or type(session.get("policy")) is not dict
                            or session["policy"].get("passed") is not True
                        ):
                            failures.append(
                                "golden replay corpus has missing/failed session evidence"
                            )
                            break
            quality = QualityVector(
                completion_reliability=exact_metric(
                    perception.get("gate_recall"), fallback=0.0
                ),
                centering_margin=-exact_metric(
                    perception.get(
                        "worst_center_error_px_p95"
                        if is_corpus
                        else "center_error_px_p95"
                    ),
                    fallback=1.0e12,
                ),
                stability_margin=-exact_metric(
                    perception.get(
                        "worst_temporal_center_step_px_p95"
                        if is_corpus
                        else "temporal_center_step_px_p95"
                    ),
                    fallback=1.0e12,
                ),
            )
            return CandidateEvaluation(
                trial_id,
                tier,
                quality=quality,
                metrics=dict(metrics),
                eligibility=TierEligibility(
                    "golden-replay",
                    not failures,
                    evidence_hash=evidence_hash,
                    failures=tuple(failures),
                ),
            )
        promotion: Any = metrics.get("promotion")
        if promotion is None:
            for value in reversed(list(metrics.values())):
                if isinstance(value, Mapping) and "promotion" in value:
                    promotion = value["promotion"]
                    break
        try:
            if not isinstance(promotion, Mapping):
                raise ValueError("missing promotion evidence")
            gates = HardGates.from_mapping(promotion["hard_gates"])
            quality_raw = promotion["quality"]
            if not isinstance(quality_raw, Mapping):
                raise TypeError("quality must be an object")
            expected_quality = {
                "completion_reliability",
                "centering_margin",
                "stability_margin",
                "race_time_s",
            }
            if set(quality_raw) != expected_quality:
                raise ValueError("promotion quality fields are incomplete")
            quality = QualityVector(**quality_raw)
        except (KeyError, TypeError, ValueError):
            gates = HardGates(True, True, True, True, True, True, False)
            quality = QualityVector()
        return CandidateEvaluation(trial_id, tier, gates, quality, metrics=dict(metrics))

    def _validate_decided_promotion_round(
        self,
        round_row: Mapping[str, Any],
        tier: Tier,
    ) -> Dict[str, Any]:
        """Revalidate a durable decision before applying any side effects."""

        if (
            round_row.get("status") != "decided"
            or type(round_row.get("tier")) is not int
            or round_row["tier"] != int(tier)
        ):
            raise RuntimeError("promotion round is not a decided round for this tier")
        raw_members = round_row.get("member_trial_ids")
        if (
            type(raw_members) is not tuple
            or not raw_members
            or any(type(member) is not str or not member for member in raw_members)
            or len(raw_members) != len(set(raw_members))
        ):
            raise RuntimeError("promotion round member identity is invalid")
        members = set(raw_members)
        raw_decision = round_row.get("decision")
        expected_fields = {
            "tier",
            "keep_fraction",
            "minimum_survivors",
            "promoted",
            "rejected_hard_gate",
            "eliminated_by_halving",
            "next_tier",
            "failed_evaluation",
        }
        if type(raw_decision) is not dict or set(raw_decision) != expected_fields:
            raise RuntimeError("promotion round decision fields are invalid")
        if type(raw_decision["tier"]) is not int or raw_decision["tier"] != int(tier):
            raise RuntimeError("promotion round decision tier is stale")
        keep_fraction = raw_decision["keep_fraction"]
        minimum_survivors = raw_decision["minimum_survivors"]
        if (
            type(keep_fraction) is not float
            or not math.isfinite(keep_fraction)
            or not 0.0 < keep_fraction <= 1.0
            or (tier is Tier.T0_AFFECTED and keep_fraction != 1.0)
            or type(minimum_survivors) is not int
            or minimum_survivors < 1
        ):
            raise RuntimeError("promotion round halving policy is invalid")

        promoted = raw_decision["promoted"]
        eliminated = raw_decision["eliminated_by_halving"]
        rejected = raw_decision["rejected_hard_gate"]
        failed = raw_decision["failed_evaluation"]
        if any(
            type(values) is not list
            or any(type(value) is not str or not value for value in values)
            or len(values) != len(set(values))
            for values in (promoted, eliminated)
        ):
            raise RuntimeError("promotion round decision member lists are invalid")
        if (
            type(rejected) is not dict
            or any(type(candidate) is not str or not candidate for candidate in rejected)
            or any(
                type(reasons) is not list
                or not reasons
                or any(type(reason) is not str or not reason for reason in reasons)
                for reasons in rejected.values()
            )
        ):
            raise RuntimeError("promotion round hard-gate rejection map is invalid")
        if (
            type(failed) is not dict
            or any(type(candidate) is not str or not candidate for candidate in failed)
            or any(type(reason) is not str or not reason for reason in failed.values())
        ):
            raise RuntimeError("promotion round failed-evaluation map is invalid")
        partitions = [set(promoted), set(eliminated), set(rejected), set(failed)]
        if (
            set().union(*partitions) != members
            or sum(len(partition) for partition in partitions) != len(members)
        ):
            raise RuntimeError("promotion round decision does not partition its members")

        eligible: list[CandidateEvaluation] = []
        expected_rejected: Dict[str, list[str]] = {}
        for trial_id in raw_members:
            row = self.ledger.get_trial(trial_id)
            # This rechecks every immutable completed checkpoint, including its
            # metrics hash and tier-domain scope, on the decided/resume path.
            self._verify_completed_checkpoint_identities(row)
            checkpoint = self.ledger.get_checkpoint(trial_id, int(tier))
            if trial_id in failed:
                if (
                    row.get("status") not in {"failed", "cancelled"}
                    or row.get("failure_reason") != failed[trial_id]
                    or (
                        checkpoint is not None
                        and checkpoint.get("status") == "completed"
                    )
                ):
                    raise RuntimeError(
                        "promotion round failed-evaluation state is inconsistent"
                    )
                continue
            if checkpoint is None or checkpoint.get("status") != "completed":
                raise RuntimeError(
                    "promotion round decision references an incomplete evaluation"
                )
            evaluation = self._promotion_evaluation(
                trial_id, tier, checkpoint["metrics"]
            )
            if tier <= Tier.T1_VQ2_REPLAY:
                eligibility = evaluation.eligibility
                if eligibility is None:
                    raise RuntimeError("promotion round eligibility evidence is missing")
                failures = () if eligibility.passed else eligibility.failures
            else:
                hard_gates = evaluation.hard_gates
                if hard_gates is None:
                    raise RuntimeError("promotion round hard-gate evidence is missing")
                failures = hard_gates.failures()
            if failures:
                expected_rejected[trial_id] = list(failures)
            else:
                eligible.append(evaluation)

        if rejected != expected_rejected:
            raise RuntimeError("promotion round hard-gate decision is stale")
        eligible.sort(
            key=lambda item: (item.quality.ordering_key(), item.candidate_id),
            reverse=True,
        )
        if promoted + eliminated != [item.candidate_id for item in eligible]:
            raise RuntimeError("promotion round quality ordering is stale")
        if eligible and not promoted:
            raise RuntimeError("promotion round eliminated every eligible candidate")
        if tier is Tier.T0_AFFECTED and eliminated:
            raise RuntimeError("T0 promotion round cannot halve eligible candidates")
        expected_promoted = (
            len(eligible)
            if tier is Tier.T0_AFFECTED
            else min(
                len(eligible),
                max(
                    minimum_survivors,
                    int(math.ceil(len(eligible) * keep_fraction)),
                ),
            )
            if eligible
            else 0
        )
        if len(promoted) != expected_promoted:
            raise RuntimeError("promotion round successive-halving cutoff is stale")
        expected_next_tier = (
            int(tier) + 1
            if eligible or expected_rejected or tier < Tier.T4_FULL_NON_LIVE
            else None
        )
        if (
            type(raw_decision["next_tier"]) not in {int, type(None)}
            or raw_decision["next_tier"] != expected_next_tier
        ):
            raise RuntimeError("promotion round next-tier decision is stale")
        return dict(raw_decision)

    def run_round(
        self,
        tier: Tier,
        *,
        keep_fraction: float = 0.5,
        minimum_survivors: int = 1,
    ) -> Optional[Mapping[str, Any]]:
        """Evaluate one tier for a cohort, halve, and leave survivors resumable."""

        if tier is Tier.T5_AUTHORIZED_LIVE:
            raise PermissionError("live trials are not scheduler promotion rounds")
        effective_keep_fraction = (
            1.0 if tier is Tier.T0_AFFECTED else keep_fraction
        )
        round_ladder = PromotionLadder(
            keep_fraction=effective_keep_fraction,
            minimum_survivors=minimum_survivors,
        )
        effective_keep_fraction = round_ladder.keep_fraction
        minimum_survivors = round_ladder.minimum_survivors
        if not self.ledger.acquire_global_lease(
            _ORCHESTRATION_LEASE, self.owner, ttl_s=self.lease_ttl_s
        ):
            raise RuntimeError("another scheduler owns the active lease")
        try:
            round_row = self.ledger.open_promotion_round(int(tier))
            command = self.commands.get(tier)
            if command is None:
                raise RuntimeError(f"no command configured for {tier.name}")

            def cohort_identity(row: Mapping[str, Any]) -> str:
                manifest_identity = self._tier_identity_hash(
                    row,
                    tier,
                    required=any(
                        step.trusted_files_sha256 for step in command.steps
                    ),
                )
                return json_hash(
                    {
                        "tier": int(tier),
                        "manifest_tier_identity_sha256": manifest_identity,
                        "command_plan_sha256": json_hash(
                            dataclasses.asdict(command)
                        ),
                        "environment_fingerprint": row.get(
                            "environment_fingerprint"
                        ),
                        "promotion_policy": {
                            "keep_fraction": effective_keep_fraction,
                            "minimum_survivors": minimum_survivors,
                        },
                    }
                )

            if round_row is None:
                cohorts: Dict[str, list[str]] = {}
                for row in self.ledger.list_trials():
                    if row["status"] not in {"pending", "running"}:
                        continue
                    completed = set(self.ledger.completed_tiers(row["trial_id"]))
                    if int(tier) not in completed and all(lower in completed for lower in range(int(tier))) and not any(
                        value > int(tier) for value in completed
                    ):
                        identity = cohort_identity(row)
                        cohorts.setdefault(identity, []).append(row["trial_id"])
                if not cohorts:
                    return None
                selected_identity = sorted(cohorts)[0]
                members = cohorts[selected_identity]
                round_row = self.ledger.create_or_get_promotion_round(
                    int(tier),
                    members,
                    cohort_identity_sha256=selected_identity,
                )
            current_identities = {
                cohort_identity(self.ledger.get_trial(trial_id))
                for trial_id in round_row["member_trial_ids"]
            }
            if len(current_identities) != 1:
                raise RuntimeError("promotion round mixes evaluator cohort identities")
            current_identity = next(iter(current_identities))
            expected_round_id = json_hash(
                {
                    "tier": int(tier),
                    "members": tuple(sorted(round_row["member_trial_ids"])),
                    "cohort_identity_sha256": current_identity,
                }
            )
            if round_row["round_id"] != expected_round_id:
                raise RuntimeError("promotion round cohort identity is stale")

            if round_row["status"] == "planned":
                evaluations: list[CandidateEvaluation] = []
                failed: Dict[str, str] = {}
                for trial_id in round_row["member_trial_ids"]:
                    checkpoint = self.ledger.get_checkpoint(trial_id, int(tier))
                    if checkpoint is not None and checkpoint["status"] == "failed":
                        row = self.ledger.get_trial(trial_id)
                        reason = f"{tier.name} has a durable failed checkpoint"
                        if row["status"] in {"pending", "running"}:
                            if not self.ledger.lease_trial(
                                trial_id, self.owner, ttl_s=self.lease_ttl_s
                            ):
                                raise RuntimeError(
                                    f"failed promotion member {trial_id} has an active lease"
                                )
                            self.ledger.finish_trial(
                                trial_id,
                                self.owner,
                                success=False,
                                failure_reason=reason,
                                phase_timings=self._checkpoint_phase_timings(trial_id),
                                artifact_hashes=self._checkpoint_artifacts(trial_id),
                                stdout_stderr_tail=checkpoint.get("stdout_stderr_tail"),
                            )
                        else:
                            reason = row.get("failure_reason") or reason
                        failed[trial_id] = reason
                        continue
                    if checkpoint is not None and checkpoint["status"] == "completed":
                        row = self.ledger.get_trial(trial_id)
                        try:
                            self._verify_completed_checkpoint_identities(row)
                        except (RuntimeError, TypeError, ValueError) as exc:
                            reason = (
                                f"completed promotion member {trial_id} has invalid "
                                f"evidence: {exc}"
                            )
                            if row["status"] in {"pending", "running"}:
                                if not self.ledger.lease_trial(
                                    trial_id, self.owner, ttl_s=self.lease_ttl_s
                                ):
                                    raise RuntimeError(
                                        f"invalid promotion member {trial_id} has an active lease"
                                    )
                                self.ledger.finish_trial(
                                    trial_id,
                                    self.owner,
                                    success=False,
                                    failure_reason=reason,
                                    phase_timings=self._checkpoint_phase_timings(trial_id),
                                    artifact_hashes=self._checkpoint_artifacts(trial_id),
                                    stdout_stderr_tail=checkpoint.get(
                                        "stdout_stderr_tail"
                                    ),
                                )
                            failed[trial_id] = reason
                            continue
                        if row["status"] in {"pending", "running"}:
                            if not self.ledger.lease_trial(
                                trial_id, self.owner, ttl_s=self.lease_ttl_s
                            ):
                                raise RuntimeError(
                                    f"completed promotion member {trial_id} has an active lease"
                                )
                            if tier < Tier.T4_FULL_NON_LIVE:
                                self.ledger.yield_trial(trial_id, self.owner)
                        evaluations.append(
                            self._promotion_evaluation(
                                trial_id, tier, checkpoint["metrics"]
                            )
                        )
                        continue
                    row = self.ledger.get_trial(trial_id)
                    if row["status"] in {"failed", "cancelled"}:
                        failed[trial_id] = row.get("failure_reason") or "terminal failure"
                        continue
                    if not self.ledger.lease_trial(
                        trial_id, self.owner, ttl_s=self.lease_ttl_s
                    ):
                        raise RuntimeError(
                            f"promotion member {trial_id} still has an active lease"
                        )
                    try:
                        worktree, config_path = self._prepare_trial(row)
                    except Exception as exc:
                        reason = f"trial setup failed: {type(exc).__name__}: {exc}"
                        self.ledger.finish_trial(
                            trial_id,
                            self.owner,
                            success=False,
                            failure_reason=reason,
                        )
                        failed[trial_id] = reason
                        continue
                    self.ledger.checkpoint(
                        trial_id,
                        int(tier),
                        owner=self.owner,
                        status="running",
                    )
                    outcome = self._run_tier_command(
                        trial_id,
                        row,
                        command,
                        worktree,
                        config_path,
                        row["config_hash"],
                    )
                    binding_failure = self._tier_evidence_binding_failure(
                        row, tier, outcome.metrics
                    )
                    if (
                        binding_failure is not None
                        and outcome.failure_reason is None
                    ):
                        outcome = dataclasses.replace(
                            outcome, failure_reason=binding_failure
                        )
                    passed = (
                        outcome.returncode == 0
                        and not outcome.timed_out
                        and outcome.failure_reason is None
                    )
                    self.ledger.checkpoint(
                        trial_id,
                        int(tier),
                        owner=self.owner,
                        status="completed" if passed else "failed",
                        metrics=outcome.metrics,
                        artifact_hashes=self._artifacts_for_tier(
                            row,
                            tier,
                            outcome.metrics,
                            trusted_required=any(
                                step.trusted_files_sha256 for step in command.steps
                            ),
                        ),
                        stdout_stderr_tail=bounded_tail(
                            (outcome.stdout, outcome.stderr)
                        ),
                        elapsed_s=outcome.elapsed_s,
                    )
                    if not passed:
                        reason = outcome.failure_reason or "tier command failed"
                        self.ledger.finish_trial(
                            trial_id,
                            self.owner,
                            success=False,
                            failure_reason=reason,
                            phase_timings=self._checkpoint_phase_timings(trial_id),
                            artifact_hashes=self._checkpoint_artifacts(trial_id),
                            stdout_stderr_tail=bounded_tail(
                                (outcome.stdout, outcome.stderr)
                            ),
                            worktree_path=str(worktree),
                        )
                        failed[trial_id] = reason
                        continue
                    self.ledger.yield_trial(trial_id, self.owner)
                    evaluations.append(
                        self._promotion_evaluation(trial_id, tier, outcome.metrics)
                    )

                if evaluations:
                    ladder_decision = round_ladder.decide(evaluations)
                    decision: Dict[str, Any] = dataclasses.asdict(ladder_decision)
                    decision["tier"] = int(tier)
                    decision["next_tier"] = (
                        int(ladder_decision.next_tier)
                        if ladder_decision.next_tier is not None
                        else None
                    )
                else:
                    decision = {
                        "tier": int(tier),
                        "promoted": [],
                        "rejected_hard_gate": {},
                        "eliminated_by_halving": [],
                        "next_tier": int(tier) + 1 if tier < Tier.T4_FULL_NON_LIVE else None,
                    }
                decision["keep_fraction"] = effective_keep_fraction
                decision["minimum_survivors"] = minimum_survivors
                decision["failed_evaluation"] = failed
                self.ledger.decide_promotion_round(round_row["round_id"], decision)
                round_row = self.ledger.get_promotion_round(round_row["round_id"])

            decision = self._validate_decided_promotion_round(round_row, tier)
            promoted = set(decision.get("promoted", []))
            rejected = set(decision.get("rejected_hard_gate", {}))
            eliminated = set(decision.get("eliminated_by_halving", [])) | rejected
            for trial_id in promoted:
                row = self.ledger.get_trial(trial_id)
                if tier is Tier.T4_FULL_NON_LIVE and row["status"] in {"pending", "running"}:
                    if not self.ledger.lease_trial(
                        trial_id, self.owner, ttl_s=self.lease_ttl_s
                    ):
                        raise RuntimeError("could not finalize T4 survivor")
                    self.ledger.finish_trial(
                        trial_id,
                        self.owner,
                        success=True,
                        safety_and_completion_metrics={
                            "promotion_round": decision,
                            "completed_tiers": list(
                                self.ledger.completed_tiers(trial_id)
                            ),
                        },
                        phase_timings=self._checkpoint_phase_timings(trial_id),
                        artifact_hashes=self._checkpoint_artifacts(trial_id),
                    )
            for trial_id in eliminated:
                row = self.ledger.get_trial(trial_id)
                if row["status"] in {"pending", "running"}:
                    if not self.ledger.lease_trial(
                        trial_id, self.owner, ttl_s=self.lease_ttl_s
                    ):
                        raise RuntimeError("could not finalize eliminated candidate")
                    self.ledger.finish_trial(
                        trial_id,
                        self.owner,
                        success=False,
                        failure_reason=f"successive-halving elimination after {tier.name}",
                        safety_and_completion_metrics={"promotion_round": decision},
                        phase_timings=self._checkpoint_phase_timings(trial_id),
                        artifact_hashes=self._checkpoint_artifacts(trial_id),
                    )
            self.ledger.mark_promotion_round_applied(round_row["round_id"])
            return decision
        finally:
            self.ledger.release_global_lease(_ORCHESTRATION_LEASE, self.owner)


class SingleMerger:
    """Serialize an explicitly requested fast-forward merge."""

    def __init__(self, ledger: TrialLedger, repository: Path | str, *, owner: Optional[str] = None) -> None:
        self.ledger = ledger
        self.repository = Path(repository).resolve()
        self.owner = owner or f"merger-{uuid.uuid4().hex}"

    def merge_completed(self, trial_id: str) -> str:
        trial = self.ledger.get_trial(trial_id)
        validate_promotion_chain(self.ledger, trial_id)
        if not self.ledger.acquire_global_lease(
            _ORCHESTRATION_LEASE, self.owner, ttl_s=180.0
        ):
            raise RuntimeError("another merger owns the active lease")
        stop = threading.Event()
        heartbeat_failures: list[str] = []

        def renew() -> None:
            while not stop.wait(10.0):
                try:
                    if not self.ledger.acquire_global_lease(
                        _ORCHESTRATION_LEASE, self.owner, ttl_s=180.0
                    ):
                        raise RuntimeError("lost singleton merger lease")
                except Exception as exc:
                    heartbeat_failures.append(f"{type(exc).__name__}: {exc}")
                    return

        heartbeat = threading.Thread(
            target=renew, name="aigp-merger-heartbeat", daemon=True
        )
        heartbeat.start()
        try:
            if not self.ledger.acquire_global_lease(
                _ORCHESTRATION_LEASE, self.owner, ttl_s=180.0
            ):
                raise RuntimeError("lost singleton merger lease before merge")
            dirty = run_checked(["git", "status", "--porcelain"], cwd=self.repository)
            if dirty:
                raise RuntimeError("merge checkout is not clean")
            subprocess.run(
                ["git", "merge", "--ff-only", trial["commit_hash"]],
                cwd=str(self.repository),
                check=True,
                capture_output=True,
                text=True,
                timeout=60.0,
                shell=False,
            )
            if heartbeat_failures:
                raise RuntimeError(
                    f"merger lease heartbeat failed: {heartbeat_failures[0]}"
                )
            return run_checked(["git", "rev-parse", "HEAD"], cwd=self.repository)
        finally:
            stop.set()
            heartbeat.join(timeout=2.0)
            if heartbeat.is_alive():
                heartbeat_failures.append("merger heartbeat thread did not stop")
            self.ledger.release_global_lease(_ORCHESTRATION_LEASE, self.owner)

    @staticmethod
    def _safety_evidence_passed(evidence: Mapping[str, Any]) -> bool:
        required = (
            "valid",
            "completed",
            "correct_gate_sequence",
            "cleanup_confirmed",
            "no_collision",
            "no_disqualification",
            "no_stale_stream_flight",
        )
        return all(name in evidence and evidence[name] is True for name in required)
