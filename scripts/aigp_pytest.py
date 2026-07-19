"""Hash-pinned, isolated bootstrap for the exact T1 VQ2 pytest gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


_DISCOVERY_NAMES = frozenset(
    {"conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"}
)
_STARTUP_NAMES = frozenset({"sitecustomize.py", "usercustomize.py"})
_IMPORT_SHADOW_NAMES = frozenset({"pytest.py", "pytest_timeout.py"})
_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        ".aigp-loop",
        ".cache",
        ".loop",
        ".research_loop",
        ".pytest_cache",
    }
)
_RUNTIME_PYCACHE_CONTEXTS: list[tempfile.TemporaryDirectory[str]] = []


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
        object_pairs_hook=_unique_object,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_control_files(root: Path) -> set[str]:
    found: set[str] = set()
    for directory, names, files in os.walk(root):
        # Prune explicitly external/runtime state before applying the source
        # bytecode boundary.  In particular, a repository-local virtualenv is
        # required to launch this bootstrap but is never candidate source.
        names[:] = [
            name
            for name in names
            if name.casefold()
            not in {ignored.casefold() for ignored in _IGNORED_DIRECTORY_NAMES}
        ]
        if any(name.casefold() == "__pycache__" for name in names) or any(
            name.casefold().endswith((".pyc", ".pyo")) for name in files
        ):
            raise ValueError("candidate contains executable bytecode/cache")
        if any(name.casefold() in {"pytest", "pytest_timeout"} for name in names):
            found.add(
                (Path(directory) / next(
                    name
                    for name in names
                    if name.casefold() in {"pytest", "pytest_timeout"}
                )).relative_to(root).as_posix()
            )
        base = Path(directory)
        for name in files:
            folded = name.casefold()
            if (
                folded in _DISCOVERY_NAMES
                or folded in _STARTUP_NAMES
                or folded in _IMPORT_SHADOW_NAMES
                or folded.endswith(".pth")
            ):
                found.add((base / name).relative_to(root).as_posix())
    return found


def audit_candidate(
    root: Path, manifest: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[Path, ...]:
    """Reject unreviewed pytest discovery/startup inputs and unpinned tests."""

    root = Path(root).resolve(strict=True)
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "files"}
        or manifest.get("schema") != "aigp-trusted-evaluator-files/1"
        or type(manifest.get("files")) is not dict
    ):
        raise ValueError("trusted manifest has an invalid exact schema")
    if (
        type(policy) is not dict
        or set(policy)
        != {
            "schema",
            "expected_passed",
            "pytest_version",
            "pytest_timeout_version",
            "test_files",
            "trusted_discovery_files",
        }
        or policy.get("schema") != "aigp-t1-pytest-policy/1"
        or type(policy.get("expected_passed")) is not int
        or policy["expected_passed"] < 1
        or type(policy.get("test_files")) is not list
        or not policy["test_files"]
        or type(policy.get("trusted_discovery_files")) is not list
    ):
        raise ValueError("T1 pytest policy has an invalid exact schema")
    files = manifest["files"]
    expected_control = set(policy["trusted_discovery_files"])
    observed_control = _walk_control_files(root)
    if observed_control != expected_control:
        raise ValueError(
            "unreviewed pytest/startup discovery inputs: "
            f"expected {sorted(expected_control)}, observed {sorted(observed_control)}"
        )
    test_names = policy["test_files"]
    if (
        len(test_names) != len(set(test_names))
        or any(type(name) is not str or not name for name in test_names)
    ):
        raise ValueError("T1 test inventory must contain unique paths")
    required = {
        *expected_control,
        *test_names,
        "config/t1_pytest.ini",
        "config/t1_pytest_policy.json",
        "scripts/aigp_pytest.py",
    }
    if not required <= set(files):
        raise ValueError("trusted manifest omits the T1 bootstrap closure")
    resolved_tests: list[Path] = []
    for relative in sorted(required):
        path = root / relative
        expected_hash = files.get(relative)
        if (
            type(expected_hash) is not str
            or len(expected_hash) != 64
            or path.is_symlink()
            or not path.is_file()
            or _sha256(path) != expected_hash
        ):
            raise ValueError(f"trusted T1 file mismatch: {relative}")
        if relative in test_names:
            resolved_tests.append(path)
    return tuple(resolved_tests)


class _ExactOutcomePlugin:
    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.collected = 0
        self.call_passed = 0
        self.nonpassing: list[str] = []

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when == "call" and report.passed and not hasattr(report, "wasxfail"):
            self.call_passed += 1
        elif report.failed or report.skipped or hasattr(report, "wasxfail"):
            self.nonpassing.append(f"{report.nodeid}:{report.when}:{report.outcome}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("vq2", "affected"))
    parser.add_argument("tests", nargs="*")
    args = parser.parse_args(argv)
    root_text = os.environ.get("AIGP_CANDIDATE_WORKTREE")
    if not root_text:
        parser.error("AIGP_CANDIDATE_WORKTREE is required")
    root = Path(root_text).resolve(strict=True)
    if Path.cwd().resolve() != root:
        parser.error("T1 bootstrap must run from the exact candidate worktree")
    runtime_root = Path(sys.prefix).resolve()
    if any(
        _is_within(Path(entry), root) and not _is_within(Path(entry), runtime_root)
        for entry in sys.path
        if entry and Path(entry).exists()
    ):
        parser.error("candidate root entered sys.path before trusted bootstrap")

    # ``-I`` ignores PYTHONDONTWRITEBYTECODE from the launching environment.
    # Install a fresh external prefix before even reading/auditing candidate
    # inputs so this bootstrap itself cannot create source-adjacent caches.
    pycache_context = tempfile.TemporaryDirectory(prefix="aigp-t1-pycache-")
    _RUNTIME_PYCACHE_CONTEXTS.append(pycache_context)
    pycache_prefix = Path(pycache_context.name).resolve(strict=True)
    if _is_within(pycache_prefix, root):
        parser.error("trusted bytecode prefix must be outside candidate worktree")
    sys.pycache_prefix = str(pycache_prefix)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_prefix)

    manifest = _load_json(root / "config" / "promotion_trusted_files.json")
    policy = _load_json(root / "config" / "t1_pytest_policy.json")
    policy_tests = audit_candidate(root, manifest, policy)
    if args.mode == "vq2":
        if args.tests:
            parser.error("vq2 mode uses the exact frozen test inventory")
        tests = policy_tests
        expected_passed = policy["expected_passed"]
    else:
        if not args.tests:
            parser.error("affected mode requires at least one pinned test file")
        files = manifest["files"]
        selected = []
        for raw in args.tests:
            relative = Path(raw)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix != ".py"
            ):
                parser.error("affected tests must be safe repository-relative .py files")
            normalized = relative.as_posix()
            target = root / relative
            if (
                normalized not in files
                or target.is_symlink()
                or not target.is_file()
                or _sha256(target) != files[normalized]
            ):
                parser.error(f"affected test is not hash-pinned: {normalized}")
            selected.append(target)
        tests = tuple(selected)
        expected_passed = None
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for name in tuple(os.environ):
        if name.upper().startswith("PYTEST_") and name.upper() != "PYTEST_DISABLE_PLUGIN_AUTOLOAD":
            os.environ.pop(name, None)

    import pytest
    import pytest_timeout

    if (
        pytest.__version__ != policy["pytest_version"]
        or importlib.metadata.version("pytest-timeout")
        != policy["pytest_timeout_version"]
        or (
            _is_within(Path(pytest.__file__), root)
            and not _is_within(Path(pytest.__file__), runtime_root)
        )
        or (
            _is_within(Path(pytest_timeout.__file__), root)
            and not _is_within(Path(pytest_timeout.__file__), runtime_root)
        )
    ):
        parser.error("trusted pytest runtime identity mismatch")
    sys.path.insert(0, str(root))
    outcome = _ExactOutcomePlugin(expected_passed or 0)
    marker_expression = (
        "not slow and not benchmark and not live"
        if args.mode == "vq2"
        else "not live"
    )
    pytest_args = [
        "-q",
        "-p",
        "no:cacheprovider",
        "--noconftest",
        "--import-mode=importlib",
        "-m",
        marker_expression,
        "--rootdir",
        str(root),
        "-c",
        str(root / "config" / "t1_pytest.ini"),
        *(str(path) for path in tests),
    ]
    result = pytest.main(pytest_args, plugins=[pytest_timeout, outcome])
    if (
        int(result) != 0
        or outcome.nonpassing
        or outcome.collected < 1
        or (
            expected_passed is not None
            and outcome.collected != expected_passed
        )
        or outcome.call_passed != outcome.collected
    ):
        print(
            "T1 exact outcome failed: "
            f"exit={int(result)} collected={outcome.collected} "
            f"passed={outcome.call_passed} nonpassing={outcome.nonpassing[:10]}",
            file=sys.stderr,
        )
        return 2
    return 0


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
