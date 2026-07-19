"""Generate an offline CycloneDX inventory for the active Python environment.

This deliberately uses only the standard library so inventory generation does
not itself add a runtime dependency. Package license fields are copied from
installed metadata and must be reviewed; they are not a legal determination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

SCHEMA_VERSION = 3
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9_.-]+")


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _purl(name: str, version: str) -> str:
    return f"pkg:pypi/{quote(_canonical_name(name))}@{quote(version)}"


def _license_entries(package_metadata: metadata.PackageMetadata) -> list[dict]:
    expression = package_metadata.get("License-Expression")
    if expression and expression.strip().upper() != "UNKNOWN":
        return [{"expression": expression.strip()}]

    declared = package_metadata.get("License")
    if declared:
        declared = " ".join(declared.split())
        if declared and declared.upper() != "UNKNOWN" and len(declared) <= 256:
            return [{"license": {"name": declared}}]

    classifiers = package_metadata.get_all("Classifier") or []
    names = sorted(
        {
            item.rsplit(" :: ", 1)[-1]
            for item in classifiers
            if item.startswith("License ::")
            and not item.endswith("OSI Approved")
        }
    )
    return [{"license": {"name": name}} for name in names]


def _git_bytes(repo_root: Path, *args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _git_value(repo_root: Path, *args: str) -> str | None:
    payload = _git_bytes(repo_root, *args)
    if payload is None:
        return None
    return payload.decode("utf-8", errors="replace").strip()


def _untracked_tree_identity(repo_root: Path) -> tuple[str, str]:
    """Hash every non-ignored untracked file without disclosing its contents.

    ``git diff HEAD`` intentionally omits untracked files.  A mere dirty flag
    therefore cannot distinguish two inventories generated from different new
    source files.  Git supplies a NUL-delimited path list so unusual filenames
    remain unambiguous; regular-file bytes and symlink targets are then bound
    to those repository-relative identities.  Any unreadable/unsafe entry
    fails closed to ``unknown`` instead of publishing a misleading digest.
    """

    names = _git_bytes(
        repo_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    if names is None:
        return "unknown", "unknown"

    raw_paths = sorted(path for path in names.split(b"\0") if path)
    digest = hashlib.sha256(b"aigp-untracked-tree-v2\0")
    try:
        for raw_path in raw_paths:
            relative_text = os.fsdecode(raw_path)
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                return "unknown", "unknown"
            path = repo_root / relative
            file_stat = path.lstat()
            digest.update(raw_path)
            digest.update(b"\0")
            if stat.S_ISREG(file_stat.st_mode):
                digest.update(b"regular\0")
                flags = os.O_RDONLY
                for optional_flag in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
                    flags |= int(getattr(os, optional_flag, 0))
                fd = os.open(path, flags)
                try:
                    opened = os.fstat(fd)
                    if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(
                        opened, file_stat
                    ):
                        return "unknown", "unknown"

                    def hash_open_file() -> bytes:
                        os.lseek(fd, 0, os.SEEK_SET)
                        content_digest = hashlib.sha256()
                        while True:
                            chunk = os.read(fd, 1024 * 1024)
                            if not chunk:
                                return content_digest.digest()
                            content_digest.update(chunk)

                    before = (
                        opened.st_size,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                    )
                    first_digest = hash_open_file()
                    middle = os.fstat(fd)
                    second_digest = hash_open_file()
                    after = os.fstat(fd)
                    named_after = path.lstat()
                    after_signature = (
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    )
                    if (
                        first_digest != second_digest
                        or before != after_signature
                        or not os.path.samestat(opened, middle)
                        or not os.path.samestat(opened, after)
                        or not os.path.samestat(opened, named_after)
                    ):
                        return "unknown", "unknown"
                    digest.update(first_digest)
                finally:
                    os.close(fd)
            elif stat.S_ISLNK(file_stat.st_mode):
                digest.update(b"symlink\0")
                first_target = os.readlink(path)
                named_after = path.lstat()
                second_target = os.readlink(path)
                if (
                    not os.path.samestat(file_stat, named_after)
                    or first_target != second_target
                ):
                    return "unknown", "unknown"
                digest.update(os.fsencode(first_target))
            else:
                return "unknown", "unknown"
            digest.update(b"\0")
    except (OSError, ValueError):
        return "unknown", "unknown"
    return digest.hexdigest(), str(len(raw_paths))


def _repository_properties(repo_root: Path) -> list[dict[str, str]]:
    def capture() -> tuple[bytes | None, bytes | None, bytes | None, str, str]:
        return (
            _git_bytes(repo_root, "rev-parse", "HEAD"),
            _git_bytes(
                repo_root, "status", "--porcelain=v1", "--untracked-files=all"
            ),
            _git_bytes(
                repo_root,
                "diff",
                "--binary",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
            ),
            *_untracked_tree_identity(repo_root),
        )

    previous = capture()
    stable = False
    current = previous
    for _ in range(3):
        current = capture()
        if current == previous:
            stable = True
            break
        previous = current

    commit_bytes, status_bytes, diff_bytes, untracked_digest, untracked_count = current
    available = (
        commit_bytes is not None
        and status_bytes is not None
        and diff_bytes is not None
        and untracked_digest != "unknown"
        and untracked_count != "unknown"
    )
    snapshot_stable = stable and available
    if snapshot_stable:
        commit = commit_bytes.decode("ascii", errors="strict").strip()
        if len(commit) not in {40, 64} or any(
            char not in "0123456789abcdefABCDEF" for char in commit
        ):
            snapshot_stable = False
    if not snapshot_stable:
        commit = "unknown"
        dirty = "unknown"
        diff_digest = "unknown"
        untracked_digest = "unknown"
        untracked_count = "unknown"
    else:
        dirty = str(bool(status_bytes.strip())).lower()
        diff_digest = hashlib.sha256(diff_bytes).hexdigest()
    return [
        {"name": "aigp:repository_commit", "value": commit},
        {"name": "aigp:repository_dirty", "value": dirty},
        {"name": "aigp:tracked_diff_sha256", "value": diff_digest},
        {"name": "aigp:untracked_tree_sha256", "value": untracked_digest},
        {"name": "aigp:untracked_file_count", "value": untracked_count},
        {
            "name": "aigp:repository_snapshot_stable",
            "value": str(snapshot_stable).lower(),
        },
        {"name": "aigp:inventory_generator_schema", "value": str(SCHEMA_VERSION)},
    ]


def build_inventory(
    distributions: Iterable[metadata.Distribution] | None = None,
    *,
    repo_root: Path | None = None,
) -> dict:
    """Return a CycloneDX JSON object for installed Python distributions."""
    repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    installed = list(distributions if distributions is not None else metadata.distributions())

    records: dict[str, tuple[str, str, metadata.PackageMetadata, list[str]]] = {}
    for dist in installed:
        name = dist.metadata.get("Name")
        version = dist.version
        if not name or not version:
            continue
        requires = dist.requires or []
        records[_canonical_name(name)] = (name, version, dist.metadata, requires)

    components: list[dict] = []
    dependencies: list[dict] = []
    for canonical in sorted(records):
        name, version, package_metadata, requires = records[canonical]
        component_ref = _purl(name, version)
        component: dict = {
            "type": "library",
            "bom-ref": component_ref,
            "name": name,
            "version": version,
            "purl": component_ref,
        }
        licenses = _license_entries(package_metadata)
        if licenses:
            component["licenses"] = licenses
        homepage = package_metadata.get("Home-page")
        if homepage and homepage.startswith(("https://", "http://")):
            component["externalReferences"] = [
                {"type": "website", "url": homepage}
            ]
        components.append(component)

        dependency_refs: set[str] = set()
        for requirement in requires:
            match = _REQUIREMENT_NAME.match(requirement)
            if not match:
                continue
            dependency = records.get(_canonical_name(match.group(0)))
            if dependency is not None:
                dependency_refs.add(_purl(dependency[0], dependency[1]))
        dependencies.append(
            {"ref": component_ref, "dependsOn": sorted(dependency_refs)}
        )

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    properties = _repository_properties(repo_root)
    properties.extend(
        [
            {"name": "aigp:python_version", "value": platform.python_version()},
            {"name": "aigp:python_implementation", "value": platform.python_implementation()},
            {"name": "aigp:platform", "value": platform.platform()},
        ]
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "killallhumans dependency inventory",
                        "version": str(SCHEMA_VERSION),
                    }
                ]
            },
            "properties": properties,
        },
        "components": components,
        "dependencies": dependencies,
    }


def write_inventory(inventory: dict, output_path: Path) -> None:
    """Atomically write an inventory as UTF-8 JSON."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary_path = Path(handle.name)
    temporary_path.replace(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".artifacts/dependency-inventory.cdx.json"),
        help="output JSON path (default: .artifacts/dependency-inventory.cdx.json)",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    inventory = build_inventory(repo_root=repo_root)
    write_inventory(inventory, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
