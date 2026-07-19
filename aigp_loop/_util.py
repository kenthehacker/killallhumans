"""Small deterministic serialization and provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


def canonical_json(value: Any) -> str:
    """Return stable, strict JSON suitable for hashing and SQLite storage."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    """Decode RFC-compliant JSON, rejecting Python's NaN/Infinity extension."""

    return json.loads(
        text,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )


def strict_json_load(path: Path | str) -> Any:
    """Decode one stable, non-indirected UTF-8 JSON file snapshot."""

    return strict_json_loads(read_secure_regular_file(path).decode("utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def secure_relative_regular_file(root: Path | str, relative: Path | str) -> Path:
    """Resolve a trusted file without traversing symlinks or reparse points.

    ``Path.resolve()`` alone is unsafe for a trust manifest because it follows
    the very indirection the manifest is supposed to exclude.  Inspect every
    lexical component first (including Windows junctions/reparse points), then
    require the final object to be a regular file contained by ``root``.
    """

    trusted_root = Path(root).resolve(strict=True)
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or relative_path.drive
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"trusted file path is not canonical and relative: {relative!s}")
    probe = trusted_root
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for component in relative_path.parts:
        probe = probe / component
        try:
            info = probe.lstat()
        except OSError as exc:
            raise ValueError(f"trusted file is missing: {relative!s}") from exc
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & reparse_flag
        ):
            raise ValueError(
                f"trusted file path traverses a symlink/reparse point: {relative!s}"
            )
    if not stat.S_ISREG(probe.lstat().st_mode):
        raise ValueError(f"trusted file is not a regular file: {relative!s}")
    resolved = probe.resolve(strict=True)
    try:
        resolved.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(f"trusted file escapes repository: {relative!s}") from exc
    return resolved


def secure_regular_file(path: Path | str) -> Path:
    """Return an existing regular file without accepting path indirection."""

    lexical = Path(path)
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    # ``absolute`` normalization is lexical; unlike ``resolve`` it does not
    # erase a symlink/reparse component before we inspect it.
    lexical = Path(os.path.abspath(lexical))
    probe = Path(lexical.anchor)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for component in lexical.parts[1:]:
        probe = probe / component
        try:
            info = probe.lstat()
        except OSError as exc:
            raise ValueError(f"file is missing: {lexical}") from exc
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & reparse_flag
        ):
            raise ValueError(f"file path traverses a symlink/reparse point: {lexical}")
    if not stat.S_ISREG(probe.lstat().st_mode):
        raise ValueError(f"path is not a regular file: {lexical}")
    return probe.resolve(strict=True)


def secure_directory(path: Path | str) -> Path:
    """Return an existing directory without accepting symlink/reparse paths."""

    lexical = Path(path)
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    lexical = Path(os.path.abspath(lexical))
    probe = Path(lexical.anchor)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for component in lexical.parts[1:]:
        probe = probe / component
        try:
            info = probe.lstat()
        except OSError as exc:
            raise ValueError(f"directory is missing: {lexical}") from exc
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & reparse_flag
        ):
            raise ValueError(
                f"directory path traverses a symlink/reparse point: {lexical}"
            )
    if not stat.S_ISDIR(probe.lstat().st_mode):
        raise ValueError(f"path is not a directory: {lexical}")
    return probe.resolve(strict=True)


def read_secure_regular_file(
    path: Path | str, *, maximum_bytes: Optional[int] = None
) -> bytes:
    """Read one stable bounded regular-file snapshot.

    The ceiling is enforced on the opened descriptor and during both reads,
    rather than by a pathname pre-check that a replacement race could bypass.
    """

    if maximum_bytes is not None and (
        type(maximum_bytes) is not int or maximum_bytes < 0
    ):
        raise ValueError("maximum_bytes must be a non-negative exact integer")

    target = secure_regular_file(path)
    flags = os.O_RDONLY
    for optional_flag in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, optional_flag, 0))
    fd = os.open(target, flags)
    try:
        opened = os.fstat(fd)
        named = os.stat(target, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, named):
            raise ValueError(f"file changed while opening snapshot: {target}")
        if maximum_bytes is not None and opened.st_size > maximum_bytes:
            raise ValueError(f"file exceeds resource limit: {target}")
        stability_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )

        def signature(info: os.stat_result) -> tuple[Any, ...]:
            return tuple(getattr(info, field, None) for field in stability_fields)

        def read_all() -> bytes:
            chunks: list[bytes] = []
            length = 0
            while True:
                read_size = 1024 * 1024
                if maximum_bytes is not None:
                    # Read at most one byte beyond the ceiling so growth after
                    # fstat is rejected without allocating the replacement.
                    read_size = min(read_size, maximum_bytes + 1 - length)
                chunk = os.read(fd, read_size)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
                length += len(chunk)
                if maximum_bytes is not None and length > maximum_bytes:
                    raise ValueError(f"file exceeds resource limit: {target}")

        first = read_all()
        after_first = os.fstat(fd)
        if signature(after_first) != signature(opened):
            raise ValueError(f"file mutated while reading snapshot: {target}")
        os.lseek(fd, 0, os.SEEK_SET)
        second = read_all()
        after_second = os.fstat(fd)
        if signature(after_second) != signature(after_first) or second != first:
            raise ValueError(f"file mutated while confirming snapshot: {target}")
        named_after = os.stat(target, follow_symlinks=False)
        if (
            not os.path.samestat(opened, named_after)
            or named_after.st_size != after_second.st_size
        ):
            raise ValueError(f"file changed while reading snapshot: {target}")
        return first
    finally:
        os.close(fd)


def json_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def environment_fingerprint() -> str:
    """Hash the interpreter/platform and installed distribution versions."""

    distributions = sorted(
        (dist.metadata.get("Name", "").lower(), dist.version)
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    payload = {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": distributions,
    }
    return json_hash(payload)


def run_checked(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float = 20.0,
) -> str:
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        shell=False,
    )
    return completed.stdout.strip()


def _untracked_files_digest(root: Path, relative_names: Sequence[str]) -> bytes:
    """Hash untracked regular files with an unambiguous, versioned encoding."""

    digest = hashlib.sha256()
    digest.update(b"aigp-untracked-files/3\0")
    names = sorted(relative_names)
    if len(names) != len(set(names)):
        raise ValueError("git returned duplicate untracked file names")
    digest.update(len(names).to_bytes(8, "big"))
    for relative in names:
        if type(relative) is not str or not relative or "\0" in relative:
            raise ValueError("untracked file name must be a non-empty exact string")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"untracked path escapes repository: {relative!r}")
        try:
            path = secure_relative_regular_file(root, relative_path)
            content = read_secure_regular_file(path)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot safely read untracked file {relative!r}: {exc}"
            ) from exc
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.digest()


def git_provenance(repo: Path) -> tuple[str, str, str]:
    """Return ``(commit, dirty_diff_hash, combined_code_hash)``.

    Staged, unstaged, and untracked file names/content are included.  The
    combined hash prevents two dirty candidates at the same commit from
    deduplicating accidentally.
    """

    root = Path(repo).resolve()

    def capture() -> tuple[str, bytes, bytes]:
        commit = run_checked(["git", "rev-parse", "HEAD"], cwd=root)
        if (
            len(commit) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError("Git HEAD must be an exact hexadecimal object identity")
        diff = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
            ],
            cwd=str(root),
            check=True,
            capture_output=True,
            timeout=30.0,
            shell=False,
        ).stdout
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=str(root),
            check=True,
            capture_output=True,
            timeout=30.0,
            shell=False,
        )
        try:
            untracked_names = [
                raw.decode("utf-8")
                for raw in untracked_result.stdout.split(b"\0")
                if raw
            ]
        except UnicodeDecodeError as exc:
            raise ValueError("untracked Git file names must be UTF-8") from exc
        return commit, diff, _untracked_files_digest(root, untracked_names)

    previous = capture()
    for _ in range(3):
        current = capture()
        if current == previous:
            break
        previous = current
    else:
        raise RuntimeError("worktree changed while capturing Git provenance")
    commit, diff, untracked_digest = current
    dirty_hash = sha256_bytes(diff + untracked_digest)
    return commit, dirty_hash, sha256_text(f"{commit}\0{dirty_hash}")


def bounded_tail(parts: Iterable[str], *, max_chars: int = 32_000) -> str:
    text = "\n".join(part for part in parts if part)
    return text[-max_chars:]


def private_path_guard(path: Path, repo: Optional[Path] = None) -> None:
    """Fail if ``path`` is tracked, or would be tracked, by the repository.

    Recording competition material can require organizer approval.  Bundle
    creation therefore refuses a path Git would include.  Paths outside the
    repository are accepted; paths inside must match an ignore rule.
    """

    target = Path(path).resolve()
    root = Path(repo).resolve() if repo is not None else None
    if root is None:
        probe = target
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if probe.is_file():
            probe = probe.parent
        try:
            root = Path(
                run_checked(["git", "rev-parse", "--show-toplevel"], cwd=probe)
            ).resolve()
        except (subprocess.SubprocessError, OSError):
            return
    try:
        relative = target.relative_to(root)
    except ValueError:
        return
    probe = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative.as_posix()],
        cwd=str(root),
        capture_output=True,
        shell=False,
    )
    if probe.returncode != 0:
        raise ValueError(
            f"private replay path is not Git-ignored: {target}; choose an "
            "external directory or captures/replays/"
        )
