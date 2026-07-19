"""Verify and score private VQ2 replay bundles without a simulator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

_REPO = Path(__file__).resolve().parent.parent
_REPLAY_HOST_FILES = frozenset(
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
_MAX_TRUSTED_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_TRUSTED_SOURCE_BYTES = 64 * 1024 * 1024
_TRUSTED_MANIFEST_PATH = "config/promotion_trusted_files.json"

# Keep these tiny policy helpers independent of the repository package.  In a
# promotion process, no ``aigp_loop`` code is imported until effective argv has
# been parsed and the reviewed source manifest has been verified.
MAX_REPLAY_POLICY_BYTES = 2 * 1024 * 1024
_PYCACHE_CONTEXT: Optional[tempfile.TemporaryDirectory[str]] = None
_PYCACHE_PREFIX: Optional[Path] = None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json_loads(text: str) -> Any:
    return _strict_local_json(text.encode("utf-8"))


def read_secure_regular_file(
    path: Path | str, *, maximum_bytes: Optional[int] = None
) -> bytes:
    # Deliberately lazy: evidence entry points have completed their standalone
    # bootstrap before this import is reached.
    from aigp_loop._util import read_secure_regular_file as trusted_reader

    return trusted_reader(path, maximum_bytes=maximum_bytes)


def _secure_replay_host_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or candidate.drive
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("trusted replay path is unsafe")
    probe = root
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for component in candidate.parts:
        probe = probe / component
        info = probe.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & reparse_flag
        ):
            raise ValueError("trusted replay path contains indirection")
    if not stat.S_ISREG(probe.lstat().st_mode):
        raise ValueError("trusted replay path is not a regular file")
    return probe


def _stable_replay_host_read(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY
    for optional in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, optional, 0))
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        named_before = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or not os.path.samestat(before, named_before)
            or before.st_size > maximum_bytes
        ):
            raise ValueError("trusted replay file exceeds stable regular-file boundary")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            first = handle.read(maximum_bytes + 1)
            handle.seek(0)
            second = handle.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
        named_after = path.stat(follow_symlinks=False)
    finally:
        os.close(descriptor)
    signature = lambda info: (
        info.st_mode,
        info.st_dev,
        info.st_ino,
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
    )
    if (
        len(first) > maximum_bytes
        or first != second
        or signature(before) != signature(after)
        or not os.path.samestat(after, named_after)
    ):
        raise ValueError("trusted replay file mutated while being read")
    return first


def _strict_local_json(payload: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )


def _reject_replay_host_import_collisions(root: Path) -> None:
    native_suffixes = (".pyd", ".so", ".dll", ".dylib")
    package = root / "aigp_loop"
    entries = tuple(package.iterdir())
    for entry in root.iterdir():
        folded = entry.name.casefold()
        if folded == "aigp_loop.py" or (
            folded.startswith("aigp_loop.") and folded.endswith(native_suffixes)
        ):
            raise ValueError("trusted replay import-boundary collision")
    trusted_stems = {"_util", "ledger", "promotion", "replay"}
    for entry in entries:
        folded = entry.name.casefold()
        if folded.startswith("__init__.") and folded.endswith(
            (*native_suffixes, ".pyc", ".pyo")
        ):
            raise ValueError("trusted replay import-boundary collision")
        if entry.is_dir() and folded in trusted_stems:
            raise ValueError("trusted replay import-boundary collision")
        for stem in trusted_stems:
            if folded.startswith(stem + ".") and folded.endswith(
                (*native_suffixes, ".pyc", ".pyo")
            ):
                raise ValueError("trusted replay import-boundary collision")


def _reject_replay_host_bytecode(root: Path) -> None:
    """Reject source-adjacent bytecode without executing Git or repo code."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    package = root / "aigp_loop"
    package_info = package.lstat()
    if stat.S_ISLNK(package_info.st_mode) or (
        getattr(package_info, "st_file_attributes", 0) & reparse_flag
    ):
        raise ValueError("trusted replay package contains indirection")
    if not stat.S_ISDIR(package_info.st_mode):
        raise ValueError("trusted replay package is not a directory")

    # ``aigp_loop.__init__`` executes before the requested submodule and can
    # reach every package descendant.  Refuse all source-adjacent caches,
    # including ignored/untracked and case-variant spellings.
    pending = [package]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                folded = entry.name.casefold()
                if folded == "__pycache__" or folded.endswith((".pyc", ".pyo")):
                    raise ValueError("trusted replay import boundary contains bytecode")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or (
                    getattr(info, "st_file_attributes", 0) & reparse_flag
                ):
                    # Exact reviewed files are checked again by the manifest;
                    # do not recurse through any other import alternative.
                    continue
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))

    scripts = root / "scripts"
    scripts_info = scripts.lstat()
    if stat.S_ISLNK(scripts_info.st_mode) or (
        getattr(scripts_info, "st_file_attributes", 0) & reparse_flag
    ) or not stat.S_ISDIR(scripts_info.st_mode):
        raise ValueError("trusted replay scripts boundary is unsafe")
    with os.scandir(scripts) as entries:
        for entry in entries:
            folded = entry.name.casefold()
            if folded == "__pycache__" or (
                folded.startswith("aigp_replay.")
                and folded.endswith((".pyc", ".pyo"))
            ):
                raise ValueError("trusted replay import boundary contains bytecode")


def _activate_fresh_pycache_prefix(root: Path) -> Path:
    global _PYCACHE_CONTEXT, _PYCACHE_PREFIX

    context = tempfile.TemporaryDirectory(prefix="aigp-replay-pycache-")
    prefix = Path(context.name).resolve(strict=True)
    if root == prefix or root in prefix.parents:
        context.cleanup()
        raise RuntimeError("trusted replay bytecode prefix must be external")
    _PYCACHE_CONTEXT = context
    _PYCACHE_PREFIX = prefix
    sys.pycache_prefix = str(prefix)
    os.environ["PYTHONPYCACHEPREFIX"] = str(prefix)
    return prefix


def _verify_replay_host_manifest(argument: str) -> None:
    if type(argument) is not str or argument != _TRUSTED_MANIFEST_PATH:
        raise ValueError(
            f"trusted replay manifest must be {_TRUSTED_MANIFEST_PATH}"
        )
    manifest_path = _secure_replay_host_file(_REPO, _TRUSTED_MANIFEST_PATH)
    manifest = _strict_local_json(
        _stable_replay_host_read(
            manifest_path, maximum_bytes=_MAX_TRUSTED_MANIFEST_BYTES
        )
    )
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "files"}
        or manifest.get("schema") != "aigp-trusted-evaluator-files/1"
        or type(manifest.get("files")) is not dict
        or not _REPLAY_HOST_FILES <= set(manifest["files"])
    ):
        raise ValueError("trusted replay manifest has an invalid exact schema")
    for relative_name in _REPLAY_HOST_FILES:
        expected = manifest["files"].get(relative_name)
        if (
            type(expected) is not str
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("trusted replay manifest contains an invalid digest")
        target = _secure_replay_host_file(_REPO, relative_name)
        payload = _stable_replay_host_read(
            target, maximum_bytes=_MAX_TRUSTED_SOURCE_BYTES
        )
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"trusted replay host hash mismatch: {relative_name}")


def _bootstrap_trusted_replay_host(
    trusted_manifest: Optional[str], *, promotion_evidence: bool
) -> None:
    """Establish the source boundary before importing the evaluator package."""

    _reject_replay_host_import_collisions(_REPO)
    if promotion_evidence:
        _reject_replay_host_bytecode(_REPO)
    if trusted_manifest is not None:
        _verify_replay_host_manifest(trusted_manifest)
    _activate_fresh_pycache_prefix(_REPO)
    if str(_REPO) not in sys.path:
        # Append after the isolated interpreter's stdlib/site paths so a
        # checkout cannot shadow Python/NumPy dependencies in the host.
        sys.path.append(str(_REPO))
    package_spec = importlib.util.find_spec("aigp_loop")
    if (
        package_spec is None
        or package_spec.origin is None
        or Path(package_spec.origin).resolve()
        != _REPO / "aigp_loop" / "__init__.py"
    ):
        raise RuntimeError("trusted replay package did not resolve to this checkout")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser(
        "verify",
        help="validate manifest, hashes, and frames",
        allow_abbrev=False,
    )
    verify.add_argument("bundle")
    verify.add_argument("--skip-frame-decode", action="store_true")

    score = commands.add_parser(
        "score", help="score a labeled replay session", allow_abbrev=False
    )
    score.add_argument("bundle")
    score.add_argument("--annotations")
    score.add_argument(
        "--processor",
        help="optional local deterministic module:function frame processor",
    )

    corpus = commands.add_parser(
        "corpus",
        help="score a frozen multi-session golden corpus",
        allow_abbrev=False,
    )
    corpus.add_argument("manifest")
    corpus.add_argument("--processor", required=True)
    corpus.add_argument("--max-center-error-px", type=float, default=80.0)
    corpus.add_argument("--out")
    corpus.add_argument("--isolation-wrapper", required=True)
    corpus.add_argument("--isolation-wrapper-sha256", required=True)
    corpus.add_argument("--trusted-manifest", required=True)
    corpus.add_argument(
        "--candidate-worktree",
        default=os.environ.get("AIGP_CANDIDATE_WORKTREE"),
        help="exact candidate Git worktree visible only to the isolated worker",
    )
    score.add_argument("--max-center-error-px", type=float, default=80.0)
    score.add_argument("--out")
    score.add_argument("--isolation-wrapper")
    score.add_argument("--isolation-wrapper-sha256")
    score.add_argument("--trusted-manifest")
    score.add_argument(
        "--candidate-worktree",
        default=os.environ.get("AIGP_CANDIDATE_WORKTREE"),
    )
    score.add_argument(
        "--policy",
        help="versioned JSON policy; returns exit 2 on missing/violated evidence",
    )

    split = commands.add_parser(
        "split",
        help="assign whole sessions to train/validation groups",
        allow_abbrev=False,
    )
    split.add_argument("bundles", nargs="+")
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--salt", default="aigp-vq2-session-split-v1")

    # Parse the caller-provided effective argv, not process-global ``sys.argv``.
    # This is essential for embedded invocation and also lets argparse handle
    # both ``--flag value`` and ``--flag=value`` before trust decisions.
    args = parser.parse_args(argv)
    trusted_manifest = getattr(args, "trusted_manifest", None)
    if trusted_manifest is not None and trusted_manifest != _TRUSTED_MANIFEST_PATH:
        parser.error(
            f"--trusted-manifest must be {_TRUSTED_MANIFEST_PATH}"
        )
    if args.command == "score":
        if args.processor and (
            not args.isolation_wrapper or not args.isolation_wrapper_sha256
        ):
            parser.error(
                "candidate --processor requires a pinned --isolation-wrapper "
                "and --isolation-wrapper-sha256"
            )
        if args.processor and not args.candidate_worktree:
            parser.error(
                "candidate --processor requires an explicit --candidate-worktree"
            )
        if args.processor and args.isolation_wrapper and not args.trusted_manifest:
            parser.error(
                "isolated candidate scoring requires --trusted-manifest"
            )
    elif args.command == "corpus" and not args.candidate_worktree:
        parser.error("corpus scoring requires an explicit --candidate-worktree")

    evidence_mode = args.command == "corpus" or (
        args.command == "score" and bool(args.processor)
    )
    if evidence_mode and trusted_manifest != _TRUSTED_MANIFEST_PATH:
        parser.error(
            f"replay evidence requires --trusted-manifest {_TRUSTED_MANIFEST_PATH}"
        )
    _bootstrap_trusted_replay_host(
        trusted_manifest, promotion_evidence=evidence_mode
    )

    # These are intentionally local imports.  The standalone bootstrap above
    # has already rejected import alternatives/bytecode and verified the
    # canonical manifest for every promotion-evidence path.
    from aigp_loop.replay import (
        ReplayBundleReader,
        evaluate_score_policy,
        evaluation_input_hash,
        evaluation_result_hash,
        grouped_session_split,
        score_bundle,
        score_corpus,
    )

    exit_code = 0
    if args.command == "verify":
        result = ReplayBundleReader(args.bundle).verify(
            verify_frames=not args.skip_frame_decode
        )
    elif args.command == "score":
        result = score_bundle(
            args.bundle,
            annotations_path=args.annotations,
            processor_spec=args.processor,
            max_center_error_px=args.max_center_error_px,
            isolation_wrapper=args.isolation_wrapper,
            isolation_wrapper_sha256=args.isolation_wrapper_sha256,
            candidate_worktree=args.candidate_worktree,
        )
        if args.policy:
            try:
                policy_payload = read_secure_regular_file(
                    args.policy, maximum_bytes=MAX_REPLAY_POLICY_BYTES
                )
            except ValueError as exc:
                if "exceeds resource limit" in str(exc):
                    raise ValueError("replay policy exceeds resource limit") from exc
                raise
            try:
                policy = strict_json_loads(policy_payload.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("replay policy must be UTF-8") from exc
            result["policy"] = evaluate_score_policy(result, policy)
            result["policy"]["policy_file_sha256"] = sha256_bytes(
                policy_payload
            )
            try:
                result["evaluation_input_hash"] = evaluation_input_hash(
                    result, result["policy"]
                )
                result["evaluation_evidence_hash"] = result[
                    "evaluation_input_hash"
                ]
                result["evaluation_result_hash"] = evaluation_result_hash(
                    result, result["policy"]
                )
            except ValueError as exc:
                result["evaluation_evidence_hash"] = None
                result["policy"]["passed"] = False
                result["policy"]["violations"].append(
                    {
                        "metric": "evaluation_evidence_hash",
                        "reason": "missing_provenance",
                        "observed": str(exc),
                    }
                )
            if not result["policy"]["passed"]:
                exit_code = 2
    elif args.command == "corpus":
        result = score_corpus(
            args.manifest,
            processor_spec=args.processor,
            max_center_error_px=args.max_center_error_px,
            isolation_wrapper=args.isolation_wrapper,
            isolation_wrapper_sha256=args.isolation_wrapper_sha256,
            candidate_worktree=args.candidate_worktree,
        )
        if result["policy"]["passed"] is not True:
            exit_code = 2
    else:
        sessions = []
        for bundle in args.bundles:
            reader = ReplayBundleReader(bundle)
            sessions.append((reader.session_id, reader.dataset_hash))
        result = grouped_session_split(
            sessions,
            validation_fraction=args.validation_fraction,
            salt=args.salt,
        )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if getattr(args, "out", None):
        # Score output is a small derived report, never raw frame data.
        Path(args.out).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
