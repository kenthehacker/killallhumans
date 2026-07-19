"""Manage the resumable SQLite trial ledger and non-live scheduler."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_REPO = Path(__file__).resolve().parent.parent
_STARTUP_PYCACHE_CONTEXT = tempfile.TemporaryDirectory(
    prefix="aigp-trials-pycache-"
)
_STARTUP_PYCACHE_PREFIX = Path(_STARTUP_PYCACHE_CONTEXT.name).resolve(strict=True)
if _REPO == _STARTUP_PYCACHE_PREFIX or _REPO in _STARTUP_PYCACHE_PREFIX.parents:
    raise RuntimeError("trial bootstrap bytecode prefix must be external")
sys.pycache_prefix = str(_STARTUP_PYCACHE_PREFIX)
os.environ["PYTHONPYCACHEPREFIX"] = str(_STARTUP_PYCACHE_PREFIX)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from aigp_loop._util import (
    environment_fingerprint,
    git_provenance,
    json_hash,
    run_checked,
    secure_directory,
    secure_relative_regular_file,
    strict_json_load,
)
from aigp_loop.ledger import SCHEMA_VERSION, TrialKey, TrialLedger
from aigp_loop.promotion import Tier
from aigp_loop.scheduler import (
    GitWorktreePool,
    SingleMerger,
    TrialScheduler,
    load_tier_commands,
)


DEFAULT_LEDGER = Path(".aigp-loop") / "trials.sqlite3"


def _publish_json(path: Path | str, encoded: str, *, overwrite: bool) -> Path:
    """Atomically publish a complete file with an explicit overwrite policy."""

    target = Path(path).resolve()
    if not target.parent.is_dir():
        raise ValueError(f"output parent directory is missing: {target.parent}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp, target)
        else:
            # A same-directory hard-link is an atomic create-if-absent on both
            # NTFS and POSIX filesystems; unlike rename it never overwrites.
            os.link(temp, target)
            temp.unlink()
        return target
    finally:
        if temp.exists():
            temp.unlink()


def _json_mapping(path: Path | str) -> Mapping[str, Any]:
    value = strict_json_load(path)
    if not isinstance(value, Mapping):
        raise ValueError("configuration must be a JSON object")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="create/migrate the local ledger")
    prepare = commands.add_parser(
        "prepare-ladder-config",
        help="embed a frozen T0-T4 identity manifest and print TrialKey values",
    )
    prepare.add_argument("--base-config", required=True)
    prepare.add_argument("--tier-identities", required=True)
    prepare.add_argument(
        "--commands",
        required=True,
        help="reviewed T0-T4 command plan whose exact hashes enter the TrialKey",
    )
    prepare.add_argument("--out", required=True)
    prepare.add_argument(
        "--overwrite", action="store_true", help="atomically replace --out"
    )
    trust = commands.add_parser(
        "build-trusted-manifest",
        help="hash trusted evaluator/test files for operator review",
    )
    trust.add_argument("--repo", default=".")
    trust.add_argument("--out", required=True)
    trust.add_argument(
        "--overwrite", action="store_true", help="atomically replace --out"
    )
    trust.add_argument(
        "paths",
        nargs="+",
        help="repository-relative files or directories (directories include all files recursively)",
    )
    status = commands.add_parser("status", help="print structured trial rows")
    status.add_argument("--status", dest="trial_status")

    enqueue = commands.add_parser("enqueue", help="idempotently enqueue a candidate")
    enqueue.add_argument("--repo", default=".")
    enqueue.add_argument("--config", required=True)
    enqueue.add_argument("--dataset-hash", required=True)
    enqueue.add_argument("--evaluator-version", required=True)
    enqueue.add_argument("--seed", type=int, required=True)
    enqueue.add_argument("--name")
    enqueue.add_argument("--parent-trial-id")
    enqueue.add_argument("--simulator-build")

    run = commands.add_parser("run", help="run/resume one non-live leased trial")
    run.add_argument("--repo", default=".")
    run.add_argument("--worktree-root", required=True)
    run.add_argument("--commands", required=True)
    run.add_argument("--through", type=int, choices=range(5), default=4)
    run.add_argument("--owner", default=f"scheduler-{uuid.uuid4().hex}")

    round_command = commands.add_parser(
        "round", help="run one cohort tier and apply successive halving"
    )
    round_command.add_argument("--repo", default=".")
    round_command.add_argument("--worktree-root", required=True)
    round_command.add_argument("--commands", required=True)
    round_command.add_argument("--tier", type=int, choices=range(5), required=True)
    round_command.add_argument("--keep-fraction", type=float, default=0.5)
    round_command.add_argument("--minimum-survivors", type=int, default=1)
    round_command.add_argument("--owner", default=f"scheduler-{uuid.uuid4().hex}")

    importer = commands.add_parser(
        "import-history", help="one-time import of benchmark_history.jsonl"
    )
    importer.add_argument("source", nargs="?", default="benchmark_history.jsonl")

    merge = commands.add_parser(
        "merge", help="explicitly fast-forward one completed candidate"
    )
    merge.add_argument("trial_id")
    merge.add_argument("--repo", default=".")
    merge.add_argument("--owner", default=f"merger-{uuid.uuid4().hex}")

    args = parser.parse_args(argv)
    ledger = TrialLedger(args.ledger)
    if args.command == "init":
        result: Any = {"ledger": str(ledger.path), "schema_version": SCHEMA_VERSION}
    elif args.command == "prepare-ladder-config":
        config = dict(_json_mapping(args.base_config))
        identity_input = strict_json_load(args.tier_identities)
        if (
            type(identity_input) is not dict
            or set(identity_input) != {"schema", "tiers"}
            or identity_input.get("schema")
            != "aigp-promotion-ladder-evidence-identities/1"
            or type(identity_input.get("tiers")) is not list
            or len(identity_input["tiers"]) != 5
        ):
            raise ValueError("tier identities must be an exact T0-T4 ladder manifest")
        commands_by_tier = load_tier_commands(args.commands)
        if set(commands_by_tier) != set(Tier) - {Tier.T5_AUTHORIZED_LIVE}:
            raise ValueError("reviewed command plan must define exactly T0 through T4")
        identity_fields = {
            "tier",
            "dataset_hash",
            "config_hash",
            "seed",
            "repetitions",
            "evaluator_version",
        }
        enriched = []
        for identity in identity_input["tiers"]:
            if type(identity) is not dict or set(identity) != identity_fields:
                raise ValueError(
                    "tier evidence identity has missing or unknown fields"
                )
            tier_value = identity.get("tier")
            if type(tier_value) is not int or tier_value not in range(5):
                raise ValueError("tier evidence identity number must be 0..4")
            command = commands_by_tier[Tier(tier_value)]
            enriched.append(
                {
                    **identity,
                    "command_plan_sha256": json_hash(dataclasses.asdict(command)),
                }
            )
        manifest = {
            "schema": "aigp-promotion-ladder-manifest/2",
            "tiers": enriched,
        }
        # Reuse the scheduler's strict identity validator against a synthetic
        # row before materializing anything operators may enqueue.
        manifest_hash = json_hash(manifest)
        probe = {
            "resolved_config": {"promotion_ladder_manifest": manifest},
            "dataset_hash": manifest_hash,
            "evaluator_version": f"aigp-ladder/2:{manifest_hash}",
        }
        for tier in Tier:
            if tier <= Tier.T4_FULL_NON_LIVE:
                TrialScheduler._tier_identity_hash(probe, tier, required=True)
        config["promotion_ladder_manifest"] = manifest
        encoded = json.dumps(config, indent=2, sort_keys=True) + "\n"
        output_path = _publish_json(args.out, encoded, overwrite=args.overwrite)
        result = {
            "config": str(output_path),
            "config_hash": json_hash(config),
            "dataset_hash": manifest_hash,
            "evaluator_version": f"aigp-ladder/2:{manifest_hash}",
        }
    elif args.command == "build-trusted-manifest":
        from aigp_loop._util import read_secure_regular_file, sha256_bytes

        repository = secure_directory(args.repo)
        tracked_output = subprocess.run(
            ["git", "ls-files", "--cached", "-z"],
            cwd=str(repository),
            check=True,
            capture_output=True,
            timeout=30.0,
            shell=False,
        ).stdout
        tracked_names = [
            Path(os.fsdecode(raw_name))
            for raw_name in tracked_output.split(b"\0")
            if raw_name
        ]
        if any(
            any(part.casefold() == "__pycache__" for part in relative.parts)
            or relative.name.casefold().endswith((".pyc", ".pyo"))
            for relative in tracked_names
        ):
            raise ValueError(
                "trusted repository contains tracked executable bytecode/cache"
            )
        files = []
        for raw in args.paths:
            relative = Path(raw)
            if (
                relative.is_absolute()
                or relative.drive
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError("trusted path escapes repository")
            raw_candidate = repository / relative
            if not raw_candidate.exists():
                raise ValueError(f"trusted path is missing: {raw}")
            candidate = raw_candidate
            if candidate.is_dir():
                secure_directory(candidate)
                ignored_directories = {
                    ".git",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".ruff_cache",
                    "__pycache__",
                }
                for directory, names, filenames in os.walk(
                    candidate, topdown=True, followlinks=False
                ):
                    base = Path(directory)
                    secure_directory(base)
                    retained_names = []
                    for name in sorted(names):
                        if name.casefold() in {
                            ignored.casefold() for ignored in ignored_directories
                        }:
                            continue
                        secure_directory(base / name)
                        retained_names.append(name)
                    names[:] = retained_names
                    for filename in sorted(filenames):
                        if filename.casefold().endswith((".pyc", ".pyo")):
                            continue
                        path = base / filename
                        files.append(
                            secure_relative_regular_file(
                                repository, path.relative_to(repository)
                            )
                        )
            elif candidate.is_file():
                files.append(secure_relative_regular_file(repository, relative))
            else:
                raise ValueError(f"trusted path is missing: {raw}")
        hashes = {
            path.relative_to(repository).as_posix(): sha256_bytes(
                read_secure_regular_file(path)
            )
            for path in sorted(set(files))
        }
        if not hashes:
            raise ValueError("trusted manifest cannot be empty")
        manifest = {
            "schema": "aigp-trusted-evaluator-files/1",
            "files": hashes,
        }
        output_path = _publish_json(
            args.out,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            overwrite=args.overwrite,
        )
        result = {
            "manifest": str(output_path),
            "file_count": len(hashes),
            "manifest_hash": json_hash(manifest),
        }
    elif args.command == "status":
        result = ledger.list_trials(status=args.trial_status)
    elif args.command == "enqueue":
        config = _json_mapping(args.config)
        repo = Path(args.repo).resolve()
        if run_checked(["git", "status", "--porcelain"], cwd=repo):
            raise SystemExit(
                "refusing to enqueue a dirty candidate: commit the exact code first "
                "so its isolated worktree matches ledger provenance"
            )
        commit, dirty_hash, code_hash = git_provenance(repo)
        identifier, created = ledger.create_or_get_trial(
            key=TrialKey(
                code_hash=code_hash,
                config_hash=json_hash(config),
                dataset_hash=args.dataset_hash,
                seed=args.seed,
                evaluator_version=args.evaluator_version,
            ),
            commit_hash=commit,
            dirty_diff_hash=dirty_hash,
            resolved_config=config,
            environment_fingerprint=environment_fingerprint(),
            parent_trial_id=args.parent_trial_id,
            candidate_name=args.name,
            simulator_build=args.simulator_build,
        )
        result = {"trial_id": identifier, "created": created}
    elif args.command in {"run", "round"}:
        scheduler = TrialScheduler(
            ledger,
            GitWorktreePool(args.repo, args.worktree_root),
            load_tier_commands(args.commands),
            owner=args.owner,
        )
        if args.command == "run":
            identifier = scheduler.run_once(through=Tier(args.through))
            result = {"trial_id": identifier, "ran": identifier is not None}
        else:
            decision = scheduler.run_round(
                Tier(args.tier),
                keep_fraction=args.keep_fraction,
                minimum_survivors=args.minimum_survivors,
            )
            result = {"decision": decision, "ran": decision is not None}
    elif args.command == "import-history":
        count = ledger.import_legacy_benchmark_history(args.source)
        result = {"source": str(Path(args.source).resolve()), "historical_rows": count}
    else:
        commit = SingleMerger(ledger, args.repo, owner=args.owner).merge_completed(
            args.trial_id
        )
        result = {"trial_id": args.trial_id, "merged_commit": commit}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
