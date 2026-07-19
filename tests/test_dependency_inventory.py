from __future__ import annotations

import json
import subprocess
from pathlib import Path

import scripts.dependency_inventory as inventory_module
from scripts.dependency_inventory import _repository_properties, build_inventory, main


_ROOT = Path(__file__).resolve().parents[1]
_SETUP_VENV = _ROOT / "scripts" / "setup_venv.sh"


def test_inventory_contains_installed_pytest_and_repository_metadata():
    inventory = build_inventory()
    assert inventory["bomFormat"] == "CycloneDX"
    assert inventory["specVersion"] == "1.5"
    names = {component["name"].lower() for component in inventory["components"]}
    assert "pytest" in names
    properties = {
        item["name"]: item["value"]
        for item in inventory["metadata"]["properties"]
    }
    assert len(properties["aigp:repository_commit"]) == 40
    assert properties["aigp:repository_dirty"] in {"true", "false", "unknown"}
    assert properties["aigp:repository_snapshot_stable"] in {"true", "false"}
    assert len(properties["aigp:tracked_diff_sha256"]) == 64
    assert len(properties["aigp:untracked_tree_sha256"]) == 64
    assert int(properties["aigp:untracked_file_count"]) >= 0


def test_repository_metadata_binds_untracked_file_contents(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "inventory-test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Inventory Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path, check=True
    )

    untracked = tmp_path / "new-source.py"
    untracked.write_text("VALUE = 1\n", encoding="utf-8")
    first = {
        item["name"]: item["value"] for item in _repository_properties(tmp_path)
    }
    untracked.write_text("VALUE = 2\n", encoding="utf-8")
    second = {
        item["name"]: item["value"] for item in _repository_properties(tmp_path)
    }

    assert first["aigp:repository_commit"] == second["aigp:repository_commit"]
    assert first["aigp:tracked_diff_sha256"] == second["aigp:tracked_diff_sha256"]
    assert first["aigp:untracked_file_count"] == "1"
    assert second["aigp:untracked_file_count"] == "1"
    assert first["aigp:untracked_tree_sha256"] != second["aigp:untracked_tree_sha256"]


def test_repository_metadata_reports_git_failure_as_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory_module, "_git_bytes", lambda *args, **kwargs: None)

    properties = {
        item["name"]: item["value"] for item in _repository_properties(tmp_path)
    }

    assert properties["aigp:repository_commit"] == "unknown"
    assert properties["aigp:repository_dirty"] == "unknown"
    assert properties["aigp:tracked_diff_sha256"] == "unknown"
    assert properties["aigp:untracked_tree_sha256"] == "unknown"
    assert properties["aigp:untracked_file_count"] == "unknown"
    assert properties["aigp:repository_snapshot_stable"] == "false"


def test_repository_metadata_rejects_never_stable_snapshot(tmp_path, monkeypatch):
    calls = {"status": 0}

    def changing_git(_repo_root, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return ("a" * 40 + "\n").encode("ascii")
        if args and args[0] == "status":
            calls["status"] += 1
            return f"status-{calls['status']}".encode("ascii")
        if args and args[0] == "diff":
            return b""
        if args and args[0] == "ls-files":
            return b""
        raise AssertionError(args)

    monkeypatch.setattr(inventory_module, "_git_bytes", changing_git)
    properties = {
        item["name"]: item["value"] for item in _repository_properties(tmp_path)
    }

    assert properties["aigp:repository_commit"] == "unknown"
    assert properties["aigp:tracked_diff_sha256"] == "unknown"
    assert properties["aigp:repository_snapshot_stable"] == "false"


def test_repository_metadata_accepts_sha256_object_format(tmp_path, monkeypatch):
    commit = "a" * 64

    def stable_git(_repo_root, *args):
        if args == ("rev-parse", "HEAD"):
            return (commit + "\n").encode("ascii")
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b""
        if args == ("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD"):
            return b""
        raise AssertionError(args)

    monkeypatch.setattr(inventory_module, "_git_bytes", stable_git)
    monkeypatch.setattr(
        inventory_module, "_untracked_tree_identity", lambda _repo_root: ("b" * 64, "0")
    )
    properties = {
        item["name"]: item["value"] for item in _repository_properties(tmp_path)
    }

    assert properties["aigp:repository_snapshot_stable"] == "true"
    assert properties["aigp:repository_commit"] == commit


def test_inventory_cli_writes_parseable_json(tmp_path):
    destination = tmp_path / "inventory.json"
    assert main(["--output", str(destination)]) == 0
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["components"]
    assert loaded["dependencies"]


def test_bash_setup_defaults_to_locked_development_profile_and_isolates_legacy():
    source = _SETUP_VENV.read_text(encoding="utf-8")
    assert (
        'REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT_DIR/requirements/'
        'development-test.lock.txt}"'
    ) in source
    assert 'VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"' in source
    assert 'LEGACY_REQUIREMENTS="$(resolve_path ' in source
    assert "refusing to mix legacy simulation dependencies" in source
    assert ".aigp-environment-profile" in source
    assert "already bound to a different requirements profile" in source
    assert "refusing to adopt populated unbound environment" in source
    assert "os.link(temporary, marker)" in source
    assert source.index("refusing to mix legacy simulation dependencies") < source.index(
        '"$PYTHON_BIN" -m venv "$VENV_DIR"'
    )
    assert source.index("os.link(temporary, marker)") < source.index(
        '"$PYTHON_BIN" -m venv "$VENV_DIR"'
    )
    assert source.index('EXISTING_PROFILE="$(<"$PROFILE_MARKER")"') < source.index(
        '"$PYTHON_BIN" -m venv "$VENV_DIR"'
    )
    assert "pip install --upgrade" not in source
    assert 'python -m pip install -r "$REQUIREMENTS_FILE"' in source
