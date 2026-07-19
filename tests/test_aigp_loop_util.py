import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from aigp_loop._util import (
    _untracked_files_digest,
    git_provenance,
    read_secure_regular_file,
    strict_json_load,
    strict_json_loads,
)


def test_strict_json_rejects_constants_and_duplicate_keys():
    with pytest.raises(ValueError, match="constant"):
        strict_json_loads('{"value": NaN}')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        strict_json_loads('{"valid": true, "valid": false}')


def test_strict_json_file_rejects_symlinked_input(tmp_path):
    target = tmp_path / "target.json"
    target.write_text('{"valid":true}', encoding="utf-8")
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("creating a file symlink is unavailable on this host")
    with pytest.raises(ValueError, match="symlink|reparse"):
        strict_json_load(alias)


def test_untracked_digest_length_prefixes_names_and_content(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_bytes(b"bc")
    (second / "ab").write_bytes(b"c")
    assert _untracked_files_digest(first.resolve(), ["a"]) != _untracked_files_digest(
        second.resolve(), ["ab"]
    )


def test_untracked_digest_rejects_symlink(tmp_path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("secret", encoding="utf-8")
    link = root / "linked.txt"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("creating a file symlink is unavailable on this Windows host")
    with pytest.raises(ValueError, match="symlink"):
        _untracked_files_digest(root, ["linked.txt"])


def test_secure_snapshot_rejects_in_place_metadata_drift(tmp_path, monkeypatch):
    target = tmp_path / "mutable.bin"
    target.write_bytes(b"stable bytes")
    real_fstat = os.fstat
    calls = 0

    def drifting_fstat(fd):
        nonlocal calls
        info = real_fstat(fd)
        calls += 1
        if calls != 2:
            return info
        return SimpleNamespace(
            st_mode=info.st_mode,
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_size=info.st_size + 1,
            st_mtime_ns=info.st_mtime_ns,
            st_ctime_ns=info.st_ctime_ns,
        )

    monkeypatch.setattr(os, "fstat", drifting_fstat)
    with pytest.raises(ValueError, match="mutated while reading"):
        read_secure_regular_file(target)


def test_secure_snapshot_ceiling_is_enforced_after_path_replacement(
    tmp_path, monkeypatch
):
    import aigp_loop._util as util

    target = tmp_path / "snapshot.bin"
    replacement = tmp_path / "replacement.bin"
    target.write_bytes(b"small")
    replacement.write_bytes(b"x" * 1_000_000)
    real_secure = util.secure_regular_file
    real_read = os.read
    read_sizes = []

    def replace_after_path_check(path):
        checked = real_secure(path)
        replacement.replace(target)
        return checked

    def observed_read(descriptor, size):
        read_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(util, "secure_regular_file", replace_after_path_check)
    monkeypatch.setattr(os, "read", observed_read)
    with pytest.raises(ValueError, match="exceeds resource limit"):
        util.read_secure_regular_file(target, maximum_bytes=64)
    assert read_sizes == []


def test_git_provenance_rejects_malformed_head_identity(tmp_path, monkeypatch):
    import aigp_loop._util as util

    monkeypatch.setattr(util, "run_checked", lambda *_args, **_kwargs: "not-a-commit")
    with pytest.raises(ValueError, match="hexadecimal object identity"):
        git_provenance(tmp_path)
