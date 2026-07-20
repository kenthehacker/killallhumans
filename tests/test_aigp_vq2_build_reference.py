import ast
import hashlib
import json
import struct
from pathlib import Path

import pytest

import scripts.aigp_vq2_build_reference as reference


_NATIVE_WINDOWS_PRIVATE_ACL_HANDLE = reference._windows_private_acl_handle


def _fstring(value):
    encoded = value.encode("utf-8") + b"\x00"
    return struct.pack("<i", len(encoded)) + encoded


def _secondary_indices(
    directories, *, path_locations=None, directory_locations=None
):
    paths = []
    raw_entries = []
    for directory, filenames in directories:
        raw_entries.extend((directory, filename) for filename in filenames)
    if path_locations is None:
        path_locations = list(range(len(raw_entries)))
    if directory_locations is None:
        directory_locations = list(range(len(raw_entries)))
    assert len(path_locations) == len(raw_entries)
    assert len(directory_locations) == len(raw_entries)
    full = bytearray(struct.pack("<I", len(directories)))
    location_index = 0
    for directory, filenames in directories:
        full += _fstring(directory)
        full += struct.pack("<I", len(filenames))
        for filename in filenames:
            full += _fstring(filename)
            full += struct.pack("<i", directory_locations[location_index])
            paths.append(f"{directory}{filename}")
            location_index += 1
    path_hash = bytearray(struct.pack("<I", len(paths)))
    for index, location in enumerate(path_locations, start=1):
        path_hash += struct.pack("<Qi", index, location)
    return bytes(path_hash), bytes(full), paths


def _synthetic_pak(
    *,
    mount="../../../",
    directories=None,
    path_locations=None,
    directory_locations=None,
    encoded_entries=bytes(range(16)),
):
    if directories is None:
        directories = [
            (
                "FlightSim/Content/Anduril-TrackEditor/Gates/",
                ["BP_gate.uasset", "SM_Gates_Anduril_Square_Combined.uasset"],
            ),
            ("FlightSim/Content/levels/", ["MAP_arsenal_track01.uasset"]),
            (
                "FlightSim/Content/levelsMaster/",
                ["MAP_arsenal_master.uasset"],
            ),
        ]
    path_hash, full_directory, paths = _secondary_indices(
        directories,
        path_locations=path_locations,
        directory_locations=directory_locations,
    )
    path_offset = 64
    full_offset = path_offset + len(path_hash)
    index_offset = full_offset + len(full_directory)
    primary = bytearray()
    primary += _fstring(mount)
    primary += struct.pack("<IQ", len(paths), 0x123456789ABCDEF0)
    primary += struct.pack("<IQQ", 1, path_offset, len(path_hash))
    primary += hashlib.sha1(path_hash).digest()
    primary += struct.pack("<IQQ", 1, full_offset, len(full_directory))
    primary += hashlib.sha1(full_directory).digest()
    primary += struct.pack("<I", len(encoded_entries))
    primary += encoded_entries
    primary += struct.pack("<I", 0)
    compression = b"".join(
        name.encode("ascii") + b"\x00" * (32 - len(name))
        for name in ("Zlib", "Gzip", "Oodle", "Zstd", "LZ4")
    )
    footer = b"".join(
        (
            b"\x00" * 16,
            b"\x00",
            struct.pack("<IIQQ", reference.PAK_MAGIC, 11, index_offset, len(primary)),
            hashlib.sha1(primary).digest(),
            compression,
        )
    )
    assert len(footer) == reference.PAK_V11_FOOTER_BYTES
    pak = bytearray(b"fixture" + b"\x00" * (path_offset - len(b"fixture")))
    pak += path_hash
    pak += full_directory
    pak += primary
    pak += footer
    return bytes(pak), {
        "path_offset": path_offset,
        "full_offset": full_offset,
        "index_offset": index_offset,
        "index_size": len(primary),
        "footer_offset": len(pak) - len(footer),
    }


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clearance_value(hashes):
    return {
        "schema": reference.RULES_SCHEMA,
        "record_id": "synthetic-review-only",
        "reviewer": "synthetic-test-reviewer",
        "reviewed_at_utc": "2026-07-20T00:00:00Z",
        "authority_basis": "synthetic fixture; no production authority",
        "build_sha256": hashes,
        "asset_scope": sorted(reference.FROZEN_PACKAGE_STEMS),
        "local_read_only_derivation_permitted": True,
        "competition_use_permitted": True,
        "publication_limits": [],
    }


def _evidence(identifier):
    return {
        "id": identifier,
        "producer_id": f"producer-{identifier}",
        "method_id": f"method-{identifier}",
        "artifact_sha256": hashlib.sha256(identifier.encode()).hexdigest(),
    }


def _file_identity(path):
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _hash(path),
    }


def _candidate_value(
    *, launcher, payload, pak, clearance_path, clearance_bytes, pak_index
):
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    identity += [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    mesh = reference.FROZEN_PACKAGE_STEMS[1]
    source = Path(reference.__file__).resolve()
    return {
        "schema": reference.CANDIDATE_SCHEMA,
        "build": {
            "build": 3385,
            "mode": "Training",
            "launcher": _file_identity(launcher),
            "payload": _file_identity(payload),
            "pak": _file_identity(pak),
            "pak_index": {
                key: pak_index[key]
                for key in (
                    "magic",
                    "version",
                    "encryption_guid",
                    "encrypted",
                    "index_offset",
                    "index_size",
                    "index_sha1",
                    "path_hash_seed",
                    "path_hash_index_offset",
                    "path_hash_index_size",
                    "path_hash_index_sha1",
                    "full_directory_index_offset",
                    "full_directory_index_size",
                    "full_directory_index_sha1",
                    "compression_names",
                    "mount_point",
                    "entry_count",
                )
            },
            "package_paths": pak_index["candidate_package_paths"],
        },
        "parser": {
            "implementation_id": reference.PARSER_IMPLEMENTATION,
            "source": _file_identity(source),
            "interpreter_sha256": reference.INTERPRETER_SHA256,
            "dependencies": [],
            "config_sha256": reference.parser_config_sha256(),
        },
        "geometry": {
            "mesh_package": mesh,
            "active_lod": 0,
            "render_not_collision_only": True,
            "coordinate_convention": "right-handed-meters",
            "units": "m",
            "features": [
                {
                    "id": "v0",
                    "kind": "vertex",
                    "coordinates": [[0.0, 0.0, 0.0]],
                    "references": [],
                    "evidence": [_evidence("vertex-0")],
                },
                {
                    "id": "v1",
                    "kind": "vertex",
                    "coordinates": [[1.0, 0.0, 0.0]],
                    "references": [],
                    "evidence": [_evidence("vertex-1")],
                },
                {
                    "id": "e0",
                    "kind": "edge",
                    "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "references": ["v0", "v1"],
                    "evidence": [_evidence("edge-0")],
                },
                {
                    "id": "s0",
                    "kind": "surface",
                    "coordinates": [[0.0, 0.0, 0.0]],
                    "references": ["e0"],
                    "evidence": [_evidence("surface-0")],
                },
            ],
            "bounds": {
                "planarity": 0.001,
                "aspect": 0.001,
                "thickness": 0.001,
                "bevel": 0.001,
            },
        },
        "transform_chain": {
            "links": [
                {
                    "id": "mesh-to-component",
                    "parent_frame": "mesh",
                    "child_frame": "component",
                    "matrix_row_major": identity,
                    "determinant": 1.0,
                    "singular_values": [1.0, 1.0, 1.0],
                    "handedness": "right",
                    "scale_model": "uniform",
                    "evidence": [_evidence("transform")],
                }
            ],
            "active_actor_overrides": [
                {
                    "id": "component-to-actor",
                    "parent_frame": "component",
                    "child_frame": "actor",
                    "matrix_row_major": identity,
                    "determinant": 1.0,
                    "singular_values": [1.0, 1.0, 1.0],
                    "handedness": "right",
                    "scale_model": "uniform",
                    "evidence": [_evidence("actor-override")],
                }
            ],
        },
        "training_linkage": {
            "master_map": "MAP_arsenal_master",
            "track_map": "MAP_arsenal_track01",
            "gate_blueprint": "BP_gate",
            "component": "GateMesh",
            "mesh": mesh,
            "material": "gate-material",
            "lod": 0,
            "udp_camera": "FPVCamera-to-UDP-JPEG",
            "proved": True,
            "evidence": [_evidence("linkage")],
        },
        "visibility": {
            "model_id": "visibility-v1",
            "surface_ids": ["s0"],
            "feature_ids": ["e0", "v0", "v1"],
            "front_policy": "explicit-front",
            "back_policy": "explicit-back",
            "bevel_policy": "explicit-bevel",
            "clipping_policy": "reject-clipped",
            "occlusion_policy": "reject-occluded",
            "systematic_bounds": {
                "front_back_px": 0.5,
                "bevel_px": 0.5,
                "clipping_px": 0.5,
                "occlusion_px": 0.5,
            },
            "evidence": [_evidence("visibility")],
        },
        "uncertainty": {
            "conditional_pixel_model_id": "conditional-pixel-v1",
            "shared_nuisance_ledger_id": "shared-ledger-v1",
            "pixel_center_convention": "integer-coordinates-are-pixel-centers",
            "bounds": {
                "render_lod_px": 0.5,
                "material_px": 0.5,
                "antialias_px": 0.5,
                "jpeg_px": 0.5,
                "annotation_px": 0.5,
                "geometry_units": 0.001,
                "transform_units": 0.001,
            },
            "evidence": [_evidence("uncertainty")],
        },
        "independent_checks": [
            {
                "check_id": "check-a",
                "implementation_id": "implementation-a",
                "producer_id": "checker-a",
                "input_sha256": "2" * 64,
                "output_sha256": "3" * 64,
                "passed": True,
            },
            {
                "check_id": "check-b",
                "implementation_id": "implementation-b",
                "producer_id": "checker-b",
                "input_sha256": "2" * 64,
                "output_sha256": "4" * 64,
                "passed": True,
            },
        ],
        "rules": {
            "clearance": {
                "path": str(clearance_path.resolve()),
                "size_bytes": len(clearance_bytes),
                "sha256": hashlib.sha256(clearance_bytes).hexdigest(),
            },
            "record_id": "synthetic-review-only",
        },
        "annotation_contract": {
            "observation_schema": reference.OBSERVATION_SCHEMA,
            "producer_id": "annotation-producer",
            "producer_sha256": "5" * 64,
            "preprocessing_sha256": "6" * 64,
            "correspondence_sha256": "7" * 64,
            "rejection_sha256": "8" * 64,
            "covariance_sha256": "9" * 64,
            "shared_nuisance_ledger_sha256": "a" * 64,
            "checker_id": "annotation-checker",
            "checker_sha256": "b" * 64,
        },
    }


@pytest.fixture
def synthetic_context(tmp_path, monkeypatch):
    root = tmp_path / "reference"
    root.mkdir(mode=0o700)
    launcher = _write(root / "FlightSim.exe", b"synthetic launcher")
    payload = _write(root / "DCGame-Win64-Shipping.exe", b"synthetic payload")
    pak_bytes, offsets = _synthetic_pak()
    pak = _write(root / "FlightSim-WindowsNoEditor.pak", pak_bytes)
    hashes = {"launcher": _hash(launcher), "payload": _hash(payload), "pak": _hash(pak)}
    monkeypatch.setattr(reference, "REFERENCE_ROOT", root)
    monkeypatch.setattr(reference, "LAUNCHER_PATH", launcher.resolve())
    monkeypatch.setattr(reference, "PAYLOAD_PATH", payload.resolve())
    monkeypatch.setattr(reference, "PAK_PATH", pak.resolve())
    monkeypatch.setattr(reference, "EXPECTED_SHA256", hashes)
    if reference.os.name == "nt":
        monkeypatch.setattr(reference, "_windows_private_acl", lambda _path: True)
        monkeypatch.setattr(
            reference, "_windows_private_acl_handle", lambda _handle: True
        )
    clearance = _clearance_value(hashes)
    clearance_bytes = reference.canonical_json_bytes(clearance)
    clearance_path = _write(root / "rules-clearance.json", clearance_bytes)
    with pak.open("rb") as handle:
        pak_index = reference.inspect_pak(handle, pak.stat().st_size)
    return {
        "root": root,
        "launcher": launcher,
        "payload": payload,
        "pak": pak,
        "pak_bytes": pak_bytes,
        "offsets": offsets,
        "hashes": hashes,
        "clearance": clearance,
        "clearance_bytes": clearance_bytes,
        "clearance_path": clearance_path,
        "pak_index": pak_index,
    }


def test_canonical_json_and_envelope_hash_are_exact():
    value = {"z": "é", "a": [1, True]}
    expected = b'{"a":[1,true],"z":"\\u00e9"}\n'
    assert reference.canonical_json_bytes(value) == expected
    encoded = reference.envelope_bytes("schema/1", value)
    decoded = json.loads(encoded)
    assert set(decoded) == {"schema", "payload", "payload_sha256"}
    assert decoded["payload_sha256"] == hashlib.sha256(expected).hexdigest()
    assert encoded.endswith(b"\n") and not encoded.startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize(
    "payload",
    [
        b'{"x":1,"x":2}',
        b'{"nested":{"x":1,"x":2}}',
        b'{"x":NaN}',
        b'{"x":1e999}',
        b"\xef\xbb\xbf{}",
        b'"\xff"',
        b'{"x":' + b"9" * 5000 + b"}",
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_bom_or_bad_utf8(payload):
    with pytest.raises(reference.ReferenceToolError):
        reference.strict_json_bytes(payload)


def test_rules_clearance_is_exact_and_never_self_authorizing(synthetic_context):
    value = synthetic_context["clearance"]
    assert reference.validate_rules_clearance(value) is value
    for mutation in (
        {**value, "competition_use_permitted": 1},
        {**value, "unknown": True},
        {**value, "asset_scope": list(reversed(value["asset_scope"]))},
        {
            **value,
            "build_sha256": {
                **value["build_sha256"],
                "pak": value["build_sha256"]["pak"].upper(),
            },
        },
    ):
        with pytest.raises(reference.ReferenceToolError):
            reference.validate_rules_clearance(mutation)


def test_rules_clearance_publication_limits_preserve_authority_order(
    synthetic_context
):
    value = {
        **synthetic_context["clearance"],
        "publication_limits": ["second", "first", "first"],
    }
    assert reference.validate_rules_clearance(value) is value


def test_synthetic_v11_pak_parses_only_frozen_index_facts(synthetic_context):
    pak = synthetic_context["pak"]
    with pak.open("rb") as handle:
        result = reference.inspect_pak(handle, pak.stat().st_size)
    assert result["magic"] == reference.PAK_MAGIC
    assert result["version"] == 11
    assert result["encrypted"] is False
    assert result["entry_count"] == 4
    assert result["candidate_package_paths"] == sorted(
        f"{stem}.uasset" for stem in reference.FROZEN_PACKAGE_STEMS
    )
    assert result["directory_index_paths"] == sorted(
        [
            "FlightSim/Content/Anduril-TrackEditor/Gates/BP_gate.uasset",
            "FlightSim/Content/Anduril-TrackEditor/Gates/"
            "SM_Gates_Anduril_Square_Combined.uasset",
            "FlightSim/Content/levels/MAP_arsenal_track01.uasset",
            "FlightSim/Content/levelsMaster/MAP_arsenal_master.uasset",
        ]
    )
    assert not any("geometry" in key or "transform" in key for key in result)


def test_synthetic_v11_fixture_has_frozen_golden_layout():
    pak, offsets = _synthetic_pak()
    assert len(pak) == 738
    assert hashlib.sha256(pak).hexdigest() == (
        "06b296db2668a970bd5ca1201347965686c2d7c9a97788400be89be3c64eab55"
    )
    assert offsets == {
        "path_offset": 64,
        "full_offset": 116,
        "index_offset": 387,
        "index_size": 130,
        "footer_offset": 517,
    }
    assert hashlib.sha1(pak[387:517]).hexdigest() == (
        "3f5443921bf703369fdc1753478bce2b65755a90"
    )
    assert hashlib.sha1(pak[64:116]).hexdigest() == (
        "c66509d40d0f7c0cdadccf0add41fca20c628b35"
    )
    assert hashlib.sha1(pak[116:387]).hexdigest() == (
        "fc0cedaf09fac9c08c6631d45d8c51ede56f2575"
    )


def test_discovery_does_not_emit_unrelated_directory_inventory(tmp_path):
    directories = [
        (
            "FlightSim/Content/Anduril-TrackEditor/Gates/",
            ["BP_gate.uasset", "SM_Gates_Anduril_Square_Combined.uasset"],
        ),
        ("FlightSim/Content/levels/", ["MAP_arsenal_track01.uasset"]),
        ("FlightSim/Content/levelsMaster/", ["MAP_arsenal_master.uasset"]),
        ("Private/Unrelated/", ["DoNotDisclose.uasset"]),
    ]
    pak_bytes, _ = _synthetic_pak(directories=directories)
    pak = _write(tmp_path / "inventory.pak", pak_bytes)
    with pak.open("rb") as handle:
        result = reference.inspect_pak(handle, pak.stat().st_size)
    assert result["entry_count"] == 5
    assert len(result["directory_index_paths"]) == 4
    assert all("DoNotDisclose" not in path for path in result["directory_index_paths"])


@pytest.mark.parametrize(
    ("path_locations", "directory_locations", "message"),
    [
        ([-1, 1, 2, 3], None, "path-hash entry location"),
        ([16, 1, 2, 3], None, "path-hash entry location"),
        (None, [-1, 1, 2, 3], "directory entry location"),
        (None, [16, 1, 2, 3], "directory entry location"),
        ([0, 1, 2, 3], [0, 1, 2, 4], "do not reconcile"),
    ],
)
def test_pak_encoded_entry_locations_fail_closed(
    tmp_path, path_locations, directory_locations, message
):
    bad, _ = _synthetic_pak(
        path_locations=path_locations,
        directory_locations=directory_locations,
    )
    path = _write(tmp_path / "bad-location.pak", bad)
    with path.open("rb") as handle:
        with pytest.raises(reference.ReferenceToolError, match=message):
            reference.inspect_pak(handle, path.stat().st_size)


def test_pak_requires_encoded_entries_when_index_is_nonempty(tmp_path):
    bad, _ = _synthetic_pak(encoded_entries=b"")
    path = _write(tmp_path / "empty-encoded-table.pak", bad)
    with path.open("rb") as handle:
        with pytest.raises(reference.ReferenceToolError, match="encoded-entry"):
            reference.inspect_pak(handle, path.stat().st_size)


@pytest.mark.parametrize("region", ["primary", "path", "directory"])
def test_every_pak_index_hash_is_verified(synthetic_context, region):
    data = bytearray(synthetic_context["pak_bytes"])
    offsets = synthetic_context["offsets"]
    target = {
        "primary": offsets["index_offset"],
        "path": offsets["path_offset"] + 4,
        "directory": offsets["full_offset"] + 4,
    }[region]
    data[target] ^= 1
    path = _write(synthetic_context["root"] / f"bad-{region}.pak", bytes(data))
    with path.open("rb") as handle:
        with pytest.raises(reference.ReferenceToolError, match="SHA-1 mismatch"):
            reference.inspect_pak(handle, path.stat().st_size)


@pytest.mark.parametrize(
    ("footer_relative", "replacement", "message"),
    [
        (16, b"\x01", "encrypted"),
        (17, struct.pack("<I", 0), "magic/version"),
        (21, struct.pack("<I", 10), "magic/version"),
        (25, struct.pack("<Q", 2**63), "signed int64"),
    ],
)
def test_pak_footer_fails_closed(
    synthetic_context, footer_relative, replacement, message
):
    data = bytearray(synthetic_context["pak_bytes"])
    start = synthetic_context["offsets"]["footer_offset"] + footer_relative
    data[start : start + len(replacement)] = replacement
    path = _write(
        synthetic_context["root"] / f"bad-footer-{footer_relative}.pak",
        bytes(data),
    )
    with path.open("rb") as handle:
        with pytest.raises(reference.ReferenceToolError, match=message):
            reference.inspect_pak(handle, path.stat().st_size)


def test_pak_rejects_directory_traversal_and_missing_candidate(tmp_path):
    bad, _ = _synthetic_pak(directories=[("../escape/", ["BP_gate.uasset"])])
    path = _write(tmp_path / "bad.pak", bad)
    with path.open("rb") as handle:
        with pytest.raises(reference.ReferenceToolError, match="traversal"):
            reference.inspect_pak(handle, path.stat().st_size)


@pytest.mark.parametrize("mount", ["../../../Other/", "../../../Bad//"])
def test_pak_mount_suffix_cannot_alias_frozen_paths(tmp_path, mount):
    bad, _ = _synthetic_pak(mount=mount)
    path = _write(tmp_path / "bad-mount.pak", bad)
    with path.open("rb") as handle:
        with pytest.raises(reference.ReferenceToolError):
            reference.inspect_pak(handle, path.stat().st_size)


def test_inspect_build_writes_canonical_unadmitted_report(synthetic_context):
    output = synthetic_context["root"] / "discovery.json"
    result = reference.run_inspect_build(
        rules_clearance=synthetic_context["clearance_path"], output=output
    )
    encoded = output.read_bytes()
    envelope = reference.strict_json_bytes(encoded)
    expected_payload_hash = hashlib.sha256(
        reference.canonical_json_bytes(envelope["payload"])
    ).hexdigest()
    assert envelope["payload_sha256"] == expected_payload_hash
    assert result["sha256"] == hashlib.sha256(encoded).hexdigest()
    assert envelope["payload"]["admitted"] is False
    assert envelope["payload"]["missing_r3_claims"] == list(
        reference.MISSING_R3_CLAIMS
    )
    assert envelope["payload"]["semantic_claims"] == "unmeasured"


def test_invalid_clearance_fails_before_any_build_source_read(tmp_path, monkeypatch):
    called_build = False
    original_identity = reference._identity_for

    def guarded(path, *, expected_hash=None):
        nonlocal called_build
        if Path(path) in {
            reference.LAUNCHER_PATH,
            reference.PAYLOAD_PATH,
            reference.PAK_PATH,
        }:
            called_build = True
            raise AssertionError("build source was touched")
        return original_identity(path, expected_hash=expected_hash)

    monkeypatch.setattr(reference, "_identity_for", guarded)
    with pytest.raises(reference.ReferenceToolError, match="must be absolute"):
        reference.run_inspect_build(
            rules_clearance=Path("relative.json"), output=tmp_path / "unused.json"
        )
    assert called_build is False


def test_wrong_interpreter_fails_before_any_build_source_read(
    synthetic_context, monkeypatch
):
    calls = []
    original_identity = reference._identity_for

    def rejected_identity(path, *, expected_hash=None):
        resolved = Path(path).resolve()
        calls.append(resolved)
        if resolved == Path(reference.sys.executable).resolve():
            raise reference.ReferenceToolError("interpreter hash mismatch")
        return original_identity(path, expected_hash=expected_hash)

    monkeypatch.setattr(reference, "_identity_for", rejected_identity)
    with pytest.raises(reference.ReferenceToolError, match="interpreter"):
        reference.run_inspect_build(
            rules_clearance=synthetic_context["clearance_path"],
            output=synthetic_context["root"] / "unused.json",
        )
    assert calls == [
        Path(reference.__file__).resolve(),
        Path(reference.sys.executable).resolve(),
    ]


def test_candidate_validation_is_structural_only_and_never_admits(synthetic_context):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    checks = reference.validate_candidate(
        candidate,
        clearance_path=synthetic_context["clearance_path"].resolve(),
        clearance=synthetic_context["clearance"],
        clearance_bytes=synthetic_context["clearance_bytes"],
    )
    assert all(checks.values())
    assert "admitted" not in checks


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("unknown", True),
        lambda value: value["geometry"].__setitem__("active_lod", True),
        lambda value: value["geometry"]["features"][2].__setitem__(
            "references", ["missing"]
        ),
        lambda value: value["transform_chain"]["links"][0].__setitem__(
            "matrix_row_major",
            [
                1.0,
                0.2,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ),
        lambda value: value["independent_checks"][1].__setitem__(
            "producer_id", value["independent_checks"][0]["producer_id"]
        ),
        lambda value: value["annotation_contract"].__setitem__(
            "checker_id", value["annotation_contract"]["producer_id"]
        ),
        lambda value: value["transform_chain"]["active_actor_overrides"][
            0
        ].__setitem__(
            "id", value["transform_chain"]["links"][0]["id"]
        ),
        lambda value: value["build"]["package_paths"].append("ZZZ.uasset"),
    ],
)
def test_candidate_rejects_unknown_bool_reference_shear_or_fake_independence(
    synthetic_context, mutate
):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    mutate(candidate)
    with pytest.raises(reference.ReferenceToolError):
        reference.validate_candidate(
            candidate,
            clearance_path=synthetic_context["clearance_path"].resolve(),
            clearance=synthetic_context["clearance"],
            clearance_bytes=synthetic_context["clearance_bytes"],
        )


def test_transform_checks_are_relative_at_tiny_scale(synthetic_context):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    link = candidate["transform_chain"]["links"][0]
    link["matrix_row_major"] = [
        1e-12,
        0.0,
        0.0,
        0.0,
        0.0,
        2e-12,
        0.0,
        0.0,
        0.0,
        0.0,
        3e-12,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    link["determinant"] = 6e-36
    link["singular_values"] = [1e-12, 2e-12, 3e-12]
    with pytest.raises(reference.ReferenceToolError, match="uniform"):
        reference.validate_candidate(
            candidate,
            clearance_path=synthetic_context["clearance_path"].resolve(),
            clearance=synthetic_context["clearance"],
            clearance_bytes=synthetic_context["clearance_bytes"],
        )


def test_transform_rejects_computed_overflow(synthetic_context):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    link = candidate["transform_chain"]["links"][0]
    scale = 1e154
    link["matrix_row_major"] = [
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    link["determinant"] = 1e308
    link["singular_values"] = [scale, scale, scale]
    with pytest.raises(reference.ReferenceToolError, match="determinant"):
        reference.validate_candidate(
            candidate,
            clearance_path=synthetic_context["clearance_path"].resolve(),
            clearance=synthetic_context["clearance"],
            clearance_bytes=synthetic_context["clearance_bytes"],
        )


@pytest.mark.parametrize(
    ("delta", "accepted"),
    [(1e-6, True), (2e-6, False)],
)
def test_transform_relative_roundoff_boundary(
    synthetic_context, delta, accepted
):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    link = candidate["transform_chain"]["links"][0]
    link["matrix_row_major"][10] = 1.0 + delta
    link["determinant"] = 1.0 + delta
    link["singular_values"] = [1.0, 1.0, 1.0 + delta]
    arguments = {
        "clearance_path": synthetic_context["clearance_path"].resolve(),
        "clearance": synthetic_context["clearance"],
        "clearance_bytes": synthetic_context["clearance_bytes"],
    }
    if accepted:
        reference.validate_candidate(candidate, **arguments)
    else:
        with pytest.raises(reference.ReferenceToolError, match="uniform"):
            reference.validate_candidate(candidate, **arguments)


@pytest.mark.parametrize(
    ("normalized_dot", "accepted"),
    [(0.5e-6, True), (2e-6, False)],
)
def test_transform_orthogonality_roundoff_boundary(
    synthetic_context, normalized_dot, accepted
):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    perpendicular = (1.0 - normalized_dot**2) ** 0.5
    link = candidate["transform_chain"]["links"][0]
    link["matrix_row_major"][1] = normalized_dot
    link["matrix_row_major"][5] = perpendicular
    link["determinant"] = perpendicular
    arguments = {
        "clearance_path": synthetic_context["clearance_path"].resolve(),
        "clearance": synthetic_context["clearance"],
        "clearance_bytes": synthetic_context["clearance_bytes"],
    }
    if accepted:
        reference.validate_candidate(candidate, **arguments)
    else:
        with pytest.raises(reference.ReferenceToolError, match="shear"):
            reference.validate_candidate(candidate, **arguments)


def test_parser_config_hash_binds_both_roundoff_constants(monkeypatch):
    original_relative = reference.TRANSFORM_RELATIVE_ROUNDOFF
    original_orthogonality = reference.TRANSFORM_ORTHOGONALITY_ROUNDOFF
    baseline = reference.parser_config_sha256()
    monkeypatch.setattr(
        reference, "TRANSFORM_RELATIVE_ROUNDOFF", original_relative * 2
    )
    relative_changed = reference.parser_config_sha256()
    monkeypatch.setattr(
        reference, "TRANSFORM_RELATIVE_ROUNDOFF", original_relative
    )
    monkeypatch.setattr(
        reference,
        "TRANSFORM_ORTHOGONALITY_ROUNDOFF",
        original_orthogonality * 2,
    )
    orthogonality_changed = reference.parser_config_sha256()
    assert len({baseline, relative_changed, orthogonality_changed}) == 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["build"].__setitem__("build", 3385.0),
        lambda value: value["build"]["pak_index"].__setitem__(
            "magic", float(reference.PAK_MAGIC)
        ),
        lambda value: value["build"]["pak_index"].__setitem__(
            "version", 11.0
        ),
        lambda value: value["build"]["pak_index"].__setitem__(
            "path_hash_seed", 2**1000
        ),
        lambda value: value["geometry"]["features"][0]["coordinates"].__setitem__(
            0, [10**1000, 0, 0]
        ),
    ],
)
def test_candidate_rejects_float_integers_or_numeric_overflow(
    synthetic_context, mutate
):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    mutate(candidate)
    with pytest.raises(reference.ReferenceToolError):
        reference.validate_candidate(
            candidate,
            clearance_path=synthetic_context["clearance_path"].resolve(),
            clearance=synthetic_context["clearance"],
            clearance_bytes=synthetic_context["clearance_bytes"],
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["build"]["pak_index"].__setitem__("index_size", 0),
        lambda value: value["build"]["pak_index"].__setitem__(
            "index_offset", 2**63
        ),
        lambda value: value["build"]["pak_index"].__setitem__(
            "path_hash_index_offset", value["build"]["pak_index"]["index_offset"]
        ),
        lambda value: value["build"]["pak_index"].__setitem__(
            "index_offset", value["build"]["pak"]["size_bytes"]
        ),
    ],
)
def test_candidate_rejects_impossible_pak_index_geometry(synthetic_context, mutate):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    mutate(candidate)
    with pytest.raises(reference.ReferenceToolError):
        reference.validate_candidate(
            candidate,
            clearance_path=synthetic_context["clearance_path"].resolve(),
            clearance=synthetic_context["clearance"],
            clearance_bytes=synthetic_context["clearance_bytes"],
        )


def test_candidate_rejects_noncanonical_mount(synthetic_context):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    candidate["build"]["pak_index"]["mount_point"] = ".././"
    with pytest.raises(reference.ReferenceToolError, match="mount"):
        reference.validate_candidate(
            candidate,
            clearance_path=synthetic_context["clearance_path"].resolve(),
            clearance=synthetic_context["clearance"],
            clearance_bytes=synthetic_context["clearance_bytes"],
        )


def test_annotation_checker_requires_distinct_code_hash(synthetic_context):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    annotation = candidate["annotation_contract"]
    annotation["checker_sha256"] = annotation["producer_sha256"]
    with pytest.raises(reference.ReferenceToolError, match="code must differ"):
        reference.validate_candidate(
            candidate,
            clearance_path=synthetic_context["clearance_path"].resolve(),
            clearance=synthetic_context["clearance"],
            clearance_bytes=synthetic_context["clearance_bytes"],
        )


def test_validate_command_report_stays_unadmitted(synthetic_context):
    candidate = _candidate_value(
        launcher=synthetic_context["launcher"],
        payload=synthetic_context["payload"],
        pak=synthetic_context["pak"],
        clearance_path=synthetic_context["clearance_path"],
        clearance_bytes=synthetic_context["clearance_bytes"],
        pak_index=synthetic_context["pak_index"],
    )
    candidate_path = _write(
        synthetic_context["root"] / "candidate.json",
        reference.canonical_json_bytes(candidate),
    )
    output = synthetic_context["root"] / "validation.json"
    reference.run_validate_candidate(
        rules_clearance=synthetic_context["clearance_path"],
        candidate=candidate_path,
        output=output,
    )
    report = reference.strict_json_bytes(output.read_bytes())["payload"]
    assert report["structurally_valid"] is True
    assert report["admitted"] is False
    assert report["independent_review_required"] is True
    assert report["rules_authority"] == "unverified-by-tool"


def test_output_is_beneath_root_nonexistent_and_no_overwrite(
    synthetic_context, monkeypatch
):
    existing = _write(synthetic_context["root"] / "existing.json", b"original")
    with pytest.raises(reference.ReferenceToolError, match="already exists"):
        reference._write_envelope(existing, "schema/1", {"ok": True})
    assert existing.read_bytes() == b"original"
    outside = synthetic_context["root"].parent / "outside.json"
    with pytest.raises(reference.ReferenceToolError, match="outside"):
        reference._write_envelope(outside, "schema/1", {"ok": True})
    if reference.os.name == "nt":
        monkeypatch.setattr(reference, "_windows_private_acl", lambda _path: False)
        with pytest.raises(reference.ReferenceToolError, match="current-user-only"):
            reference._write_envelope(
                synthetic_context["root"] / "acl-rejected.json",
                "schema/1",
                {"ok": True},
            )


def test_output_race_does_not_delete_competing_file(synthetic_context):
    output = synthetic_context["root"] / "raced.json"
    sentinel = b"created by another writer"
    output.write_bytes(sentinel)
    with pytest.raises(FileExistsError):
        if reference.os.name == "nt":
            reference._windows_write_new(output, b"ours")
        else:
            output.open("xb")
    assert output.read_bytes() == sentinel


@pytest.mark.skipif(reference.os.name != "nt", reason="Windows ADS syntax")
def test_output_rejects_windows_alternate_data_stream(synthetic_context):
    output = synthetic_context["root"] / "artifact.json:stream"
    with pytest.raises(reference.ReferenceToolError, match="Windows path alias"):
        reference._write_envelope(output, "schema/1", {"ok": True})


@pytest.mark.skipif(reference.os.name != "nt", reason="Windows device aliases")
@pytest.mark.parametrize(
    "name",
    ["CONIN$", "CONOUT$", "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³"],
)
def test_windows_extended_device_aliases_are_rejected(tmp_path, name):
    candidate = tmp_path.resolve() / name / "artifact.json"
    with pytest.raises(reference.ReferenceToolError, match="Windows path alias"):
        reference._absolute_lexical(candidate, label="synthetic")


@pytest.mark.skipif(reference.os.name != "nt", reason="Windows path namespaces")
def test_windows_network_paths_are_rejected_before_io(monkeypatch):
    with pytest.raises(reference.ReferenceToolError, match="network path"):
        reference._absolute_lexical(
            Path(r"\\server\share\clearance.json"), label="synthetic"
        )
    for drive_type in (0, 1, 2, 4, 5, 6):
        monkeypatch.setattr(
            reference, "_windows_drive_type", lambda _root, value=drive_type: value
        )
        with pytest.raises(reference.ReferenceToolError, match="fixed local drive"):
            reference._absolute_lexical(
                Path(r"Z:\clearance.json"), label="synthetic"
            )


@pytest.mark.skipif(reference.os.name != "nt", reason="Windows handle locking")
def test_windows_directory_handle_detects_parent_rename(synthetic_context):
    parent = synthetic_context["root"]
    moved = parent.with_name(f"{parent.name}-moved")
    locked = reference._windows_lock_components(parent)
    try:
        parent.rename(moved)
        with pytest.raises(reference.ReferenceToolError, match="moved"):
            reference._windows_validate_locks(locked)
    finally:
        reference._windows_close_locks(locked)
        if moved.exists():
            moved.rename(parent)


@pytest.mark.skipif(reference.os.name != "nt", reason="Windows handle ACL")
def test_windows_private_acl_is_checked_from_held_handle():
    repository = Path(reference.__file__).resolve().parents[1]
    locked = reference._windows_lock_components(repository)
    try:
        assert _NATIVE_WINDOWS_PRIVATE_ACL_HANDLE(locked[-1][0]) is True
    finally:
        reference._windows_close_locks(locked)


@pytest.mark.skipif(reference.os.name != "nt", reason="Windows handle access")
def test_windows_read_lock_can_omit_read_control(monkeypatch):
    class FakeKernel:
        def __init__(self):
            self.desired_access = None

        def CreateFileW(
            self,
            _path,
            desired_access,
            _share,
            _security,
            _disposition,
            _flags,
            _template,
        ):
            self.desired_access = desired_access
            return 123

        def CloseHandle(self, _handle):
            return 1

    kernel = FakeKernel()
    monkeypatch.setattr(reference, "_windows_file_api", lambda: kernel)
    monkeypatch.setattr(
        reference,
        "_windows_file_identity",
        lambda _handle, *, directory: (1, 2, 3),
    )
    handle, _identity = reference._windows_open_locked(
        Path(r"C:\synthetic.bin"), directory=False, read_control=False
    )
    assert handle == 123
    assert kernel.desired_access == 0x00000080


@pytest.mark.skipif(reference.os.name != "nt", reason="Windows handle cleanup")
def test_windows_component_final_path_failure_closes_current_handle(monkeypatch):
    closed = []
    monkeypatch.setattr(
        reference,
        "_absolute_lexical",
        lambda _path, *, label: Path(r"C:\expected"),
    )
    monkeypatch.setattr(
        reference,
        "_windows_open_locked",
        lambda _path, **_kwargs: (123, (1, 2, 3)),
    )
    monkeypatch.setattr(
        reference, "_windows_final_path", lambda _handle: Path(r"C:\wrong")
    )
    monkeypatch.setattr(
        reference, "_windows_close_locks", lambda locks: closed.extend(locks)
    )
    with pytest.raises(reference.ReferenceToolError, match="final path"):
        reference._windows_lock_components(Path(r"C:\expected"))
    assert [item[0] for item in closed] == [123]


@pytest.mark.skipif(reference.os.name != "nt", reason="Windows handle cleanup")
def test_windows_file_final_path_failure_closes_current_handle(monkeypatch):
    closed = []
    monkeypatch.setattr(reference, "_windows_lock_components", lambda _path, **_kw: [])
    monkeypatch.setattr(
        reference,
        "_windows_open_locked",
        lambda _path, **_kwargs: (456, (1, 2, 3)),
    )
    monkeypatch.setattr(
        reference, "_windows_final_path", lambda _handle: Path(r"C:\wrong")
    )
    monkeypatch.setattr(
        reference, "_windows_close_locks", lambda locks: closed.extend(locks)
    )
    with pytest.raises(reference.ReferenceToolError, match="final path"):
        reference._windows_lock_file(Path(r"C:\expected.bin"))
    assert [item[0] for item in closed] == [456]


@pytest.mark.skipif(reference.os.name != "nt", reason="Windows reparse points")
def test_windows_reparse_ancestor_is_rejected_before_input_open(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    _write(target / "input.json", b"{}")
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(reference.ReferenceToolError, match="reparse"):
        reference._secure_input(link / "input.json", label="synthetic input")


def test_stable_identity_hashes_and_parses_one_unchanged_handle(tmp_path, monkeypatch):
    target = _write(tmp_path / "source.bin", b"first-content")
    handles = []
    original_hash = reference._hash_handle

    def observed_hash(handle):
        handles.append(handle.fileno())
        return original_hash(handle)

    def observed_inspect(handle, size):
        handles.append(handle.fileno())
        assert size == len(b"first-content")
        return {"seen": handle.read(5)}

    monkeypatch.setattr(reference, "_hash_handle", observed_hash)
    identity, inspected = reference._stable_identity(
        target, label="synthetic source", inspect=observed_inspect
    )
    assert len(set(handles)) == 1
    assert identity["sha256"] == hashlib.sha256(b"first-content").hexdigest()
    assert inspected == {"seen": b"first"}


def test_stable_identity_rejects_in_place_mutation(tmp_path, monkeypatch):
    target = _write(tmp_path / "source.bin", b"first-content")
    original_hash = reference._hash_handle
    calls = 0

    def mutate_after_first_hash(handle):
        nonlocal calls
        calls += 1
        digest = original_hash(handle)
        if calls == 1:
            target.write_bytes(b"other-content")
        return digest

    monkeypatch.setattr(reference, "_hash_handle", mutate_after_first_hash)
    with pytest.raises(reference.ReferenceToolError, match="changed"):
        reference._stable_identity(target, label="synthetic source")


def test_wrong_source_hash_fails_before_inspection(tmp_path):
    target = _write(tmp_path / "wrong-source.bin", b"wrong")
    inspected = False

    def forbidden_inspect(_handle, _size):
        nonlocal inspected
        inspected = True
        raise AssertionError("wrong-hash source was inspected")

    with pytest.raises(reference.ReferenceToolError, match="hash mismatch"):
        reference._stable_identity(
            target,
            label="synthetic source",
            inspect=forbidden_inspect,
            expected_hash="0" * 64,
        )
    assert inspected is False


def test_cli_has_only_two_nonabbreviated_commands():
    parser = reference._build_parser()
    assert parser.parse_args(
        ["inspect-build", "--rules-clearance", "x", "--output", "y"]
    ).command == "inspect-build"
    with pytest.raises(SystemExit):
        parser.parse_args(["inspect", "--rules-clearance", "x", "--output", "y"])
    with pytest.raises(SystemExit):
        parser.parse_args(["inspect-build", "--rules", "x", "--output", "y"])


def test_module_imports_are_standard_library_only():
    tree = ast.parse(Path(reference.__file__).read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "ctypes",
        "hashlib",
        "json",
        "math",
        "os",
        "stat",
        "struct",
        "sys",
        "pathlib",
        "typing",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed


def test_r1_fixtures_never_contain_or_open_the_real_pak(
    synthetic_context, monkeypatch
):
    real_pak = Path(
        r"C:\Users\John\AIGP\AIGP_3385\FlightSim\Content\Paks"
        r"\FlightSim-WindowsNoEditor.pak"
    ).resolve()
    assert synthetic_context["pak"].resolve() != real_pak
    assert synthetic_context["pak"].is_relative_to(synthetic_context["root"])
    original_open = reference.os.open
    opened = []

    def guarded_open(path, flags, *args, **kwargs):
        resolved = Path(path).resolve()
        assert resolved != real_pak
        opened.append(resolved)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(reference.os, "open", guarded_open)
    output = synthetic_context["root"] / "guarded-discovery.json"
    reference.run_inspect_build(
        rules_clearance=synthetic_context["clearance_path"], output=output
    )
    assert synthetic_context["pak"].resolve() in opened
    assert real_pak not in opened
