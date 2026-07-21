import ast
import hashlib
import json
from pathlib import Path

import pytest

import scripts.aigp_vq2_calibration_target as target


ROOT = Path(__file__).resolve().parents[1]
TRACKED_CONFIG = ROOT / "config" / "aigp_vq2_calibration_target_build3385.json"
TRACKED_CONFIG_SHA256 = (
    "e16e2a70e6be8d6d083e5739773473090c62d244a1b69f120ce027f51b84f82b"
)


def _target_value():
    return json.loads(TRACKED_CONFIG.read_text(encoding="utf-8"))


def _authorization_value():
    return {
        "schema": "aigp-vq2-simulation-capture-authorization/1",
        "authority": {
            "kind": "user_operator",
            "authority_id": "conversation-2026-07-21-package2-f01-sim-capture",
            "authorized_on": "2026-07-21",
            "source": "direct_user_instruction",
        },
        "task_id": "vq2-package2-import-environment-recovery",
        "domain": "simulator_only",
        "simulator": {"build": 3385, "mode": "Training"},
        "session_ids": ["F01"],
        "allowed_purposes": [
            "calibration_discovery",
            "independent_review",
            "integrity_audit",
            "offline_replay_and_analysis",
        ],
        "allowed_classes": [
            "annotations_crops_features",
            "commands",
            "decoded_frames",
            "derived_replay_and_analysis",
            "highres_imu",
            "process_lease_cleanup",
            "race_heartbeat_actuator_collision",
            "reconstructed_jpegs",
            "source_and_host_timestamps",
            "udp_camera_datagrams",
        ],
        "storage": {
            "private_root": (
                r"C:\Users\John\aigp-evidence"
                r"\2026-07-21-package2-import-environment-recovery"
            ),
            "git": False,
            "public_release": False,
            "network_export": False,
            "external_service_upload": False,
        },
        "retention": {
            "through": "package2_simulator_audit_closeout",
            "after": "sealed_quarantine_pending_explicit_disposition",
            "automatic_deletion": False,
        },
        "transfer": {
            "successor_task": False,
            "new_session": False,
            "new_build_or_mode": False,
            "submitted_run": False,
            "physical_or_hil": False,
        },
        "organizer_media_credential": False,
        "publication_permitted": False,
    }


def _write_json(path, value, *, indent=None):
    path.write_text(
        json.dumps(value, indent=indent, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _documents(tmp_path, *, config=None, authorization=None):
    config_path = _write_json(
        tmp_path / "target.json",
        _target_value() if config is None else config,
        indent=2,
    )
    authorization_path = _write_json(
        tmp_path / "authorization.json",
        _authorization_value() if authorization is None else authorization,
        indent=2,
    )
    return config_path, authorization_path


def test_tracked_config_matches_frozen_exact_byte_identity():
    document = target.read_target_config(
        TRACKED_CONFIG, expected_sha256=TRACKED_CONFIG_SHA256
    )

    assert document.path == TRACKED_CONFIG.resolve()
    assert document.size_bytes == len(TRACKED_CONFIG.read_bytes())
    assert document.sha256 == TRACKED_CONFIG_SHA256
    assert document.value["config_id"] == "vq2-build3385-training-gate0-nominal-v1"
    assert document.value["geometry"]["outer"]["width"] == 2.7
    with pytest.raises(TypeError):
        document.value["revision"] = "changed"


def test_valid_authorization_and_cross_document_scope(tmp_path):
    config_path, authorization_path = _documents(tmp_path)
    config = target.read_target_config(config_path)
    authorization = target.read_capture_authorization(authorization_path)

    target.validate_collection_protocol(config, authorization)

    assert authorization.value["session_ids"] == ("F01",)
    assert authorization.value["domain"] == "simulator_only"
    assert config.value["data_scope"]["private_simulation_capture"] is True


def test_hash_is_over_exact_stable_bytes_and_can_be_pinned(tmp_path):
    compact = _write_json(tmp_path / "compact.json", _authorization_value())
    pretty = _write_json(tmp_path / "pretty.json", _authorization_value(), indent=2)

    compact_document = target.read_capture_authorization(compact)
    pretty_document = target.read_capture_authorization(pretty)

    assert compact_document.sha256 == hashlib.sha256(compact.read_bytes()).hexdigest()
    assert pretty_document.sha256 == hashlib.sha256(pretty.read_bytes()).hexdigest()
    assert compact_document.sha256 != pretty_document.sha256
    assert target.read_capture_authorization(
        compact, expected_sha256=compact_document.sha256
    ).sha256 == compact_document.sha256


@pytest.mark.parametrize(
    "expected",
    ["0" * 64, "E" * 64, "e" * 63, "g" * 64, True],
)
def test_expected_hash_must_be_canonical_and_match(tmp_path, expected):
    path = _write_json(tmp_path / "authorization.json", _authorization_value())

    with pytest.raises(target.CalibrationTargetError):
        target.read_capture_authorization(path, expected_sha256=expected)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema":1,"schema":1}',
        b"\xef\xbb\xbf{}",
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":1e999}',
        b"\xff",
        b"{",
    ],
)
def test_strict_json_rejects_ambiguous_or_invalid_bytes(payload):
    with pytest.raises(target.CalibrationTargetError):
        target.strict_json_bytes(payload)


def test_strict_json_resource_limit():
    with pytest.raises(target.CalibrationTargetError, match="resource limit"):
        target.strict_json_bytes(b" " * (target.MAX_JSON_BYTES + 1))


@pytest.mark.parametrize("key", sorted(_target_value().keys()))
def test_target_rejects_every_missing_top_level_key(key):
    value = _target_value()
    del value[key]

    with pytest.raises(target.CalibrationTargetError, match="keys mismatch"):
        target.validate_target_config(value)


def test_target_rejects_unknown_and_pak_identity_fields():
    value = _target_value()
    value["pak_sha256"] = "0" * 64

    with pytest.raises(target.CalibrationTargetError, match="unknown"):
        target.validate_target_config(value)

    value = _target_value()
    value["data_scope"]["pak_access"] = True
    with pytest.raises(target.CalibrationTargetError, match="pak_access"):
        target.validate_target_config(value)


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("simulator", "build", 3384),
        ("simulator", "build", 3385.0),
        ("simulator", "mode", "Race"),
        ("calibration_status", "intrinsics", "known"),
        ("calibration_status", "camera_imu_time_model", "identity"),
        ("data_scope", "physical_or_hil_use", True),
        ("data_scope", "submitted_run_use", True),
        ("data_scope", "public_release", True),
        ("data_scope", "external_service_upload", True),
        ("data_scope", "git_storage", True),
    ],
)
def test_target_rejects_wrong_scope_or_inferred_calibration(
    section, key, replacement
):
    value = _target_value()
    value[section][key] = replacement

    with pytest.raises(target.CalibrationTargetError):
        target.validate_target_config(value)


def test_target_rejects_camera_default_and_pose_truth_fields():
    value = _target_value()
    value["observed_streams"]["camera"]["fx"] = 320.0
    with pytest.raises(target.CalibrationTargetError, match="unknown"):
        target.validate_target_config(value)

    value = _target_value()
    value["observed_streams"]["pose"] = {"message": "LOCAL_POSITION_NED"}
    with pytest.raises(target.CalibrationTargetError, match="unknown"):
        target.validate_target_config(value)

    value = _target_value()
    value["observed_streams"]["unavailable_as_truth"].remove("ODOMETRY")
    with pytest.raises(target.CalibrationTargetError, match="frozen array"):
        target.validate_target_config(value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["geometry"]["outer"].__setitem__("width", True),
        lambda value: value["geometry"]["inner"].__setitem__("height", 0.0),
        lambda value: value["geometry"].__setitem__("depth", float("inf")),
        lambda value: value["geometry"]["inner"].__setitem__("width", 2.7),
        lambda value: value["geometry"]["feature"]["order"].reverse(),
        lambda value: value["geometry"]["feature"]["coordinates"][0].__setitem__(
            0, -0.74
        ),
        lambda value: value["geometry"]["feature"]["coordinates"].append(
            [0.0, 0.0, 0.0]
        ),
    ],
)
def test_target_rejects_invalid_geometry(mutation):
    value = _target_value()
    mutation(value)

    with pytest.raises(target.CalibrationTargetError):
        target.validate_target_config(value)


def test_geometry_is_configurable_with_matching_features_and_new_identity(tmp_path):
    previous_path = _write_json(tmp_path / "previous.json", _target_value(), indent=2)
    value = _target_value()
    value["config_id"] = "vq2-build3385-training-gate0-nominal-v2"
    value["revision"] = "2"
    value["geometry"]["outer"] = {"width": 3.0, "height": 2.8}
    value["geometry"]["inner"] = {"width": 1.6, "height": 1.4}
    value["geometry"]["depth"] = 0.3
    value["geometry"]["feature"]["coordinates"] = [
        [-0.8, -0.7, 0.0],
        [0.8, -0.7, 0.0],
        [0.8, 0.7, 0.0],
        [-0.8, 0.7, 0.0],
    ]
    replacement_path = _write_json(tmp_path / "replacement.json", value, indent=2)

    previous = target.read_target_config(previous_path)
    replacement = target.read_target_config(replacement_path)
    target.validate_config_replacement(previous, replacement)

    assert replacement.sha256 != previous.sha256


@pytest.mark.parametrize(("new_id", "new_revision"), [(None, "2"), ("new", None)])
def test_changed_config_requires_new_id_revision_and_hash(
    tmp_path, new_id, new_revision
):
    previous_path = _write_json(tmp_path / "previous.json", _target_value(), indent=2)
    value = _target_value()
    if new_id is not None:
        value["config_id"] = new_id
    if new_revision is not None:
        value["revision"] = new_revision
    replacement_path = _write_json(tmp_path / "replacement.json", value)

    previous = target.read_target_config(previous_path)
    replacement = target.read_target_config(replacement_path)
    with pytest.raises(target.CalibrationTargetError, match="new (config_id|revision)"):
        target.validate_config_replacement(previous, replacement)


def test_identical_config_identity_is_not_a_replacement(tmp_path):
    path = _write_json(tmp_path / "target.json", _target_value(), indent=2)
    document = target.read_target_config(path)

    target.validate_config_replacement(document, document)


@pytest.mark.parametrize("key", sorted(_authorization_value().keys()))
def test_authorization_rejects_every_missing_top_level_key(key):
    value = _authorization_value()
    del value[key]

    with pytest.raises(target.CalibrationTargetError, match="keys mismatch"):
        target.validate_capture_authorization(value)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            ("authority", "authority_id"),
            "conversation-2026-07-20-package2-sim-capture",
        ),
        (("authority", "authorized_on"), "2026-07-20"),
        (("task_id",), "vq2-package2-powered-calibration-pilot"),
        (("domain",), "physical"),
        (("simulator", "build"), 3384),
        (("simulator", "mode"), "Race"),
        (("session_ids",), ["F00"]),
        (("organizer_media_credential",), True),
        (("publication_permitted",), True),
        (("storage", "git"), True),
        (("storage", "public_release"), True),
        (("storage", "network_export"), True),
        (("storage", "external_service_upload"), True),
        (
            ("storage", "private_root"),
            r"C:\Users\John\aigp-evidence"
            r"\2026-07-20-package2-powered-calibration-pilot",
        ),
        (("transfer", "successor_task"), True),
        (("transfer", "new_session"), True),
        (("transfer", "new_build_or_mode"), True),
        (("transfer", "submitted_run"), True),
        (("transfer", "physical_or_hil"), True),
    ],
)
def test_authorization_rejects_wrong_or_transferable_scope(path, replacement):
    value = _authorization_value()
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement

    with pytest.raises(target.CalibrationTargetError):
        target.validate_capture_authorization(value)


@pytest.mark.parametrize("field", ["allowed_purposes", "allowed_classes"])
def test_authorization_rejects_unsorted_or_duplicate_set_like_arrays(field):
    value = _authorization_value()
    value[field] = list(reversed(value[field]))
    with pytest.raises(target.CalibrationTargetError):
        target.validate_capture_authorization(value)

    value = _authorization_value()
    value[field].append(value[field][-1])
    with pytest.raises(target.CalibrationTargetError):
        target.validate_capture_authorization(value)


def test_authorization_rejects_unknown_or_incomplete_nested_fields():
    value = _authorization_value()
    value["storage"]["cloud_bucket"] = "example"
    with pytest.raises(target.CalibrationTargetError, match="unknown"):
        target.validate_capture_authorization(value)

    value = _authorization_value()
    del value["retention"]["after"]
    with pytest.raises(target.CalibrationTargetError, match="missing"):
        target.validate_capture_authorization(value)


def test_nonregular_input_is_rejected(tmp_path):
    with pytest.raises(target.CalibrationTargetError, match="regular"):
        target.read_target_config(tmp_path)


def test_cli_requires_both_explicit_paths_and_has_no_field_overrides(tmp_path):
    config_path, authorization_path = _documents(tmp_path)

    with pytest.raises(SystemExit) as missing:
        target.main([])
    assert missing.value.code == 2

    with pytest.raises(SystemExit) as override:
        target.main(
            [
                "--target-config",
                str(config_path),
                "--capture-authorization",
                str(authorization_path),
                "--outer-width",
                "3.0",
            ]
        )
    assert override.value.code == 2


def test_cli_emits_only_exact_document_hashes(tmp_path, capsys):
    config_path, authorization_path = _documents(tmp_path)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    authorization_hash = hashlib.sha256(authorization_path.read_bytes()).hexdigest()

    result = target.main(
        [
            "--target-config",
            str(config_path),
            "--capture-authorization",
            str(authorization_path),
            "--target-config-sha256",
            config_hash,
            "--capture-authorization-sha256",
            authorization_hash,
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "capture_authorization_sha256": authorization_hash,
        "target_config_sha256": config_hash,
    }


def test_cli_hash_failure_is_fail_closed(tmp_path, capsys):
    config_path, authorization_path = _documents(tmp_path)

    result = target.main(
        [
            "--target-config",
            str(config_path),
            "--capture-authorization",
            str(authorization_path),
            "--target-config-sha256",
            "0" * 64,
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "SHA-256 mismatch" in captured.err


def test_validator_has_no_online_powered_or_writeback_imports():
    source_path = Path(target.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert imports.isdisjoint(
        {"socket", "subprocess", "requests", "urllib", "http", "pymavlink"}
    )
    assert "aigp_vq2_build_reference" not in source_path.read_text(encoding="utf-8")
