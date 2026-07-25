from __future__ import annotations

import copy
import json

import pytest

from scripts import aigp_vq2_yaw_profile as profile_module


def _profile() -> dict:
    return profile_module.load_yaw_calibration_profile()


def _set_path(document: dict, path: tuple[object, ...], value: object) -> None:
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_default_profile_loads_with_exact_identity_and_authority():
    profile = _profile()

    assert profile["schema"] == "aigp-vq2-yaw-calibration-profile/1"
    assert profile["profile_id"] == (
        "vq2-build3385-training-yaw-authority-v1"
    )
    assert profile["simulator"] == {
        "build": 3385,
        "mode": "Training",
        "mode_basis": "configured_session_not_machine_readable",
    }
    assert profile["source"] == {
        "commit": "f4eecd6afbca6ff69cf84b0a69339ed66238cfa0",
        "stage": "calibration-excite",
        "worktree_state": "clean",
    }
    assert profile["plan"] == {
        "plan_id": "vq2-build3385-training-yaw-calibration-v1",
        "sha256": (
            "827101741ddb335a1cbfcdbdcfca65c2f7579ad4a435c140179b8d9d0eb2be1b"
        ),
    }
    assert profile["authority"] == {
        "controller_to_body_sign": 1,
        "controller_to_image_sign": 1,
        "max_abs_yaw_rate_command_rad_s": 0.08,
        "max_gyro_response_delay_s": 0.08,
        "max_first_image_observation_delay_s": 0.09,
        "max_attitude_excursion_rad": 0.05,
        "max_abs_measured_yaw_rate_rad_s": 0.5,
        "control_hold_horizon_s": 0.12,
    }


def test_profile_hash_is_frozen_and_independent_of_key_order():
    profile = _profile()
    assert profile_module.YAW_CALIBRATION_PROFILE_SHA256 == (
        "9497417108749d9ccf395a042a450297d3f8643bd0acb76178171fcc02ec3dd5"
    )
    assert profile_module.canonical_yaw_calibration_profile_sha256(profile) == (
        profile_module.YAW_CALIBRATION_PROFILE_SHA256
    )

    reordered = json.loads(
        json.dumps(profile, sort_keys=True),
        object_pairs_hook=lambda pairs: dict(reversed(pairs)),
    )
    assert profile_module.canonical_yaw_calibration_profile_sha256(
        reordered
    ) == profile_module.YAW_CALIBRATION_PROFILE_SHA256


def test_runtime_evidence_is_small_exact_and_defensive():
    profile = _profile()
    evidence = profile_module.yaw_calibration_profile_evidence(profile)

    assert evidence == {
        "profile_id": profile_module.YAW_CALIBRATION_PROFILE_ID,
        "sha256": profile_module.YAW_CALIBRATION_PROFILE_SHA256,
        "source_commit": profile_module.YAW_CALIBRATION_SOURCE_COMMIT,
        "plan_id": profile["plan"]["plan_id"],
        "plan_sha256": profile["plan"]["sha256"],
        "authority": profile["authority"],
    }
    evidence["authority"]["controller_to_body_sign"] = -1
    assert profile["authority"]["controller_to_body_sign"] == 1


def test_three_unique_clean_rows_bind_all_compact_artifact_hashes():
    evidence = _profile()["evidence"]

    assert len(evidence) == 3
    assert len({row["run_id"] for row in evidence}) == 3
    assert all(row["stage_success"] is True for row in evidence)
    assert all(row["cleanup_confirmed"] is True for row in evidence)
    assert all(row["unsafe_collision_count"] == 0 for row in evidence)
    assert {
        row["run_id"]: (
            row["benign_pad_contact_count"],
            row["benign_pad_contact_cumulative_impulse"],
        )
        for row in evidence
    } == {
        "20260725T060252Z-calibration-excite-0726702b": (
            76,
            0.20524404116440564,
        ),
        "20260725T060328Z-calibration-excite-9e3562b1": (
            76,
            0.20600187953095883,
        ),
        "20260725T060354Z-calibration-excite-d78746e7": (
            76,
            0.20585722976829857,
        ),
    }
    assert all(row["watchdog_violation_count"] == 0 for row in evidence)
    assert {
        row["run_id"]: row["artifacts_sha256"] for row in evidence
    } == {
        "20260725T060252Z-calibration-excite-0726702b": {
            "result": (
                "7092784cf67ec93cf13582562ce1581098d1783cd88d53acedab7012581f9a19"
            ),
            "manifest": (
                "fbb9d483692cd933a61e2260c35ef1b4da53c0483bb11804bdcadeaeed7a3b5c"
            ),
            "trace": (
                "78c5caf55034baaf4de430a876983863c143d387fd9dcdf233c8613e06cbeedb"
            ),
            "live_lease": (
                "d2fecfa5f33641cb9ac505fe47237121b3da0b8f61e8995d7337c0d8af35d4a6"
            ),
        },
        "20260725T060328Z-calibration-excite-9e3562b1": {
            "result": (
                "c9a6ea5c6201d5150ef15786c23bcd378d57f5fc813b5192f13ad5c23e911fb9"
            ),
            "manifest": (
                "573de7ae64e44e1e8a06114a224bee32cbb9abf7eff1224c1886902e3b19e910"
            ),
            "trace": (
                "7017a19eef5f4fe918e56b25521837f7601805d6743ae47ea91757aabd4a2456"
            ),
            "live_lease": (
                "6b756d96d7e3ed022ef2e405ce083a66adec15de4e2dd168ccf9b1098f499387"
            ),
        },
        "20260725T060354Z-calibration-excite-d78746e7": {
            "result": (
                "402a108a4551c31b2f8909e2efd6ceec764e78797535c0e4759f58e1d4e8e19c"
            ),
            "manifest": (
                "c05682f040a505b7bb4923da94c74fecffaf6e4d74c2a5b6eabec1a101cdae09"
            ),
            "trace": (
                "3e2e31bd4b737facd1485facf088c89c5e27d4833532a600e7ec35ae7d0f61d8"
            ),
            "live_lease": (
                "11a80c6433218782f8bfd4aaec2b5469022bf07f2cb631ea5d4699dcd8130932"
            ),
        },
    }


def test_observed_ranges_are_exact_row_extrema_and_inside_authority():
    profile = _profile()
    rows = profile["evidence"]
    ranges = profile["observed_ranges"]

    for range_name in (
        "gyro_rate_gain",
        "image_rate_gain_px_per_command_rad",
        "gyro_response_delay_upper_bound_s",
        "first_image_observation_delay_s",
        "max_attitude_excursion_rad",
        "max_abs_measured_yaw_rate_rad_s",
    ):
        values = [row[range_name] for row in rows]
        assert ranges[range_name] == {
            "min": min(values),
            "max": max(values),
        }

    authority = profile["authority"]
    assert ranges["gyro_response_delay_upper_bound_s"]["max"] <= (
        authority["max_gyro_response_delay_s"]
    )
    assert ranges["first_image_observation_delay_s"]["max"] <= (
        authority["max_first_image_observation_delay_s"]
    )
    assert ranges["max_attitude_excursion_rad"]["max"] <= (
        authority["max_attitude_excursion_rad"]
    )
    assert ranges["max_abs_measured_yaw_rate_rad_s"]["max"] <= (
        authority["max_abs_measured_yaw_rate_rad_s"]
    )


def test_validation_and_loading_return_defensive_copies():
    original = _profile()
    validated = profile_module.validate_yaw_calibration_profile(original)
    validated["evidence"][0]["gyro_rate_gain"] = 99.0
    assert original["evidence"][0]["gyro_rate_gain"] != 99.0

    first = _profile()
    first["authority"]["controller_to_body_sign"] = -1
    second = _profile()
    assert second["authority"]["controller_to_body_sign"] == 1


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("simulator", "build"), 3384, "build"),
        (("simulator", "mode"), "Race", "mode"),
        (
            ("simulator", "mode_basis"),
            "visually_verified",
            "mode_basis",
        ),
        (("source", "commit"), "0" * 40, "commit"),
        (("source", "stage"), "sign-id", "stage"),
        (("source", "worktree_state"), "dirty", "worktree_state"),
        (("plan", "plan_id"), "different", "plan_id"),
        (("plan", "sha256"), "0" * 64, "sha256"),
        (("authority", "controller_to_body_sign"), -1, "body_sign"),
        (("authority", "controller_to_image_sign"), -1, "image_sign"),
        (
            ("authority", "max_abs_yaw_rate_command_rad_s"),
            0.081,
            "command",
        ),
        (
            ("authority", "max_gyro_response_delay_s"),
            0.081,
            "gyro",
        ),
        (
            ("authority", "max_first_image_observation_delay_s"),
            0.091,
            "image",
        ),
        (
            ("authority", "max_attitude_excursion_rad"),
            0.051,
            "attitude",
        ),
        (
            ("authority", "max_abs_measured_yaw_rate_rad_s"),
            0.51,
            "measured",
        ),
        (
            ("authority", "control_hold_horizon_s"),
            0.121,
            "hold",
        ),
        (("evidence", 0, "stage_success"), False, "stage_success"),
        (("evidence", 0, "cleanup_confirmed"), False, "cleanup"),
        (("evidence", 0, "unsafe_collision_count"), 1, "unsafe_collision"),
        (
            ("evidence", 0, "benign_pad_contact_count"),
            75,
            "benign_pad_contact_count",
        ),
        (
            ("evidence", 0, "benign_pad_contact_cumulative_impulse"),
            0.0,
            "benign_pad_contact_cumulative_impulse",
        ),
        (
            ("evidence", 0, "watchdog_violation_count"),
            1,
            "watchdog",
        ),
        (("evidence", 0, "controller_to_body_sign"), -1, "body_sign"),
        (("evidence", 0, "controller_to_image_sign"), -1, "image_sign"),
        (("evidence", 0, "command_rate_abs_rad_s"), 0.081, "command"),
        (("evidence", 0, "control_hold_horizon_s"), 0.121, "hold"),
        (("evidence", 0, "gyro_rate_gain"), 0.0, "gain"),
        (
            ("evidence", 0, "image_rate_gain_px_per_command_rad"),
            -1.0,
            "gain",
        ),
        (
            ("evidence", 0, "gyro_response_delay_upper_bound_s"),
            0.080001,
            "gyro response",
        ),
        (
            ("evidence", 0, "first_image_observation_delay_s"),
            0.090001,
            "image observation",
        ),
        (
            ("evidence", 0, "max_attitude_excursion_rad"),
            0.050001,
            "attitude excursion",
        ),
        (
            ("evidence", 0, "max_abs_measured_yaw_rate_rad_s"),
            0.500001,
            "measured yaw",
        ),
        (
            ("observed_ranges", "gyro_rate_gain", "min"),
            1.0,
            "summarize",
        ),
    ],
)
def test_validation_rejects_identity_authority_or_evidence_mutation(
    path, value, error
):
    profile = _profile()
    _set_path(profile, path, value)
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match=error,
    ):
        profile_module.validate_yaw_calibration_profile(profile)


def test_validation_rejects_wrong_count_duplicate_rows_and_artifacts():
    profile = _profile()
    profile["evidence"].pop()
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="exactly 3",
    ):
        profile_module.validate_yaw_calibration_profile(profile)

    profile = _profile()
    profile["evidence"][1]["run_id"] = profile["evidence"][0]["run_id"]
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="unique",
    ):
        profile_module.validate_yaw_calibration_profile(profile)

    profile = _profile()
    profile["evidence"][1]["artifacts_sha256"]["trace"] = (
        profile["evidence"][0]["artifacts_sha256"]["trace"]
    )
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="globally unique",
    ):
        profile_module.validate_yaw_calibration_profile(profile)


def test_validation_rejects_shapes_types_hashes_and_nonfinite_values():
    profile = _profile()
    profile["unknown"] = True
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="unknown",
    ):
        profile_module.validate_yaw_calibration_profile(profile)

    profile = _profile()
    del profile["authority"]["control_hold_horizon_s"]
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="missing",
    ):
        profile_module.validate_yaw_calibration_profile(profile)

    profile = _profile()
    profile["evidence"][0]["unsafe_collision_count"] = False
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="unsafe_collision_count",
    ):
        profile_module.validate_yaw_calibration_profile(profile)

    profile = _profile()
    profile["evidence"][0]["gyro_rate_gain"] = float("nan")
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="finite",
    ):
        profile_module.validate_yaw_calibration_profile(profile)

    profile = _profile()
    profile["evidence"][0]["artifacts_sha256"]["result"] = "ABC"
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="lowercase SHA-256",
    ):
        profile_module.validate_yaw_calibration_profile(profile)

    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="must equal frozen",
    ):
        profile_module.validate_yaw_calibration_profile(
            _profile(),
            expected_sha256="f" * 64,
        )


def test_loader_rejects_duplicate_keys_nonfinite_json_and_invalid_json(
    tmp_path,
):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="duplicate",
    ):
        profile_module.load_yaw_calibration_profile(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="non-finite",
    ):
        profile_module.load_yaw_calibration_profile(nonfinite)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="invalid JSON",
    ):
        profile_module.load_yaw_calibration_profile(invalid)


def test_loader_rejects_missing_and_oversized_files(tmp_path):
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="cannot read",
    ):
        profile_module.load_yaw_calibration_profile(tmp_path / "absent.json")

    oversized = tmp_path / "oversized.json"
    oversized.write_text(" " * (64 * 1024 + 1), encoding="utf-8")
    with pytest.raises(
        profile_module.YawCalibrationProfileError,
        match="64 KiB",
    ):
        profile_module.load_yaw_calibration_profile(oversized)
