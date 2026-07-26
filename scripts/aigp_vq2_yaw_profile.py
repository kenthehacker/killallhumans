"""Strict checked yaw-authority profile for FlightSim build 3385.

Three independent paired-polarity runs retain the image/body sign contract.
One collision-free free-flight paired-polarity run with confirmed
reset/disarm adds measured ``+/-0.15 rad/s`` rate authority.  Its separate
post-reset pad-contact warning remains in the compact live result.  Private
artifacts are identified by digest but are deliberately not required at
runtime.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

YAW_CALIBRATION_PROFILE_SCHEMA = "aigp-vq2-yaw-calibration-profile/2"
YAW_CALIBRATION_PROFILE_ID = "vq2-build3385-training-yaw-authority-v3"
YAW_CALIBRATION_SOURCE_COMMIT = (
    "f4eecd6afbca6ff69cf84b0a69339ed66238cfa0"
)
YAW_CALIBRATION_PROFILE_SHA256 = (
    "c544eb9e1b7d6e1e0590027e168a7b46e156b7557b98567653a3f1b518dca3a5"
)
YAW_CALIBRATION_PLAN_ID = "vq2-build3385-training-yaw-calibration-v1"
YAW_CALIBRATION_PLAN_SHA256 = (
    "827101741ddb335a1cbfcdbdcfca65c2f7579ad4a435c140179b8d9d0eb2be1b"
)

DEFAULT_YAW_CALIBRATION_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "aigp_vq2_yaw_calibration_build3385.json"
)

YAW_CONTROLLER_TO_BODY_SIGN = 1
YAW_CONTROLLER_TO_IMAGE_SIGN = 1
YAW_MAX_COMMAND_RATE_RAD_S = 0.15
YAW_MAX_GYRO_RESPONSE_DELAY_S = 0.08
YAW_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S = 0.09
YAW_MAX_CALIBRATION_ATTITUDE_EXCURSION_RAD = 0.10
YAW_MAX_CALIBRATION_MEASURED_RATE_RAD_S = 0.5
YAW_OBSERVED_MAX_MEASURED_RATE_RAD_S = 0.4234286031848751
YAW_CONTROL_HOLD_HORIZON_S = 0.12
YAW_SIGN_EVIDENCE_COMMAND_RATE_RAD_S = 0.08

_EXPECTED_RUN_IDS = frozenset(
    {
        "20260725T060252Z-calibration-excite-0726702b",
        "20260725T060328Z-calibration-excite-9e3562b1",
        "20260725T060354Z-calibration-excite-d78746e7",
    }
)
_EXPECTED_BENIGN_PAD_EVIDENCE = {
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
_CAPABILITY_EVIDENCE = {
    "run_id": "20260726T152438Z-calibration-excite-4a45eba7",
    "source_commit": "f19e66db789eed488c009d4b9a0846c0b75cad2f",
    "stage": "calibration-excite",
    "worktree_state": "clean",
    "plan_id": "vq2-build3385-training-free-flight-yaw-sweep-v3",
    "plan_sha256": (
        "742ab05e9ed02263981f03120be93c545f3dc9cb0637693488106f019960300a"
    ),
    "stage_success": True,
    "cleanup_confirmed": True,
    "unsafe_collision_count": 0,
    "watchdog_violation_count": 0,
    "broad_watchdog_unchanged": True,
    "command_rate_abs_rad_s": 0.15,
    "max_abs_body_rate_rad_s": 0.4234286031848751,
    "max_abs_transverse_body_rate_rad_s": 0.1977540485188365,
    "max_attitude_excursion_rad": 0.0753933678364203,
    "pulse_summaries": [
        {
            "segment_id": "yaw-positive-0p15",
            "command_yaw_rate_rad_s": 0.15,
            "wire_yaw_rate_rad_s": -0.15,
            "peak_corrected_yaw_rate_rad_s": 0.42343512922525406,
            "peak_command_to_body_rate_gain": 2.8229008615016937,
            "response_delay_upper_bound_s": 0.0137509,
            "max_abs_roll_rate_coupling_rad_s": (
                0.0031109408382326365
            ),
            "max_abs_pitch_rate_coupling_rad_s": (
                0.09257508140057326
            ),
            "max_abs_attitude_excursion_rad": (
                0.057868230468247246
            ),
        },
        {
            "segment_id": "yaw-negative-0p15",
            "command_yaw_rate_rad_s": -0.15,
            "wire_yaw_rate_rad_s": 0.15,
            "peak_corrected_yaw_rate_rad_s": 0.4098572116345167,
            "peak_command_to_body_rate_gain": 2.732381410896778,
            "response_delay_upper_bound_s": 0.0301872,
            "max_abs_roll_rate_coupling_rad_s": (
                0.004017784530296922
            ),
            "max_abs_pitch_rate_coupling_rad_s": (
                0.06834130059927702
            ),
            "max_abs_attitude_excursion_rad": 0.07166281246523945,
        },
    ],
    "artifacts_sha256": {
        "result": (
            "37c1e3ee688a1482c91d41ee2c6fd35559e33cb97b0559d3b8d28bf49ddbe3d2"
        ),
        "manifest": (
            "5457aac7672f760a80ab418773995beacf9b2ccec63ff0e10d507ceded570351"
        ),
        "trace": (
            "22cb10fc72d9b01d2e933c9b3908b243742c96f85bbf8af06def600005bdfc40"
        ),
        "live_lease": (
            "166de062b0045b5d683195ef8ff60dfc21c8a94af03fef8f578f94df3ff11cfb"
        ),
    },
}
_RANGE_TO_ROW_FIELD = {
    "gyro_rate_gain": "gyro_rate_gain",
    "image_rate_gain_px_per_command_rad": (
        "image_rate_gain_px_per_command_rad"
    ),
    "gyro_response_delay_upper_bound_s": (
        "gyro_response_delay_upper_bound_s"
    ),
    "first_image_observation_delay_s": (
        "first_image_observation_delay_s"
    ),
    "max_attitude_excursion_rad": "max_attitude_excursion_rad",
    "max_abs_measured_yaw_rate_rad_s": (
        "max_abs_measured_yaw_rate_rad_s"
    ),
}


class YawCalibrationProfileError(ValueError):
    """The tracked yaw-calibration profile is not exactly admissible."""


def _fail(path: str, detail: str) -> None:
    raise YawCalibrationProfileError(f"{path}: {detail}")


def _exact_object(
    value: Any,
    fields: set[str] | frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if type(value) is not dict:
        _fail(path, "must be an exact object")
    if any(type(key) is not str for key in value):
        _fail(path, "keys must be exact strings")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        _fail(
            path,
            "fields differ: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}",
        )
    return value


def _exact_array(value: Any, path: str, *, length: int) -> list[Any]:
    if type(value) is not list:
        _fail(path, "must be an exact array")
    if len(value) != length:
        _fail(path, f"must contain exactly {length} entries")
    return value


def _exact_string(value: Any, path: str) -> str:
    if type(value) is not str or not value:
        _fail(path, "must be a nonempty exact string")
    return value


def _exact_int(value: Any, path: str) -> int:
    if type(value) is not int:
        _fail(path, "must be an exact int")
    return value


def _exact_float(value: Any, path: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(path, "must be an exact finite float")
    return value


def _exact_true(value: Any, path: str) -> None:
    if value is not True:
        _fail(path, "must be exact true")


def _exact_sha256(value: Any, path: str) -> str:
    digest = _exact_string(value, path)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        _fail(path, "must be a lowercase SHA-256 hex digest")
    return digest


def _expect(value: Any, expected: Any, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(path, f"must equal frozen {expected!r}")


def canonical_yaw_calibration_profile_bytes(value: Any) -> bytes:
    """Return the sole deterministic representation used for profile identity."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise YawCalibrationProfileError(
            f"$: value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_yaw_calibration_profile_sha256(value: Any) -> str:
    return hashlib.sha256(
        canonical_yaw_calibration_profile_bytes(value)
    ).hexdigest()


def yaw_calibration_profile_evidence(value: Any) -> dict[str, Any]:
    """Return the small immutable runtime/manifest authority identity."""

    profile = validate_yaw_calibration_profile(value)
    return {
        "profile_id": profile["profile_id"],
        "sha256": YAW_CALIBRATION_PROFILE_SHA256,
        "source_commit": profile["source"]["commit"],
        "plan_id": profile["plan"]["plan_id"],
        "plan_sha256": profile["plan"]["sha256"],
        "authority": deepcopy(profile["authority"]),
    }


def _validate_simulator(value: Any) -> None:
    simulator = _exact_object(
        value,
        {"build", "mode", "mode_basis"},
        "$profile.simulator",
    )
    _expect(simulator["build"], 3385, "$profile.simulator.build")
    _expect(simulator["mode"], "Training", "$profile.simulator.mode")
    _expect(
        simulator["mode_basis"],
        "configured_session_not_machine_readable",
        "$profile.simulator.mode_basis",
    )


def _validate_source(value: Any) -> None:
    source = _exact_object(
        value,
        {"commit", "stage", "worktree_state"},
        "$profile.source",
    )
    _expect(
        source["commit"],
        YAW_CALIBRATION_SOURCE_COMMIT,
        "$profile.source.commit",
    )
    _expect(source["stage"], "calibration-excite", "$profile.source.stage")
    _expect(source["worktree_state"], "clean", "$profile.source.worktree_state")


def _validate_plan(value: Any) -> None:
    plan = _exact_object(
        value,
        {"plan_id", "sha256"},
        "$profile.plan",
    )
    _expect(
        plan["plan_id"],
        YAW_CALIBRATION_PLAN_ID,
        "$profile.plan.plan_id",
    )
    _expect(
        plan["sha256"],
        YAW_CALIBRATION_PLAN_SHA256,
        "$profile.plan.sha256",
    )


def _validate_authority(value: Any) -> Mapping[str, Any]:
    authority = _exact_object(
        value,
        {
            "controller_to_body_sign",
            "controller_to_image_sign",
            "max_abs_yaw_rate_command_rad_s",
            "max_gyro_response_delay_s",
            "max_first_image_observation_delay_s",
            "max_attitude_excursion_rad",
            "max_abs_measured_yaw_rate_rad_s",
            "control_hold_horizon_s",
        },
        "$profile.authority",
    )
    expected = {
        "controller_to_body_sign": YAW_CONTROLLER_TO_BODY_SIGN,
        "controller_to_image_sign": YAW_CONTROLLER_TO_IMAGE_SIGN,
        "max_abs_yaw_rate_command_rad_s": YAW_MAX_COMMAND_RATE_RAD_S,
        "max_gyro_response_delay_s": YAW_MAX_GYRO_RESPONSE_DELAY_S,
        "max_first_image_observation_delay_s": (
            YAW_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S
        ),
        "max_attitude_excursion_rad": (
            YAW_MAX_CALIBRATION_ATTITUDE_EXCURSION_RAD
        ),
        "max_abs_measured_yaw_rate_rad_s": (
            YAW_MAX_CALIBRATION_MEASURED_RATE_RAD_S
        ),
        "control_hold_horizon_s": YAW_CONTROL_HOLD_HORIZON_S,
    }
    for name, expected_value in expected.items():
        _expect(authority[name], expected_value, f"$profile.authority.{name}")
    return authority


def _validate_artifacts(value: Any, path: str) -> tuple[str, ...]:
    artifacts = _exact_object(
        value,
        {"result", "manifest", "trace", "live_lease"},
        path,
    )
    return tuple(
        _exact_sha256(artifacts[name], f"{path}.{name}")
        for name in ("result", "manifest", "trace", "live_lease")
    )


def _validate_evidence(
    value: Any,
    authority: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    evidence = _exact_array(value, "$profile.evidence", length=3)
    row_fields = {
        "run_id",
        "stage_success",
        "cleanup_confirmed",
        "unsafe_collision_count",
        "benign_pad_contact_count",
        "benign_pad_contact_cumulative_impulse",
        "watchdog_violation_count",
        "artifacts_sha256",
        "controller_to_body_sign",
        "controller_to_image_sign",
        "command_rate_abs_rad_s",
        "control_hold_horizon_s",
        "gyro_rate_gain",
        "image_rate_gain_px_per_command_rad",
        "gyro_response_delay_upper_bound_s",
        "first_image_observation_delay_s",
        "max_attitude_excursion_rad",
        "max_abs_measured_yaw_rate_rad_s",
    }
    rows: list[Mapping[str, Any]] = []
    run_ids: set[str] = set()
    artifact_hashes: set[str] = set()
    for index, row_value in enumerate(evidence):
        path = f"$profile.evidence[{index}]"
        row = _exact_object(row_value, row_fields, path)
        run_id = _exact_string(row["run_id"], f"{path}.run_id")
        if run_id in run_ids:
            _fail(f"{path}.run_id", "must be unique")
        run_ids.add(run_id)

        _exact_true(row["stage_success"], f"{path}.stage_success")
        _exact_true(row["cleanup_confirmed"], f"{path}.cleanup_confirmed")
        _expect(
            row["unsafe_collision_count"],
            0,
            f"{path}.unsafe_collision_count",
        )
        expected_pad_count, expected_pad_impulse = (
            _EXPECTED_BENIGN_PAD_EVIDENCE.get(run_id, (None, None))
        )
        _expect(
            row["benign_pad_contact_count"],
            expected_pad_count,
            f"{path}.benign_pad_contact_count",
        )
        _expect(
            row["benign_pad_contact_cumulative_impulse"],
            expected_pad_impulse,
            f"{path}.benign_pad_contact_cumulative_impulse",
        )
        _expect(
            row["watchdog_violation_count"],
            0,
            f"{path}.watchdog_violation_count",
        )

        hashes = _validate_artifacts(
            row["artifacts_sha256"],
            f"{path}.artifacts_sha256",
        )
        for digest in hashes:
            if digest in artifact_hashes:
                _fail(
                    f"{path}.artifacts_sha256",
                    "artifact identities must be globally unique",
                )
            artifact_hashes.add(digest)

        _expect(
            row["controller_to_body_sign"],
            authority["controller_to_body_sign"],
            f"{path}.controller_to_body_sign",
        )
        _expect(
            row["controller_to_image_sign"],
            authority["controller_to_image_sign"],
            f"{path}.controller_to_image_sign",
        )
        _expect(
            row["command_rate_abs_rad_s"],
            YAW_SIGN_EVIDENCE_COMMAND_RATE_RAD_S,
            f"{path}.command_rate_abs_rad_s",
        )
        _expect(
            row["control_hold_horizon_s"],
            authority["control_hold_horizon_s"],
            f"{path}.control_hold_horizon_s",
        )

        gyro_gain = _exact_float(
            row["gyro_rate_gain"],
            f"{path}.gyro_rate_gain",
        )
        image_gain = _exact_float(
            row["image_rate_gain_px_per_command_rad"],
            f"{path}.image_rate_gain_px_per_command_rad",
        )
        gyro_delay = _exact_float(
            row["gyro_response_delay_upper_bound_s"],
            f"{path}.gyro_response_delay_upper_bound_s",
        )
        image_delay = _exact_float(
            row["first_image_observation_delay_s"],
            f"{path}.first_image_observation_delay_s",
        )
        attitude = _exact_float(
            row["max_attitude_excursion_rad"],
            f"{path}.max_attitude_excursion_rad",
        )
        measured_rate = _exact_float(
            row["max_abs_measured_yaw_rate_rad_s"],
            f"{path}.max_abs_measured_yaw_rate_rad_s",
        )
        if gyro_gain <= 0.0 or image_gain <= 0.0:
            _fail(path, "both observed yaw gains must be positive")
        if not 0.0 <= gyro_delay <= authority["max_gyro_response_delay_s"]:
            _fail(path, "gyro response delay exceeds accepted authority")
        if not (
            0.0
            <= image_delay
            <= authority["max_first_image_observation_delay_s"]
        ):
            _fail(path, "image observation delay exceeds accepted authority")
        if not 0.0 <= attitude <= authority["max_attitude_excursion_rad"]:
            _fail(path, "attitude excursion exceeds calibration bound")
        if not (
            0.0
            <= measured_rate
            <= authority["max_abs_measured_yaw_rate_rad_s"]
        ):
            _fail(path, "measured yaw rate exceeds calibration bound")
        rows.append(row)

    if run_ids != set(_EXPECTED_RUN_IDS):
        _fail(
            "$profile.evidence",
            "run identities differ from the three accepted clean runs",
        )
    return rows


def _validate_observed_ranges(
    value: Any,
    rows: list[Mapping[str, Any]],
) -> None:
    ranges = _exact_object(
        value,
        set(_RANGE_TO_ROW_FIELD),
        "$profile.observed_ranges",
    )
    for range_name, row_field in _RANGE_TO_ROW_FIELD.items():
        path = f"$profile.observed_ranges.{range_name}"
        observed = _exact_object(ranges[range_name], {"min", "max"}, path)
        minimum = _exact_float(observed["min"], f"{path}.min")
        maximum = _exact_float(observed["max"], f"{path}.max")
        values = [row[row_field] for row in rows]
        if minimum != min(values) or maximum != max(values):
            _fail(path, "must exactly summarize the three evidence rows")


def _validate_capability(
    value: Any,
    authority: Mapping[str, Any],
) -> None:
    _expect(value, _CAPABILITY_EVIDENCE, "$profile.capability")
    _expect(
        value["command_rate_abs_rad_s"],
        authority["max_abs_yaw_rate_command_rad_s"],
        "$profile.capability.command_rate_abs_rad_s",
    )
    if (
        value["max_abs_body_rate_rad_s"]
        > authority["max_abs_measured_yaw_rate_rad_s"]
        or value["max_attitude_excursion_rad"]
        > authority["max_attitude_excursion_rad"]
        or max(
            pulse["response_delay_upper_bound_s"]
            for pulse in value["pulse_summaries"]
        )
        > authority["max_gyro_response_delay_s"]
    ):
        _fail(
            "$profile.capability",
            "measured response exceeds accepted authority",
        )


def validate_yaw_calibration_profile(
    value: Any,
    *,
    expected_sha256: str = YAW_CALIBRATION_PROFILE_SHA256,
) -> dict[str, Any]:
    """Validate the frozen profile and return a defensive mutable copy."""

    profile = _exact_object(
        value,
        {
            "schema",
            "profile_id",
            "simulator",
            "source",
            "plan",
            "authority",
            "capability",
            "observed_ranges",
            "evidence",
        },
        "$profile",
    )
    _expect(
        profile["schema"],
        YAW_CALIBRATION_PROFILE_SCHEMA,
        "$profile.schema",
    )
    _expect(
        profile["profile_id"],
        YAW_CALIBRATION_PROFILE_ID,
        "$profile.profile_id",
    )
    _validate_simulator(profile["simulator"])
    _validate_source(profile["source"])
    _validate_plan(profile["plan"])
    authority = _validate_authority(profile["authority"])
    _validate_capability(profile["capability"], authority)
    rows = _validate_evidence(profile["evidence"], authority)
    _validate_observed_ranges(profile["observed_ranges"], rows)

    expected = _exact_sha256(expected_sha256, "$expected_sha256")
    if expected != YAW_CALIBRATION_PROFILE_SHA256:
        _fail(
            "$expected_sha256",
            f"must equal frozen {YAW_CALIBRATION_PROFILE_SHA256}",
        )
    actual = canonical_yaw_calibration_profile_sha256(profile)
    if actual != YAW_CALIBRATION_PROFILE_SHA256:
        _fail(
            "$profile",
            f"object SHA-256 must equal frozen "
            f"{YAW_CALIBRATION_PROFILE_SHA256}",
        )
    return deepcopy(value)


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise YawCalibrationProfileError(
                f"$: duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise YawCalibrationProfileError(
        f"$: non-finite JSON number {token!r} is forbidden"
    )


def load_yaw_calibration_profile(
    path: str | os.PathLike[str] = DEFAULT_YAW_CALIBRATION_PROFILE_PATH,
) -> dict[str, Any]:
    """Load and validate the tracked profile without consulting private evidence."""

    source_path = Path(path)
    try:
        payload = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise YawCalibrationProfileError(
            f"$file: cannot read yaw-calibration profile: {exc}"
        ) from exc
    if len(payload.encode("utf-8")) > 64 * 1024:
        _fail("$file", "yaw-calibration profile exceeds 64 KiB")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except YawCalibrationProfileError:
        raise
    except json.JSONDecodeError as exc:
        raise YawCalibrationProfileError(
            f"$file: invalid JSON: {exc.msg}"
        ) from exc
    return validate_yaw_calibration_profile(value)


__all__ = [
    "DEFAULT_YAW_CALIBRATION_PROFILE_PATH",
    "YAW_CALIBRATION_PROFILE_ID",
    "YAW_CALIBRATION_PROFILE_SCHEMA",
    "YAW_CALIBRATION_PROFILE_SHA256",
    "YAW_CALIBRATION_SOURCE_COMMIT",
    "YAW_CONTROLLER_TO_BODY_SIGN",
    "YAW_CONTROLLER_TO_IMAGE_SIGN",
    "YAW_CONTROL_HOLD_HORIZON_S",
    "YAW_MAX_CALIBRATION_ATTITUDE_EXCURSION_RAD",
    "YAW_MAX_CALIBRATION_MEASURED_RATE_RAD_S",
    "YAW_MAX_COMMAND_RATE_RAD_S",
    "YAW_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S",
    "YAW_MAX_GYRO_RESPONSE_DELAY_S",
    "YAW_OBSERVED_MAX_MEASURED_RATE_RAD_S",
    "YawCalibrationProfileError",
    "canonical_yaw_calibration_profile_bytes",
    "canonical_yaw_calibration_profile_sha256",
    "load_yaw_calibration_profile",
    "validate_yaw_calibration_profile",
    "yaw_calibration_profile_evidence",
]
