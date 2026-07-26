"""Tests for the compact, read-only visual-course cohort analyzer."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts import aigp_vq2_course_cohort as cohort


BASE_TIME_S = 1_000.0
DT_S = 0.02


def _wall_time_ns(index: int) -> int:
    return int(round((BASE_TIME_S + index * DT_S) * 1_000_000_000))


def _command(index: int) -> dict[str, float]:
    yaw_pattern = (0.0, 0.08, -0.04, 0.12, -0.10, 0.03, -0.07)
    roll_pattern = (0.04, 0.04, -0.05, -0.05, 0.06, 0.06, -0.04)
    pitch_pattern = (-0.03, 0.02, 0.06, -0.01, -0.05, 0.04, 0.0)
    thrust_pattern = (0.25, 0.27, 0.24, 0.29, 0.26, 0.23, 0.28)
    return {
        "yaw_rate": yaw_pattern[index % len(yaw_pattern)],
        "roll_rate": roll_pattern[index % len(roll_pattern)],
        "pitch_rate": pitch_pattern[index % len(pitch_pattern)],
        "thrust": thrust_pattern[index % len(thrust_pattern)],
    }


def _tick(index: int) -> dict[str, object]:
    command = _command(index)
    delayed = _command(max(0, index - 3))
    return {
        "event": "tick",
        "wall_time_ns": _wall_time_ns(index),
        "gate_index": 1,
        "stage": "visual-course/gate1/turn",
        "command": command,
        "body_rates": [
            1.8 * delayed["roll_rate"],
            2.1 * delayed["pitch_rate"],
            2.7 * delayed["yaw_rate"],
        ],
        "rpy": [0.01, -0.05, 0.02],
        "imu_accel": [
            0.0,
            0.0,
            -9.8 - 24.0 * (delayed["thrust"] - 0.25),
        ],
    }


def _track(
    *,
    track_id: str,
    role: str,
    x: float,
    y_down: float,
    log_scale: float,
) -> dict[str, object]:
    return {
        "track_id": track_id,
        "role": role,
        "visible": True,
        "ambiguous": False,
        "center_censored": False,
        "clipping_edges": 0,
        "center_norm_image_down": [x, y_down],
        "center_velocity_norm_s_image_down": [0.0, 0.0],
        "apparent_scale": math.exp(log_scale),
        "log_scale_rate_s": 0.0,
    }


def _graph_frame(index: int) -> dict[str, object]:
    return {
        "event": "visual_gate_graph_frame",
        "wall_time_ns": _wall_time_ns(index),
        "received_monotonic_s": BASE_TIME_S + index * DT_S,
        "gate_index": 1,
        "graph": {
            "current_gate_index": 1,
            "current_track_id": "current-1",
            "next_candidates": [{"track_id": "successor-2"}],
            "race_finished": False,
        },
        "tracks": [
            _track(
                track_id="current-1",
                role="current",
                x=0.24 - 0.003 * index,
                y_down=-0.10 + 0.001 * index,
                log_scale=-2.0 + 0.02 * index,
            ),
            _track(
                track_id="successor-2",
                role="next",
                x=0.62 - 0.006 * index,
                y_down=-0.25 + 0.002 * index,
                log_scale=-2.8 + 0.008 * index,
            ),
        ],
    }


def _terminal_event() -> dict[str, object]:
    return {
        "event": "visual_course_post_cleanup",
        "wall_time_ns": _wall_time_ns(40),
        "initial_gate_index": 0,
        "final_gate_index": 1,
        "maximum_authoritative_gate_index": 1,
        "authoritative_transitions": [
            {
                "from_gate_index": 0,
                "to_gate_index": 1,
                "promotion_confirmed": True,
            }
        ],
        "race_finished": False,
        "cleanup_confirmed": True,
        "outcome": "aborted",
    }


def _result(run_id: str) -> dict[str, object]:
    return {
        "schema": "aigp-vq2-fast-flight-cycle-result/2",
        "run_id": run_id,
        "stage": "visual-course",
        "success": False,
        "reason": "synthetic first physical blocker",
        "runner_result": {
            "cleanup_confirmed": True,
            "details": {
                "visual_course": {
                    "initial_gate_index": 0,
                    "final_gate_index": 1,
                    "maximum_authoritative_gate_index": 1,
                    "authoritative_transitions": [
                        {
                            "from_gate_index": 0,
                            "to_gate_index": 1,
                            "promotion_confirmed": True,
                            "post_transition_navigation_command_count": 18,
                        }
                    ],
                    "race_finished": False,
                    "outcome": "aborted",
                }
            },
        },
    }


def _write_run(
    root: Path,
    run_id: str,
    *,
    corrupt_result: bool = False,
    truncate_trace: bool = False,
) -> Path:
    directory = root / run_id
    directory.mkdir()
    result_path = directory / "result.json"
    if corrupt_result:
        result_path.write_text("{broken", encoding="utf-8")
    else:
        result_path.write_text(
            json.dumps(_result(run_id)),
            encoding="utf-8",
        )
    trace_path = directory / "session.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
        for index in range(36):
            stream.write(json.dumps(_tick(index)) + "\n")
            stream.write(json.dumps(_graph_frame(index)) + "\n")
        stream.write(json.dumps(_terminal_event()) + "\n")
    if truncate_trace:
        payload = trace_path.read_bytes()
        trace_path.write_bytes(payload[:-8])
    return directory


def _identity(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return (
        stat.st_size,
        stat.st_mtime_ns,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_delayed_effect_applies_known_command_history_delay():
    pattern = (0.11, -0.07, 0.03, 0.14, -0.12, 0.06, -0.02, 0.09)
    commands = [
        cohort.TimedValue(index * DT_S, pattern[index % len(pattern)])
        for index in range(60)
    ]
    responses = [
        cohort.TimedValue(
            index * DT_S,
            2.5 * commands[index - 3].value + 0.4,
        )
        for index in range(3, 60)
    ]

    estimate = cohort.estimate_delayed_effect(
        commands,
        responses,
        response_name="synthetic_response",
    )

    assert estimate["status"] == "identified"
    # A 20 ms step-held command can only localize this exact 60 ms response
    # to the adjacent 10 ms search cells; the report retains that honest
    # interval instead of claiming sub-tick precision.
    assert estimate["lag_ms"] == pytest.approx(50.0)
    assert estimate["lag_range_ms"] == pytest.approx(
        {"lower": 50.0, "upper": 60.0}
    )
    assert estimate["gain"] == pytest.approx(2.5)
    assert estimate["correlation"] == pytest.approx(1.0)


def test_command_metrics_record_slew_and_deadbanded_roll_reversals():
    roll_values = (0.05, -0.05, 0.06, -0.06, 0.001, -0.07)
    ticks = [
        cohort.TickSample(
            time_s=index * DT_S,
            command={
                "roll_rate": roll,
                "pitch_rate": 0.0,
                "yaw_rate": 0.0,
                "thrust": 0.25,
            },
            body_rates=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 0.0),
            imu_accel=(0.0, 0.0, -9.8),
            gate_index=0,
        )
        for index, roll in enumerate(roll_values)
    ]

    metrics = cohort._command_metrics(ticks)

    assert metrics["roll_reversals"]["count"] == 3
    assert metrics["roll_reversals"]["minimum_interval_ms"] == pytest.approx(
        20.0
    )
    assert metrics["axes"]["roll_rate"]["max_abs_slew_per_s"] == pytest.approx(
        6.0
    )


def test_cohort_reports_progress_trends_effects_and_never_mutates_evidence(
    tmp_path: Path,
):
    run_id = "20260726T210000Z-visual-course-01234567"
    run_directory = _write_run(tmp_path, run_id)
    result_path = run_directory / "result.json"
    trace_path = run_directory / "session.jsonl.gz"
    before = {
        result_path: _identity(result_path),
        trace_path: _identity(trace_path),
    }

    report = cohort.analyze_cohort(tmp_path)

    assert report["schema"] == cohort.REPORT_SCHEMA
    assert report["assumptions"]["read_only"] is True
    assert report["aggregate"]["run_count"] == 1
    assert report["aggregate"]["highest_authoritative_gate_index"] == 1
    assert report["aggregate"]["race_finished_run_count"] == 0
    run = report["runs"][0]
    assert run["authoritative_progress"]["source"] == "result"
    assert run["authoritative_progress"]["transition_count"] == 1
    assert run["commands"]["tick_count"] == 36
    assert run["commands"]["roll_reversals"]["count"] > 0
    gate = run["gate_features"][0]
    assert gate["current"]["bearing_x"]["slope_per_s"] == pytest.approx(-0.15)
    assert gate["successor"]["bearing_x"]["slope_per_s"] == pytest.approx(-0.30)
    assert gate["current"]["closure"]["expansion_rate_log_s"] == pytest.approx(
        1.0
    )
    assert run["effects"]["yaw"]["imu"]["status"] == "identified"
    assert run["effects"]["yaw"]["imu"]["lag_range_ms"]["upper"] == (
        pytest.approx(60.0)
    )
    assert run["residual_horizontal_motion"]["sample_count"] > 20
    assert "highest_gate=1" in cohort.format_table(report)
    json.dumps(report, allow_nan=False)
    assert {
        result_path: _identity(result_path),
        trace_path: _identity(trace_path),
    } == before


def test_partial_corrupt_run_retains_valid_trace_facts(tmp_path: Path):
    run_id = "20260726T205900Z-visual-course-89abcdef"
    run_directory = _write_run(
        tmp_path,
        run_id,
        corrupt_result=True,
        truncate_trace=True,
    )
    result_path = run_directory / "result.json"
    trace_path = run_directory / "session.jsonl.gz"
    before = (_identity(result_path), _identity(trace_path))

    report = cohort.analyze_cohort(tmp_path)

    run = report["runs"][0]
    assert run["result"]["status"] == "unavailable"
    assert run["trace"]["status"] == "partial"
    assert run["trace"]["event_count"] > 0
    assert run["commands"]["tick_count"] == 36
    assert run["authoritative_progress"]["source"] == "trace_summary"
    assert run["authoritative_progress"]["maximum_gate_index"] == 1
    assert report["aggregate"]["partial_or_missing_trace_count"] == 1
    assert (_identity(result_path), _identity(trace_path)) == before


def test_missing_root_and_missing_artifacts_are_machine_readable(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    report = cohort.analyze_cohort(missing)
    assert report["evidence_root_status"] == "missing"
    assert report["aggregate"]["run_count"] == 0

    empty_run = tmp_path / "20260726T205800Z-visual-course-feedface"
    empty_run.mkdir()
    report = cohort.analyze_cohort(tmp_path)
    run = report["runs"][0]
    assert run["result"]["error"] == "missing"
    assert run["trace"]["read_error"] == "missing"
    assert run["authoritative_progress"]["source"] == "unavailable"


def test_cli_emits_json_and_honors_newest_run_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    _write_run(tmp_path, "20260726T205700Z-visual-course-11111111")
    _write_run(tmp_path, "20260726T205701Z-visual-course-22222222")

    exit_code = cohort.main(
        [
            "--root",
            str(tmp_path),
            "--format",
            "json",
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection"]["selected_run_count"] == 1
    assert payload["runs"][0]["run_id"].endswith("22222222")
