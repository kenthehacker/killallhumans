"""Stage regression for the attempt-4 near-plane vertical-rate sample.

This is compact logged-state replay, not JPEG or detector replay.  It drives
the real visual-course coordinator with the exact three clean tracker facts
that preceded attempt 4's bottom censorship, including the third sample whose
raw vertical image rate exceeds the old scalar rate bound while its projected
center remains inside the safe corridor.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import math

import pytest

from planning.vq2_visual_approach import VisualApproachMode
from planning.vq2_visual_servo import (
    PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S,
    PREPASS_CURRENT_MAX_ABS_Y_NORM,
    PREPASS_CURRENT_PROJECTION_HORIZON_S,
)
from scripts.aigp_vq2_visual_course_stage import run_visual_course_stage
from tests.test_aigp_vq2_visual_course import _context
from tests.test_aigp_vq2_visual_course_coordinator_replay import (
    _CadencedCoordinatorHost,
    _CadencedCoordinatorServo,
    _cadenced_runtime,
    _state_suffix,
)
from tests.test_aigp_vq2_visual_course_recorded_facts import (
    _ATTEMPT4_BOTTOM_CENSOR,
    _ATTEMPT4_NEAR_PLANE_ROWS,
)


_ATTEMPT4_CLEAN_ROWS = _ATTEMPT4_NEAR_PLANE_ROWS[2:5]


class _Attempt4Servo(_CadencedCoordinatorServo):
    def observe(self, snapshot, *args, **kwargs):
        proposal = super().observe(snapshot, *args, **kwargs)
        if proposal.mode is not VisualApproachMode.PASSAGE:
            return proposal

        index = min(
            self.generic_passage_sample_count - 1,
            len(_ATTEMPT4_CLEAN_ROWS) - 1,
        )
        row = _ATTEMPT4_CLEAN_ROWS[index]
        proposal.current_target = replace(
            proposal.current_target,
            normalized_x=row["x"],
            normalized_y_down=row["y"],
            normalized_x_rate_s=row["x_rate"],
            normalized_y_rate_down_s=row["y_rate"],
            log_scale=math.log(row["scale"]),
            log_scale_rate_s=row["scale_rate"],
            confidence=row["confidence"],
            association_confidence=row["association"],
        )
        return proposal


class _Attempt4Host(_CadencedCoordinatorHost):
    def _install_snapshot(self, *, token, publication_s, state):
        super()._install_snapshot(
            token=token,
            publication_s=publication_s,
            state=state,
        )
        if state != "bottom":
            return
        track = self.visual_gate_graph.latest_snapshot.current_track
        track.center_norm = (
            _ATTEMPT4_BOTTOM_CENSOR["x"],
            _ATTEMPT4_BOTTOM_CENSOR["y"],
        )
        track.center_velocity_norm_s = (
            _ATTEMPT4_BOTTOM_CENSOR["x_rate"],
            _ATTEMPT4_BOTTOM_CENSOR["y_rate"],
        )
        track.apparent_scale = _ATTEMPT4_BOTTOM_CENSOR["scale"]
        track.confidence = _ATTEMPT4_BOTTOM_CENSOR["confidence"]
        track.association_confidence = (
            _ATTEMPT4_BOTTOM_CENSOR["association"]
        )


def _attempt4_runtime(host):
    runtime = _cadenced_runtime(host)
    servo_calls = []
    return replace(
        runtime,
        servo_factory=lambda *args, **kwargs: _Attempt4Servo(
            *args,
            **kwargs,
            calls=servo_calls,
            yaw_rate=0.0,
        ),
    )


def test_attempt4_safe_projected_rate_latches_then_coasts_to_credit_wait():
    third = _ATTEMPT4_CLEAN_ROWS[-1]
    projected_y = (
        third["y"]
        + third["y_rate"] * PREPASS_CURRENT_PROJECTION_HORIZON_S
    )
    assert (
        third["y_rate"]
        > PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S
    )
    assert abs(projected_y) < PREPASS_CURRENT_MAX_ABS_Y_NORM

    host = _Attempt4Host(
        credit_policy="delayed",
        finish_gate=1,
    )
    result = asyncio.run(
        run_visual_course_stage(
            host,
            _context(),
            runtime=_attempt4_runtime(host),
        )
    )

    assert result["success"] is True
    assert result["race_finished"] is True
    segment = result["segments"][0]
    assert segment["passage_authority_enabled"] is True
    assert segment["near_plane_latch"] is not None
    assert (
        segment["near_plane_latch"]["accepted_wire_frame_count"]
        == len(_ATTEMPT4_CLEAN_ROWS)
    )
    assert segment["near_plane_latch"][
        "normalized_y_rate_down_s"
    ] == pytest.approx(third["y_rate"])
    assert segment["near_plane_measurement_mode"] == "credit_wait"
    assert segment["censored_passage_coast_command_count"] >= 1
    assert segment["crossing_wait_zero_command_count"] >= 1
    assert _state_suffix(host, 1) == [
        "bottom",
        "top_bottom",
        "lost",
    ]
