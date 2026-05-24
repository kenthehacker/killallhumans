"""
Camera geometry tests — AIGP 640×360 + fx=fy=320 + 20° upward tilt.

Pre-fix:
- `PipelineConfig` defaults to image_height=480 (`race_pipeline.py:71`).
- `CameraIntrinsics` has no concept of body-frame tilt; PnP recovery
  silently assumes camera axis = body forward axis.

Post-fix (A4 + A6):
- Defaults read from `competition/aigp_geometry.py`.
- `CameraIntrinsics` carries a `pitch_offset_rad` field (+20° upward); the
  `body_R_camera` rotation is threaded through `gate_pose_to_drone_position`.
"""
from __future__ import annotations

import math

import pytest

from competition.aigp_geometry import (
    AIGP_CAM_CX,
    AIGP_CAM_CY,
    AIGP_CAM_FX,
    AIGP_CAM_FY,
    AIGP_CAM_HEIGHT_PX,
    AIGP_CAM_PITCH_OFFSET_RAD,
    AIGP_CAM_WIDTH_PX,
)


def _ci():
    """Build a default-constructed CameraIntrinsics.

    After A6, CameraIntrinsics() with no args must default to AIGP. Before A6,
    this raises TypeError — that is the expected red-phase failure mode.
    """
    from estimation.gate_pnp import CameraIntrinsics
    return CameraIntrinsics()


# ---------------------------------------------------------------------------
# I-8: intrinsics defaults must match the spec
# ---------------------------------------------------------------------------

def test_intrinsics_default_to_aigp():
    cam = _ci()
    assert cam.fx == pytest.approx(AIGP_CAM_FX)
    assert cam.fy == pytest.approx(AIGP_CAM_FY)
    assert cam.cx == pytest.approx(AIGP_CAM_CX)
    assert cam.cy == pytest.approx(AIGP_CAM_CY)
    assert cam.image_width == AIGP_CAM_WIDTH_PX
    assert cam.image_height == AIGP_CAM_HEIGHT_PX


def test_pipeline_config_defaults_to_aigp_image_size():
    from race_pipeline import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.image_width == AIGP_CAM_WIDTH_PX
    assert cfg.image_height == AIGP_CAM_HEIGHT_PX


# ---------------------------------------------------------------------------
# I-8 sub-issue: 20° upward camera tilt must be carried by intrinsics
# ---------------------------------------------------------------------------

def test_camera_pitch_offset_is_20_degrees_upward():
    cam = _ci()
    assert hasattr(cam, "pitch_offset_rad"), (
        "CameraIntrinsics must expose pitch_offset_rad (iter-001 A6)"
    )
    assert cam.pitch_offset_rad == pytest.approx(AIGP_CAM_PITCH_OFFSET_RAD)
    assert cam.pitch_offset_rad == pytest.approx(math.radians(20.0))


def test_horizon_projects_below_image_center_with_upward_tilt():
    """If the camera is pitched 20° up, a world-horizon ray projects below cy.

    Computation: for a point on the horizon at body-frame (x_body, 0, 0),
    its camera-frame depth is x_body·cos(20°) and its camera-frame
    vertical offset is x_body·sin(20°). Pinhole projection puts it at
        y_px = cy + fy · (x_body·sin(20°) / (x_body·cos(20°)))
             = cy + fy · tan(20°)
    With cy=180 and fy=320, that's 180 + 320·0.364 ≈ 296.5 px — well below
    the image centre, near the bottom edge.
    """
    cam = _ci()
    expected_y = cam.cy + cam.fy * math.tan(cam.pitch_offset_rad)
    # Locks the convention: positive pitch_offset_rad pushes horizon DOWN.
    assert expected_y == pytest.approx(296.49, abs=0.5)
    # And it must lie inside the image.
    assert 0 <= expected_y < cam.image_height


def test_camera_intrinsics_legacy_constructor_still_works():
    """The from_fov() factory must continue to accept (fov, w, h)."""
    from estimation.gate_pnp import CameraIntrinsics
    cam = CameraIntrinsics.from_fov(90.0, 640, 480)
    assert cam.image_width == 640
    assert cam.image_height == 480
    # fx from horizontal FoV: fx = w / (2·tan(fov/2)) = 640 / (2·tan(45°)) = 320.
    assert cam.fx == pytest.approx(320.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Iter-001 review composer-25-4 F8: AIGP FoV / intrinsics consistency
# ---------------------------------------------------------------------------

def test_aigp_intrinsics_hfov_matches_spec_90deg():
    """fx = 320, image width = 640 → HFoV = 2·atan(320/320) = 90°.
    The spec's "VFoV = 90°" claim is inconsistent; trust the intrinsics
    (which give H = 90° and V ≈ 58.7°)."""
    from competition.aigp_geometry import (
        AIGP_CAM_FX, AIGP_CAM_HFOV_DEG, AIGP_CAM_VFOV_DEG, AIGP_CAM_WIDTH_PX,
    )
    derived_hfov = math.degrees(
        2.0 * math.atan((AIGP_CAM_WIDTH_PX / 2) / AIGP_CAM_FX)
    )
    assert derived_hfov == pytest.approx(AIGP_CAM_HFOV_DEG, abs=0.05)
    assert derived_hfov == pytest.approx(90.0, abs=0.05)
    # The genuine VFoV is much smaller — assert we surface it correctly.
    assert AIGP_CAM_VFOV_DEG == pytest.approx(58.715, abs=0.05)
