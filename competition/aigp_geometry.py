"""
AI Grand Prix VQ1 — authoritative geometry, camera, and timing constants.

Source of truth: VADR-TS-002 issue 00.02 (2026-05-08).
DO NOT hard-code these numbers anywhere else; import from this module.

Adding a new constant here must be backed by a literal entry in the PDF
spec. Track-specific tunings (e.g. race_01.json's 1.2 m gate openings) are
NOT AIGP constants and live in their own config files.
"""
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Gate geometry — VADR-TS-002 §3.7
# ---------------------------------------------------------------------------
AIGP_GATE_INTERIOR_M: float = 1.5     # inner-square opening (1500 mm × 1500 mm)
AIGP_GATE_BORDER_M: float = 0.6       # frame thickness on each side (600 mm)
AIGP_GATE_OUTER_M: float = 2.7        # outer frame (2700 mm × 2700 mm)
AIGP_GATE_DEPTH_M: float = 0.26       # depth through gate (260 mm)


# ---------------------------------------------------------------------------
# Drone chassis — VADR-TS-002 §3.6
# ---------------------------------------------------------------------------
AIGP_DRONE_WIDTH_M: float = 0.28      # X (forward) dimension (280 mm)
AIGP_DRONE_LENGTH_M: float = 0.28     # Y (right) dimension (280 mm)
AIGP_DRONE_HEIGHT_M: float = 0.16     # Z (down) dimension (160 mm)
# Mass / thrust are NOT specified in the PDF — they must be inferred via
# SITL calibration. Do not assume Crazyflie CF2X-class values.


# ---------------------------------------------------------------------------
# Forward camera — VADR-TS-002 §3.8 and §4.6
# ---------------------------------------------------------------------------
AIGP_CAM_WIDTH_PX: int = 640
AIGP_CAM_HEIGHT_PX: int = 360
AIGP_CAM_FX: float = 320.0
AIGP_CAM_FY: float = 320.0
AIGP_CAM_CX: float = 320.0
AIGP_CAM_CY: float = 180.0
AIGP_CAM_VFOV_DEG: float = 90.0
AIGP_CAM_VFOV_RAD: float = math.radians(90.0)
# Camera is tilted 20° UPWARD from body frame (positive pitch = nose-up).
# A feature at the horizon therefore projects BELOW image centre by
# ~fy·tan(20°) ≈ 116 px.
AIGP_CAM_PITCH_OFFSET_RAD: float = math.radians(20.0)
AIGP_CAM_FPS: int = 30
AIGP_CAM_UDP_PORT: int = 5600


# ---------------------------------------------------------------------------
# MAVLink + timing — VADR-TS-002 §4
# ---------------------------------------------------------------------------
AIGP_PHYSICS_HZ: int = 120
AIGP_MAX_CMD_HZ: int = 100              # spec says "<100 Hz"; 100 is the ceiling
AIGP_MIN_HEARTBEAT_HZ: int = 2


# ---------------------------------------------------------------------------
# Round-one qualification rules
# ---------------------------------------------------------------------------
AIGP_VQ1_MAX_RUN_DURATION_S: float = 8 * 60.0   # 8 minutes
AIGP_VQ1_MAX_GATES: int = 10                    # spec says "< 10"
AIGP_VQ2_MAX_GATES: int = 20                    # spec says "< 20"
