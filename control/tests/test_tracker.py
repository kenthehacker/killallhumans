"""Tests for trajectory tracker (control/mpc_tracker.py).

Phase 2 regression tests: hover correctness, sign conventions, thrust bounds.
"""

import math

import numpy as np
import pytest

from competition.adapter import AttitudeCommand
from control.mpc_tracker import GeometricTracker, SimplePositionTracker, TrackerConfig
from planning.trajectory_optimizer import TrajectoryPoint


def _make_reference(
    position=(0, 0, 0),
    velocity=(0, 0, 0),
    acceleration=(0, 0, 0),
    yaw=0.0,
) -> TrajectoryPoint:
    return TrajectoryPoint(
        time=0.0,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        jerk=(0, 0, 0),
        yaw=yaw,
        yaw_rate=0.0,
    )


# ── GeometricTracker: hover (TIGHT thresholds — Phase 1/2 requirement) ───


class TestGeometricTrackerHover:
    """Hover tests with tight tolerances. These MUST catch the atan2 bug."""

    def test_zero_error_returns_attitude_command(self):
        """At the target with zero velocity -> produces an AttitudeCommand."""
        tracker = GeometricTracker()
        ref = _make_reference(position=(0, 0, 0), velocity=(0, 0, 0))
        cmd = tracker.track(
            current_position=(0, 0, 0),
            current_velocity=(0, 0, 0),
            current_yaw=0.0,
            reference=ref,
        )
        assert isinstance(cmd, AttitudeCommand)

    def test_hover_roll_near_zero(self):
        """CRITICAL: hover roll must be ~0, not pi. Catches the atan2(0,-1) bug."""
        tracker = GeometricTracker()
        ref = _make_reference()
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert abs(cmd.roll_rad) < 0.01, (
            f"Hover roll = {cmd.roll_rad:.4f} rad ({math.degrees(cmd.roll_rad):.1f} deg). "
            f"Expected ~0. Check atan2 formula in track()."
        )

    def test_hover_pitch_near_zero(self):
        """Hover pitch must be ~0."""
        tracker = GeometricTracker()
        ref = _make_reference()
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert abs(cmd.pitch_rad) < 0.01, (
            f"Hover pitch = {cmd.pitch_rad:.4f} rad. Expected ~0."
        )

    def test_hover_thrust_equals_mg(self):
        """Hover thrust = mg / max_thrust (within tight tolerance)."""
        cfg = TrackerConfig(mass=1.0, gravity=9.81, max_thrust_n=20.0)
        tracker = GeometricTracker(config=cfg)
        ref = _make_reference()
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        expected = cfg.mass * cfg.gravity / cfg.max_thrust_n
        assert cmd.thrust == pytest.approx(expected, abs=0.05), (
            f"Hover thrust = {cmd.thrust:.4f}, expected {expected:.4f}"
        )

    def test_hover_at_altitude(self):
        """Hover at non-zero altitude (z=-5 in NED = 5m above ground)."""
        tracker = GeometricTracker()
        ref = _make_reference(position=(0, 0, -5))
        cmd = tracker.track((0, 0, -5), (0, 0, 0), 0.0, ref)
        assert abs(cmd.roll_rad) < 0.01
        assert abs(cmd.pitch_rad) < 0.01

    def test_hover_with_nonzero_yaw(self):
        """Hover with a yaw heading should still have zero roll/pitch."""
        for yaw in [0.5, 1.0, math.pi / 2, math.pi, -1.0]:
            tracker = GeometricTracker()
            ref = _make_reference(yaw=yaw)
            cmd = tracker.track((0, 0, 0), (0, 0, 0), yaw, ref)
            assert abs(cmd.roll_rad) < 0.01, f"Roll != 0 at yaw={yaw}"
            assert abs(cmd.pitch_rad) < 0.01, f"Pitch != 0 at yaw={yaw}"

    def test_yaw_tracks_reference(self):
        tracker = GeometricTracker()
        ref = _make_reference(yaw=1.0)
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert cmd.yaw_rad == pytest.approx(1.0)


# ── Sign convention tests (Phase 2 requirement) ─────────────────────────


class TestGeometricTrackerSigns:
    """Verify correct sign conventions for NED/FRD frame contract."""

    def test_forward_acceleration_gives_negative_pitch(self):
        """
        +X acceleration (north) requires negative pitch (nose down in NED/FRD).
        Nose down tilts thrust forward.
        """
        tracker = GeometricTracker()
        ref = _make_reference(position=(10, 0, 0))  # target ahead in +x
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert cmd.pitch_rad < -0.01, (
            f"Forward (+X) error should give negative pitch, got {cmd.pitch_rad:.4f}"
        )

    def test_backward_acceleration_gives_positive_pitch(self):
        """
        -X acceleration (south) requires positive pitch (nose up in NED/FRD).
        """
        tracker = GeometricTracker()
        ref = _make_reference(position=(-10, 0, 0))  # target behind in -x
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert cmd.pitch_rad > 0.01, (
            f"Backward (-X) error should give positive pitch, got {cmd.pitch_rad:.4f}"
        )

    def test_rightward_acceleration_gives_positive_roll(self):
        """
        +Y acceleration (east) requires positive roll (right side down in FRD).
        Right side down tilts thrust eastward.
        """
        tracker = GeometricTracker()
        ref = _make_reference(position=(0, 10, 0))  # target to the right in +y
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert cmd.roll_rad > 0.01, (
            f"Rightward (+Y) error should give positive roll, got {cmd.roll_rad:.4f}"
        )

    def test_leftward_acceleration_gives_negative_roll(self):
        """
        -Y acceleration (west) requires negative roll (left side down in FRD).
        """
        tracker = GeometricTracker()
        ref = _make_reference(position=(0, -10, 0))  # target to the left in -y
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert cmd.roll_rad < -0.01, (
            f"Leftward (-Y) error should give negative roll, got {cmd.roll_rad:.4f}"
        )

    def test_yaw_90_forward_world_error_uses_roll_not_pitch(self):
        """At yaw=90°, +X world error is left in body frame -> negative roll."""
        tracker = GeometricTracker()
        ref = _make_reference(position=(10, 0, 0), yaw=math.pi / 2)
        cmd = tracker.track((0, 0, 0), (0, 0, 0), math.pi / 2, ref)
        assert cmd.roll_rad < -0.01
        assert abs(cmd.pitch_rad) < 0.01

    def test_yaw_180_forward_world_error_is_not_inverted(self):
        """At yaw=180°, +X world error is behind body frame -> positive pitch."""
        tracker = GeometricTracker()
        ref = _make_reference(position=(10, 0, 0), yaw=math.pi)
        cmd = tracker.track((0, 0, 0), (0, 0, 0), math.pi, ref)
        assert cmd.pitch_rad > 0.01

    def test_downward_error_reduces_thrust_without_tilt_flip(self):
        """
        Target below drone (larger z in NED) -> reduce thrust, not invert
        the thrust vector and bank sideways.
        """
        tracker = GeometricTracker()
        ref = _make_reference(position=(0, 0, 5))  # target below (NED: +z = down)
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        cfg = TrackerConfig()
        hover_thrust = cfg.mass * cfg.gravity / cfg.max_thrust_n
        assert cmd.thrust <= hover_thrust
        assert abs(cmd.roll_rad) < 0.01
        assert abs(cmd.pitch_rad) < 0.01

    def test_upward_error_increases_thrust(self):
        """
        Target above drone (smaller z in NED) -> MORE thrust to climb.
        In NED, ascending = decreasing z = thrust must exceed hover to accelerate upward.
        """
        tracker = GeometricTracker()
        ref = _make_reference(position=(0, 0, -5))  # target above (NED: -z = up)
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        cfg = TrackerConfig()
        hover_thrust = cfg.mass * cfg.gravity / cfg.max_thrust_n
        assert cmd.thrust > hover_thrust, (
            f"Upward error should increase thrust above hover ({hover_thrust:.3f}), got {cmd.thrust:.3f}"
        )


# ── Thrust computation ───────────────────────────────────────────────────


class TestThrustComputation:
    def test_hover_thrust_approx_mg_over_max(self):
        cfg = TrackerConfig(mass=1.0, gravity=9.81, max_thrust_n=20.0)
        tracker = GeometricTracker(config=cfg)
        ref = _make_reference()
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        expected = cfg.mass * cfg.gravity / cfg.max_thrust_n
        assert cmd.thrust == pytest.approx(expected, abs=0.05)

    def test_heavy_drone_more_thrust(self):
        cfg_light = TrackerConfig(mass=0.5, max_thrust_n=20.0)
        cfg_heavy = TrackerConfig(mass=2.0, max_thrust_n=20.0)
        t_light = GeometricTracker(config=cfg_light)
        t_heavy = GeometricTracker(config=cfg_heavy)
        ref = _make_reference()
        cmd_light = t_light.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        cmd_heavy = t_heavy.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert cmd_heavy.thrust > cmd_light.thrust

    def test_thrust_clamped_to_bounds(self):
        cfg = TrackerConfig(min_thrust_normalized=0.05, max_thrust_normalized=0.95)
        tracker = GeometricTracker(config=cfg)
        ref = _make_reference(position=(0, 0, 100))
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert cmd.thrust <= cfg.max_thrust_normalized
        assert cmd.thrust >= cfg.min_thrust_normalized


# ── Attitude limits ──────────────────────────────────────────────────────


class TestAttitudeLimits:
    def test_roll_within_limits(self):
        cfg = TrackerConfig(max_tilt_rad=0.7)
        tracker = GeometricTracker(config=cfg)
        ref = _make_reference(position=(0, 100, 0))
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert abs(cmd.roll_rad) <= cfg.max_tilt_rad + 1e-6

    def test_pitch_within_limits(self):
        cfg = TrackerConfig(max_tilt_rad=0.7)
        tracker = GeometricTracker(config=cfg)
        ref = _make_reference(position=(100, 0, 0))
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert abs(cmd.pitch_rad) <= cfg.max_tilt_rad + 1e-6

    def test_extreme_error_still_within_limits(self):
        cfg = TrackerConfig(max_tilt_rad=0.5)
        tracker = GeometricTracker(config=cfg)
        ref = _make_reference(position=(1000, 1000, 0))
        cmd = tracker.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        assert abs(cmd.roll_rad) <= cfg.max_tilt_rad + 1e-6
        assert abs(cmd.pitch_rad) <= cfg.max_tilt_rad + 1e-6


# ── SimplePositionTracker cross-validation ───────────────────────────────


class TestSimplePositionTracker:
    def test_hover_at_target(self):
        tracker = SimplePositionTracker()
        cmd = tracker.track(
            current_position=(0, 0, 0),
            current_velocity=(0, 0, 0),
            current_yaw=0.0,
            target_position=(0, 0, 0),
        )
        assert isinstance(cmd, AttitudeCommand)
        assert abs(cmd.roll_rad) < 0.01
        assert abs(cmd.pitch_rad) < 0.01

    def test_offset_produces_correction(self):
        tracker = SimplePositionTracker()
        cmd = tracker.track(
            current_position=(0, 0, 0),
            current_velocity=(0, 0, 0),
            current_yaw=0.0,
            target_position=(5, 0, 0),
        )
        assert abs(cmd.pitch_rad) > 0.01

    def test_forward_error_yaw0_commands_negative_pitch(self):
        tracker = SimplePositionTracker()
        cmd = tracker.track(
            current_position=(0, 0, 0),
            current_velocity=(0, 0, 0),
            current_yaw=0.0,
            target_position=(5, 0, 0),
        )
        assert cmd.pitch_rad < -0.01

    def test_right_world_error_at_yaw90_commands_negative_pitch(self):
        tracker = SimplePositionTracker()
        cmd = tracker.track(
            current_position=(0, 0, 0),
            current_velocity=(0, 0, 0),
            current_yaw=math.pi / 2,
            target_position=(0, 5, 0),
        )
        assert cmd.pitch_rad < -0.01
        assert abs(cmd.roll_rad) < 0.01

    def test_reset_clears_integral(self):
        tracker = SimplePositionTracker()
        for _ in range(10):
            tracker.track((0, 0, 0), (0, 0, 0), 0.0, (0, 0, 5), dt=0.01)
        tracker.reset()
        assert tracker._integral_z == 0.0

    def test_thrust_bounds(self):
        cfg = TrackerConfig(min_thrust_normalized=0.05, max_thrust_normalized=0.95)
        tracker = SimplePositionTracker(config=cfg)
        cmd = tracker.track(
            (0, 0, 0), (0, 0, 0), 0.0,
            target_position=(0, 0, 100),
        )
        assert cfg.min_thrust_normalized <= cmd.thrust <= cfg.max_thrust_normalized


# ── Cross-validation: GeometricTracker vs SimplePositionTracker ──────────


class TestDragAwareFeedforward:
    """Drag-aware differential-flatness feedforward (Faessler RA-L 2018,
    roadmap #1). OFF by default behind `use_drag_ff`; reduces to baseline
    exactly when off or when drag_ff_coeff == 0 (regression-safe), and cuts
    steady-state tracking error on a synthetic constant-velocity drag case.

    NOTE: validated here only via unit tests + the kinematic-rollout below
    (which mirrors scripts/benchmark.py's `accel = accel_des - drag·vel`).
    NOT validated on DCGame — no sim/DCL binary on this worktree.
    """

    def _rollout_cross_track_error(
        self, tracker: GeometricTracker, drag: float,
        speed: float = 8.0, dt: float = 0.01, n_steps: int = 400,
    ) -> float:
        """Fly a constant-velocity +Y reference through a drag field and
        return the mean lateral (cross-track) position error.

        Bench-faithful kinematics: realized accel = accel_des - drag·vel
        (scripts/benchmark.py:653). The reference moves at constant `speed`
        along +Y; with linear drag the un-compensated tracker settles at a
        steady lag, which the drag-FF should largely cancel.
        """
        pos = np.zeros(3)
        vel = np.zeros(3)
        errs = []
        for k in range(n_steps):
            t = k * dt
            ref_y = speed * t
            ref = _make_reference(
                position=(0.0, ref_y, 0.0),
                velocity=(0.0, speed, 0.0),
                acceleration=(0.0, 0.0, 0.0),
            )
            tracker.track(tuple(pos), tuple(vel), 0.0, ref)
            accel_des = tracker.last_desired_acceleration
            accel = np.array(accel_des) - drag * vel
            vel = vel + accel * dt
            pos = pos + vel * dt
            if t > 2.0:  # measure after the startup transient
                errs.append(abs(ref_y - pos[1]))
        return float(np.mean(errs))

    def test_drag_ff_off_is_baseline(self):
        """use_drag_ff=False must be byte-for-byte the baseline tracker."""
        ref = _make_reference(
            position=(3.0, -2.0, -1.0), velocity=(5.0, -4.0, 1.0),
            acceleration=(2.0, 1.0, -0.5), yaw=0.7,
        )
        base = GeometricTracker(TrackerConfig())
        # Flag off but coeff non-zero — must STILL be baseline (flag gates it).
        off = GeometricTracker(TrackerConfig(use_drag_ff=False, drag_ff_coeff=0.5))
        cb = base.track((1.0, 0.5, -0.5), (1.0, -1.0, 0.5), 0.3, ref)
        co = off.track((1.0, 0.5, -0.5), (1.0, -1.0, 0.5), 0.3, ref)
        assert co.roll_rad == cb.roll_rad
        assert co.pitch_rad == cb.pitch_rad
        assert co.thrust == cb.thrust
        np.testing.assert_array_equal(
            off.last_desired_acceleration, base.last_desired_acceleration
        )

    def test_drag_ff_zero_coeff_is_baseline(self):
        """Flag on but coeff 0 reduces EXACTLY to current behaviour."""
        ref = _make_reference(
            position=(3.0, -2.0, -1.0), velocity=(5.0, -4.0, 1.0),
            acceleration=(2.0, 1.0, -0.5), yaw=0.7,
        )
        base = GeometricTracker(TrackerConfig())
        zero = GeometricTracker(TrackerConfig(use_drag_ff=True, drag_ff_coeff=0.0))
        cb = base.track((1.0, 0.5, -0.5), (1.0, -1.0, 0.5), 0.3, ref)
        cz = zero.track((1.0, 0.5, -0.5), (1.0, -1.0, 0.5), 0.3, ref)
        assert cz.roll_rad == cb.roll_rad
        assert cz.pitch_rad == cb.pitch_rad
        assert cz.thrust == cb.thrust
        np.testing.assert_array_equal(
            zero.last_desired_acceleration, base.last_desired_acceleration
        )

    def test_drag_ff_adds_exact_term_to_accel_des(self):
        """When on, accel_des gains EXACTLY drag_ff_coeff·ref_vel."""
        ref = _make_reference(
            position=(0.0, 0.0, 0.0), velocity=(6.0, -3.0, 1.0),
            acceleration=(0.0, 0.0, 0.0), yaw=0.0,
        )
        coeff = 0.5
        base = GeometricTracker(TrackerConfig())
        ff = GeometricTracker(TrackerConfig(use_drag_ff=True, drag_ff_coeff=coeff))
        # Zero state error so only the FF term differs; small velocities keep
        # the result well clear of the lateral-accel / thrust clamps.
        base.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        ff.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        delta = ff.last_desired_acceleration - base.last_desired_acceleration
        expected = coeff * np.array(ref.velocity)
        np.testing.assert_allclose(delta, expected, atol=1e-9)

    def test_drag_ff_reduces_constant_velocity_tracking_error(self):
        """On a constant-velocity drag case, drag-FF cuts steady-state error.

        Setting drag_ff_coeff to the field's true drag makes the FF cancel
        the drag forcing, so the tracked lag shrinks substantially vs the
        un-compensated baseline.
        """
        drag = 0.5  # matches drone_spec.DEFAULT_LINEAR_DRAG_PER_MASS
        base = GeometricTracker(TrackerConfig())
        ff = GeometricTracker(TrackerConfig(use_drag_ff=True, drag_ff_coeff=drag))
        err_base = self._rollout_cross_track_error(base, drag)
        err_ff = self._rollout_cross_track_error(ff, drag)
        assert err_base > 0.01, (
            f"baseline drag lag too small to be a meaningful test ({err_base:.4f} m)"
        )
        assert err_ff < 0.5 * err_base, (
            f"drag-FF did not reduce tracking error: baseline {err_base:.4f} m, "
            f"FF {err_ff:.4f} m"
        )


class TestCrossValidation:
    """Both trackers should agree on basic maneuvers (sanity check)."""

    def test_hover_agreement(self):
        """Both produce near-zero roll/pitch at hover."""
        geo = GeometricTracker()
        simple = SimplePositionTracker()
        ref = _make_reference()
        cmd_geo = geo.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        cmd_simple = simple.track((0, 0, 0), (0, 0, 0), 0.0, (0, 0, 0))
        assert abs(cmd_geo.roll_rad) < 0.01
        assert abs(cmd_simple.roll_rad) < 0.01
        assert abs(cmd_geo.pitch_rad) < 0.01
        assert abs(cmd_simple.pitch_rad) < 0.01

    def test_forward_error_both_correct(self):
        """Both produce nonzero pitch for forward error."""
        geo = GeometricTracker()
        simple = SimplePositionTracker()
        ref = _make_reference(position=(10, 0, 0))
        cmd_geo = geo.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        cmd_simple = simple.track((0, 0, 0), (0, 0, 0), 0.0, (10, 0, 0))
        # GeometricTracker: negative pitch for forward (nose down in NED/FRD)
        assert cmd_geo.pitch_rad < -0.01, (
            f"GeometricTracker forward pitch should be negative, got {cmd_geo.pitch_rad:.3f}"
        )
        # SimplePositionTracker uses opposite pitch convention — just verify it responds
        assert abs(cmd_simple.pitch_rad) > 0.01

    def test_lateral_error_both_correct(self):
        """Both produce nonzero roll for lateral error."""
        geo = GeometricTracker()
        simple = SimplePositionTracker()
        ref = _make_reference(position=(0, 10, 0))
        cmd_geo = geo.track((0, 0, 0), (0, 0, 0), 0.0, ref)
        cmd_simple = simple.track((0, 0, 0), (0, 0, 0), 0.0, (0, 10, 0))
        # GeometricTracker: positive roll for eastward (right side down in FRD)
        assert cmd_geo.roll_rad > 0.01, (
            f"GeometricTracker lateral roll should be positive, got {cmd_geo.roll_rad:.3f}"
        )
        # SimplePositionTracker — just verify it responds
        assert abs(cmd_simple.roll_rad) > 0.01
