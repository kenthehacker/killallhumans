"""TRPYMixer correctness — focused on the inverted-attitude regression.

P0-3: the previous `max(cos_tilt, 0.3)` floor turned an inverted drone
(cos(roll)*cos(pitch) < 0) into MAX upward thrust toward the ground. The
inversion guard short-circuits to idle throttle so the recovery path can
flip the drone back over instead of accelerating into the dirt.
"""

import math
import unittest

from flight_control.mixer import MixerConfig, TRPYMixer
from flight_control.types import ControlCommand


def _hover_command() -> ControlCommand:
    """az=0 → just enough thrust to hover; ax=ay=0 → upright."""
    return ControlCommand(
        ax=0.0, ay=0.0, az=0.0, yaw_rate=0.0,
        desired_velocity=(0.0, 0.0, 0.0), desired_yaw=0.0,
    )


class TestMixerInversionGuard(unittest.TestCase):
    def test_inverted_attitude_idles_throttle(self) -> None:
        """current_roll=π puts the drone fully inverted; cos_tilt < 0.
        Throttle must NOT saturate to 1.0 — that would fire MAX thrust into
        the ground. Spec: throttle ≤ 0.1."""
        mixer = TRPYMixer(MixerConfig())
        cmd = mixer.mix(
            _hover_command(),
            current_roll=math.pi,   # fully inverted about +x
            current_pitch=0.0,
        )
        self.assertLessEqual(cmd.throttle, 0.1)

    def test_inverted_pitch_also_idles(self) -> None:
        """Pitched 180° → also inverted. Same guard."""
        mixer = TRPYMixer(MixerConfig())
        cmd = mixer.mix(
            _hover_command(),
            current_roll=0.0,
            current_pitch=math.pi,
        )
        self.assertLessEqual(cmd.throttle, 0.1)

    def test_steep_but_upright_still_thrusts(self) -> None:
        """At ~75° tilt, cos_tilt ≈ 0.26 — still upright; the controller
        must keep producing meaningful thrust so the drone can recover."""
        mixer = TRPYMixer(MixerConfig())
        cmd = mixer.mix(
            _hover_command(),
            current_roll=math.radians(75.0),
            current_pitch=0.0,
        )
        # Thrust should be significantly above idle (we want recovery thrust).
        self.assertGreater(cmd.throttle, 0.1)
        self.assertLessEqual(cmd.throttle, 1.0)

    def test_upright_hover_unchanged(self) -> None:
        """Sanity: at zero attitude, hover throttle = mass*g/max_thrust_n."""
        cfg = MixerConfig()
        mixer = TRPYMixer(cfg)
        cmd = mixer.mix(_hover_command(), 0.0, 0.0)
        expected = cfg.drone_mass_kg * cfg.gravity / cfg.max_thrust_n
        self.assertAlmostEqual(cmd.throttle, expected, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
