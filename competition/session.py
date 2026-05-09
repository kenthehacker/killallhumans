"""
Race session manager — orchestrates the competition lifecycle.

Handles:
  - Connection and initialization
  - Heartbeat maintenance (MAVSDK handles automatically at ≥2 Hz)
  - The race loop: perception → estimation → planning → control
  - Graceful shutdown and error recovery
  - Timing and performance monitoring

Competition constraints:
  - Max run duration: 8 minutes (480 seconds)
  - Physics: 120 Hz
  - Command rate: 50-120 Hz
  - Fully autonomous — no human interaction during timed runs
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

from .adapter import (
    AttitudeCommand,
    CameraFrame,
    CompetitionInterface,
    TelemetryState,
)

logger = logging.getLogger(__name__)

MAX_RUN_DURATION_S = 480  # 8 minutes per tech spec


class SessionState(Enum):
    IDLE = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ARMED = auto()
    RACING = auto()
    FINISHED = auto()
    ERROR = auto()


@dataclass
class RaceMetrics:
    """Performance metrics for a race run."""
    start_time: float = 0.0
    end_time: float = 0.0
    total_frames: int = 0
    control_loop_times: List[float] = field(default_factory=list)
    detection_times: List[float] = field(default_factory=list)
    gates_passed: int = 0
    total_gates: int = 0

    @property
    def elapsed_s(self) -> float:
        if self.start_time == 0:
            return 0.0
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time

    @property
    def avg_loop_hz(self) -> float:
        if not self.control_loop_times:
            return 0.0
        avg_dt = sum(self.control_loop_times) / len(self.control_loop_times)
        return 1.0 / max(avg_dt, 1e-9)

    @property
    def avg_detection_ms(self) -> float:
        if not self.detection_times:
            return 0.0
        return 1000.0 * sum(self.detection_times) / len(self.detection_times)


class RaceSession:
    """
    Manages a complete race session from connection to finish.

    The session orchestrates the high-level flow while delegating
    perception, planning, and control to pluggable callbacks.

    Usage:
        session = RaceSession(bridge)
        session.on_telemetry = my_control_callback
        await session.run()
    """

    def __init__(
        self,
        interface: CompetitionInterface,
        target_hz: float = 100.0,
        address: str = "udp://:14540",
    ):
        self.interface = interface
        self.target_hz = target_hz
        self.target_dt = 1.0 / target_hz
        self.address = address
        self.state = SessionState.IDLE
        self.metrics = RaceMetrics()

        # Callbacks — set these before calling run()
        self.on_telemetry: Optional[
            Callable[[TelemetryState, Optional[CameraFrame]], Optional[AttitudeCommand]]
        ] = None
        self.on_race_complete: Optional[Callable[[RaceMetrics], None]] = None
        self.should_stop: Optional[Callable[[], bool]] = None

        self._stop_event = asyncio.Event()

    async def run(self) -> RaceMetrics:
        """
        Execute the full race session.

        Returns metrics when the race is complete or timed out.
        """
        try:
            # Phase 1: Connect
            self.state = SessionState.CONNECTING
            await self.interface.connect(self.address)
            self.state = SessionState.CONNECTED
            logger.info("Session connected")

            # Phase 2: Arm
            await self.interface.arm()
            self.state = SessionState.ARMED
            logger.info("Vehicle armed")

            # Phase 3: Start offboard and race
            await self.interface.start_offboard()
            self.state = SessionState.RACING
            self.metrics.start_time = time.time()
            logger.info("Race started!")

            await self._race_loop()

        except Exception as e:
            logger.error("Session error: %s", e)
            self.state = SessionState.ERROR
            raise
        finally:
            self.metrics.end_time = time.time()
            await self._cleanup()

        self.state = SessionState.FINISHED
        logger.info(
            "Race complete: %.1fs, %d gates, avg %.0f Hz",
            self.metrics.elapsed_s,
            self.metrics.gates_passed,
            self.metrics.avg_loop_hz,
        )

        if self.on_race_complete:
            self.on_race_complete(self.metrics)

        return self.metrics

    async def _race_loop(self) -> None:
        """Main control loop running at target_hz."""
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            # Check timeout
            if self.metrics.elapsed_s >= MAX_RUN_DURATION_S:
                logger.warning("Race timeout reached (%.0fs)", MAX_RUN_DURATION_S)
                break

            # Check external stop condition
            if self.should_stop and self.should_stop():
                logger.info("External stop condition triggered")
                break

            # Get telemetry and camera
            try:
                telem = await self.interface.get_telemetry()
                frame = await self.interface.get_camera_frame()
            except RuntimeError:
                # No telemetry yet, wait
                await asyncio.sleep(0.01)
                continue

            # Invoke control callback
            if self.on_telemetry is not None:
                cmd = self.on_telemetry(telem, frame)
                if cmd is not None:
                    await self.interface.send_attitude(cmd)

            self.metrics.total_frames += 1

            # Timing
            loop_dt = time.perf_counter() - loop_start
            self.metrics.control_loop_times.append(loop_dt)

            # Rate limiting — sleep to maintain target_hz
            sleep_time = self.target_dt - loop_dt
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _cleanup(self) -> None:
        """Graceful shutdown."""
        try:
            await self.interface.stop_offboard()
        except Exception:
            pass
        try:
            await self.interface.disconnect()
        except Exception:
            pass

    def stop(self) -> None:
        """Signal the race loop to stop."""
        self._stop_event.set()
