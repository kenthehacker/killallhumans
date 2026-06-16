"""Sim-degradation / health-probe monitor (race-day reliability item 1, half 2).

The companion to :mod:`competition.gate_map_integrity`. That module catches the
gate-map CORRUPTION half of the sim's after-~25-runs degradation; THIS module
catches the FLIGHT-DYNAMICS half — the symptom that shows up in telemetry once
the DCGame process has degraded but is still serving a sane gate map.

Documented signature (see ``docs/aigp/2026-06-16-speed-and-spline-handoff.md``,
"Operational notes (race day)"):

* The DCGame process degrades after ~25 runs/session.
* The healthy START OVER-CLIMB — the most-negative Z a fresh run reaches in its
  first few seconds after GO — is ~-1.7 m. As the process degrades, that
  over-climb GROWS: a degraded run climbs to Z <= ~-2.4 m.
* HEALTH PROBE (verbatim from the handoff): "a fresh run's first-3 s peak climb
  should be Z ~= -1.7". "If a run's start-climb is <= -2.4 m OR the collision
  count is wildly high with a clean trajectory, suspect degradation and restart."
* A degraded PROCESS needs a full .exe restart into VQ mode; a per-run
  SIM_RESET does NOT fix it. So a degraded verdict's action is "restart the
  .exe", not "reset".

CRITICAL SIGN CONVENTION (do not flip): the frame is NED, so Z is DOWN-positive
and CLIMB is NEGATIVE Z. "Peak climb" is therefore the MOST-NEGATIVE Z reached,
i.e. ``min(z)`` over the window. Degraded => peak climb Z <= ~-2.4 (the drone
climbs HIGHER / more negative than the healthy ~-1.7). A drone that DESCENDS
(positive Z) is the opposite of over-climb and must NEVER trip this probe.

Design constraints (CLAUDE.md), mirrored from ``gate_map_integrity``:
pure-python / numpy-only, **no new dependencies and NO MAVLink import**, so it
unit-tests fully offline. The verdict NEVER raises — a degraded sim (or missing
data) is *diagnosed*, not thrown. Thresholds are documented PARAMETERS anchored
on the handoff's -1.7 / -2.4 numbers, not magic literals buried in the logic.

Two entry points:

* :func:`probe_health` — pure batch function over a sequence of ``(t_s, z_ned)``
  samples (offline-testable; the single source of truth for the verdict logic).
* :class:`SimHealthProbe` — a streaming wrapper a real-time runner feeds one
  sample at a time (``add_sample`` / ``add_collisions``) and evaluates ONCE the
  window has elapsed, so the runner needn't buffer the whole flight. It delegates
  to :func:`probe_health`, so the streaming and batch verdicts are identical on
  the same data (asserted in the tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds & window constants  (documented rationale, not magic numbers)
# ---------------------------------------------------------------------------
# All of these are exposed as keyword PARAMETERS of ``probe_health`` /
# ``SimHealthProbe`` so they can be tuned without editing the logic; the module
# constants below are the defaults, anchored directly on the handoff numbers.
#
# HEALTHY_REF_Z — the healthy fresh-run first-window peak climb, verbatim from
#   the handoff ("a fresh run's first-3 s peak climb should be Z ~= -1.7"). NED,
#   so -1.7 m is 1.7 m ABOVE the spawn (climb = negative Z). Used only as the
#   reference the message quotes ("healthy ~-1.7"); it is NOT itself a pass/fail
#   gate (the gate is PEAK_CLIMB_WARN_Z below).
HEALTHY_REF_Z: float = -1.7

# PEAK_CLIMB_WARN_Z — the degraded threshold, verbatim from the handoff
#   ("degraded <= -2.4 m" / "if a run's start-climb is <= -2.4 m ... suspect
#   degradation"). A peak climb AT OR BELOW (more negative than) this trips
#   "over_climb". The 0.7 m gap to the healthy -1.7 is the handoff's own
#   healthy/degraded separation, generous enough that ordinary run-to-run climb
#   jitter (a fresh run varies a few tenths of a metre around -1.7) does NOT
#   false-trip, while the degraded process's grown over-climb does. The boundary
#   is INCLUSIVE on the degraded side (z <= thr is degraded) so a peak exactly at
#   -2.4 is flagged and a peak just shallower (e.g. -2.399) stays healthy.
PEAK_CLIMB_WARN_Z: float = -2.4

# HEALTH_WINDOW_S — the observation window after the first post-GO sample,
#   verbatim from the handoff's "first-3 s peak climb". The start over-climb is a
#   transient that peaks within the first couple of seconds of the climb-out;
#   3 s captures it with margin while staying well inside even the fastest lap
#   (~18 s), so the probe fires and is DONE long before the finish.
HEALTH_WINDOW_S: float = 3.0

# MAX_EARLY_COLLISIONS — the collision-count trigger for the handoff's other
#   degradation tell: "the collision count is wildly high with a clean
#   trajectory". On a HEALTHY run the validated champions log ZERO collisions
#   over a full clean lap, so a handful of collisions in just the first ~3 s is
#   already anomalous. We set the threshold at 3: 1-2 early contacts can be a
#   benign grazing start, but >3 collisions before the drone has even finished
#   its climb-out is the "wildly high" signature. This is intentionally a
#   COUNT within the window, not a rate, to match how the adapter exposes
#   collisions (a drained deque count). Exceeding it (strictly greater than)
#   trips "excessive_collisions".
MAX_EARLY_COLLISIONS: int = 3

# MIN_SAMPLES — the fewest in-window samples we will render a healthy/degraded
#   verdict on. The live loop runs ~100 Hz, so 3 s is ~300 samples; we require a
#   small floor (5) so a probe that fired with almost nothing (a stalled
#   telemetry feed, an aborted run, --dry-run's single static frame) returns the
#   distinct, NON-alarming "insufficient_data" rather than a spurious "healthy"
#   or "over_climb" read off one point.
MIN_SAMPLES: int = 5


# Allowed diagnosis codes (frozenset so callers/tests can assert against the
# canonical set without sprinkling string literals — mirrors gate_map_integrity).
DIAGNOSES = frozenset(
    {
        "healthy",
        "over_climb",
        "excessive_collisions",
        "insufficient_data",
    }
)


# A single flight sample: (time_seconds, z_ned_metres). z is the RAW NED Z from
# ``TelemetryState.position_ned[2]`` — down-positive, so climb is negative.
Sample = Tuple[float, float]


@dataclass
class SimHealthVerdict:
    """Result of a sim health probe. NEVER carries/raises an exception.

    A degraded sim (or too little data) is *diagnosed*, not thrown — mirrors
    :class:`competition.gate_map_integrity.GateMapVerdict`.

    Attributes:
        healthy: the single bool the runner keys off. ``False`` for BOTH a real
            degradation (``over_climb`` / ``excessive_collisions``) AND for
            ``insufficient_data`` — but the latter is a distinct, non-alarming
            diagnosis (no data is not the same as a degraded sim), so callers
            that want to warn ONLY on true degradation can gate on
            ``verdict.degraded`` instead.
        diagnosis: one of :data:`DIAGNOSES`.
        message: human-readable, naming the signal and (when degraded) the
            action — a full .exe restart, NOT a SIM_RESET.
        details: structured diagnostics for logging / post-run capture
            (peak_climb_z, window_s, n_samples, collisions, threshold). Never
            load-bearing for the bool contract.
    """

    healthy: bool
    diagnosis: str
    message: str
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.diagnosis not in DIAGNOSES:
            # Defensive: a typo'd diagnosis code is a bug, but this monitor must
            # never raise from the happy path. Surface it in the message.
            self.message = f"[bad diagnosis code {self.diagnosis!r}] " + self.message

    @property
    def degraded(self) -> bool:
        """True iff this is a genuine DEGRADATION verdict (not insufficient data).

        ``insufficient_data`` is ``healthy=False`` but ``degraded=False`` — it is
        the non-alarming "couldn't tell" state, so the runner does not shout
        "restart the .exe" at a --dry-run / stalled-feed probe.
        """
        return self.diagnosis in ("over_climb", "excessive_collisions")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _window_samples(
    samples: Sequence[Sample], window_s: float
) -> Optional[np.ndarray]:
    """Return the (M, 2) array of (t, z) samples within ``window_s`` of the
    FIRST sample's timestamp, or ``None`` if the input is unusable.

    Never raises: a malformed sample (wrong length, non-numeric, non-finite t/z)
    makes the whole input unusable -> ``None`` -> the caller diagnoses it as
    insufficient_data rather than crashing.
    """
    try:
        rows = []
        for s in samples:
            t, z = s[0], s[1]
            t, z = float(t), float(z)
            if not (np.isfinite(t) and np.isfinite(z)):
                # Skip a single bad reading rather than discarding the whole
                # window (telemetry occasionally emits a NaN frame).
                continue
            rows.append((t, z))
        if not rows:
            return None
        arr = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError, IndexError):
        return None
    t0 = arr[0, 0]
    return arr[arr[:, 0] <= t0 + window_s]


# ---------------------------------------------------------------------------
# Primary entry point (pure batch)
# ---------------------------------------------------------------------------

def probe_health(
    samples: Sequence[Sample],
    *,
    collisions: int = 0,
    peak_climb_warn_z: float = PEAK_CLIMB_WARN_Z,
    healthy_ref_z: float = HEALTHY_REF_Z,
    window_s: float = HEALTH_WINDOW_S,
    max_early_collisions: int = MAX_EARLY_COLLISIONS,
    min_samples: int = MIN_SAMPLES,
) -> SimHealthVerdict:
    """Diagnose sim health from the first ``window_s`` of post-GO flight.

    NEVER raises — returns a :class:`SimHealthVerdict`.

    Args:
        samples: sequence of ``(t_s, z_ned)`` flight samples. ``t_s`` is seconds
            (any monotonic clock; only differences from the first sample matter);
            ``z_ned`` is the raw NED Z (down-positive, climb negative) from
            ``TelemetryState.position_ned[2]``.
        collisions: number of collisions observed within the window (e.g. the
            count drained from the adapter's collision deque since GO).
        peak_climb_warn_z: degraded threshold (NED Z). A peak climb at/below this
            (more negative) is ``over_climb``. Default :data:`PEAK_CLIMB_WARN_Z`
            = -2.4 (handoff).
        healthy_ref_z: the healthy reference peak climb quoted in the message.
            Default :data:`HEALTHY_REF_Z` = -1.7 (handoff). Not a gate itself.
        window_s: observation window after the first sample. Default
            :data:`HEALTH_WINDOW_S` = 3.0 s (handoff "first-3 s").
        max_early_collisions: collisions strictly above this within the window
            trip ``excessive_collisions``. Default :data:`MAX_EARLY_COLLISIONS`.
        min_samples: floor on in-window samples to render a real verdict; below
            it -> ``insufficient_data``. Default :data:`MIN_SAMPLES`.

    Logic (order matters — most fundamental first):
        1. too little usable data / window not covered -> ``insufficient_data``
           (healthy=False but a DISTINCT, non-alarming diagnosis);
        2. peak climb (``min(z)``) <= ``peak_climb_warn_z`` -> ``over_climb``;
        3. ``collisions > max_early_collisions`` -> ``excessive_collisions``;
        4. otherwise -> ``healthy``.

    Why over_climb is checked before collisions: the over-climb is the PRIMARY,
    most-specific degradation signature in the handoff ("start over-climb
    grows"); the collision count is the secondary "OR ... wildly high" tell. When
    both fire we report the primary one (and still record the collision count in
    ``details`` for post-run analysis).
    """
    win = _window_samples(samples, window_s)

    # 1. Insufficient data: no usable samples, too few, or the window is not yet
    #    covered (the last sample doesn't reach first + window_s). All three are
    #    "we can't tell" — healthy=False, but the non-alarming diagnosis so the
    #    runner does not cry "restart" on a dry run / stalled feed / early abort.
    if win is None or win.shape[0] < min_samples:
        n = 0 if win is None else int(win.shape[0])
        return SimHealthVerdict(
            healthy=False,
            diagnosis="insufficient_data",
            message=(
                f"sim health UNKNOWN: only {n} usable sample(s) in the first "
                f"{window_s:.1f}s (need >= {min_samples}) — not enough to judge "
                f"climb/collisions (no degradation inferred)"
            ),
            details={
                "peak_climb_z": None,
                "window_s": window_s,
                "n_samples": n,
                "collisions": int(collisions),
            },
        )

    z = win[:, 1]
    covered_s = float(win[-1, 0] - win[0, 0])
    # Coverage check with a discrete-sampling SLACK. A real-time feed samples at
    # a fixed rate, so the last sample that lands inside the window sits up to one
    # inter-sample gap SHORT of window_s (e.g. at 100 Hz the last in-window sample
    # is at ~t0+2.99 s for a 3 s window). Requiring covered_s >= window_s exactly
    # would then spuriously report "insufficient_data" on a perfectly healthy full
    # window. So we treat the window as covered if the samples reach within ~one
    # median inter-sample gap (rate-agnostic, with a small floor) of the end.
    if win.shape[0] >= 2:
        gaps = np.diff(win[:, 0])
        median_gap = float(np.median(gaps)) if gaps.size else 0.0
    else:
        median_gap = 0.0
    # Floor the slack at 50 ms (a 20 Hz feed) so a slow/jittery sampler isn't
    # punished; never let slack exceed 1/3 of the window (a sparse feed really is
    # insufficient).
    coverage_slack_s = min(max(median_gap, 0.05), window_s / 3.0)
    if covered_s < window_s - coverage_slack_s:
        # We have >= min_samples but they span meaningfully less than the full
        # window (e.g. the run ended early). Non-alarming verdict, with the
        # partial span surfaced so a reviewer sees WHY.
        return SimHealthVerdict(
            healthy=False,
            diagnosis="insufficient_data",
            message=(
                f"sim health UNKNOWN: samples span only {covered_s:.2f}s of the "
                f"{window_s:.1f}s window ({int(win.shape[0])} samples) — window "
                f"not fully covered (no degradation inferred)"
            ),
            details={
                "peak_climb_z": float(np.min(z)),
                "window_s": window_s,
                "covered_s": round(covered_s, 3),
                "n_samples": int(win.shape[0]),
                "collisions": int(collisions),
            },
        )

    # Peak climb = the MOST-NEGATIVE Z (highest above spawn) in the window.
    # Sign convention: climb is negative Z, so the peak is min(z), NOT max(z).
    peak_climb_z = float(np.min(z))
    n = int(win.shape[0])
    details = {
        "peak_climb_z": peak_climb_z,
        "window_s": window_s,
        "n_samples": n,
        "collisions": int(collisions),
        "peak_climb_warn_z": peak_climb_warn_z,
        "healthy_ref_z": healthy_ref_z,
    }

    # 2. Over-climb (PRIMARY degradation signature). z <= thr (inclusive on the
    #    degraded side) => the start over-climb has grown past the healthy ~-1.7
    #    toward / past the -2.4 degraded threshold.
    if peak_climb_z <= peak_climb_warn_z:
        return SimHealthVerdict(
            healthy=False,
            diagnosis="over_climb",
            message=(
                f"sim likely DEGRADED: first-{window_s:.0f}s peak climb "
                f"Z={peak_climb_z:.2f} <= {peak_climb_warn_z:.1f} "
                f"(healthy ~{healthy_ref_z:.1f}) — restart the DCGame .exe into "
                f"VQ mode; a SIM_RESET will NOT fix this"
            ),
            details=details,
        )

    # 3. Excessive early collisions (SECONDARY tell — "wildly high with a clean
    #    trajectory"). Strictly greater than the threshold.
    if int(collisions) > max_early_collisions:
        return SimHealthVerdict(
            healthy=False,
            diagnosis="excessive_collisions",
            message=(
                f"sim likely DEGRADED: {int(collisions)} collisions in the first "
                f"{window_s:.0f}s (> {max_early_collisions}) while climb looks "
                f"clean (peak Z={peak_climb_z:.2f}) — restart the DCGame .exe "
                f"into VQ mode; a SIM_RESET will NOT fix this"
            ),
            details=details,
        )

    # 4. Healthy.
    return SimHealthVerdict(
        healthy=True,
        diagnosis="healthy",
        message=(
            f"sim health OK: first-{window_s:.0f}s peak climb Z={peak_climb_z:.2f} "
            f"(healthy ~{healthy_ref_z:.1f}, degraded <= {peak_climb_warn_z:.1f}), "
            f"{int(collisions)} early collision(s)"
        ),
        details=details,
    )


# ---------------------------------------------------------------------------
# Streaming wrapper (how a real-time monitor is used)
# ---------------------------------------------------------------------------

class SimHealthProbe:
    """Streaming front-end to :func:`probe_health` for the live runner.

    The runner feeds samples one at a time as telemetry arrives
    (``add_sample(t, z)``) and reports collisions as they are drained
    (``add_collisions(n)``), then calls :meth:`evaluate` ONCE the window has
    elapsed. The probe buffers only the in-window samples (it drops anything
    past ``window_s`` after the first sample, so memory is bounded by the loop
    rate × window, ~300 samples at 100 Hz) and delegates the verdict to the pure
    :func:`probe_health`, so the streaming and batch verdicts are IDENTICAL on
    the same data.

    Typical use::

        probe = SimHealthProbe()
        # ... each control tick after GO:
        probe.add_sample(t_s, telem.position_ned[2])
        probe.add_collisions(len(adapter.drain_collisions()))
        if probe.window_elapsed(t_s) and not probe.done:
            verdict = probe.evaluate()
            if verdict.degraded:
                log.warning(verdict.message)

    NEVER raises from ``add_*`` / ``evaluate`` — a bad sample is dropped, and
    ``evaluate`` returns a verdict (``insufficient_data`` when appropriate).
    """

    def __init__(
        self,
        *,
        peak_climb_warn_z: float = PEAK_CLIMB_WARN_Z,
        healthy_ref_z: float = HEALTHY_REF_Z,
        window_s: float = HEALTH_WINDOW_S,
        max_early_collisions: int = MAX_EARLY_COLLISIONS,
        min_samples: int = MIN_SAMPLES,
    ) -> None:
        self.peak_climb_warn_z = peak_climb_warn_z
        self.healthy_ref_z = healthy_ref_z
        self.window_s = window_s
        self.max_early_collisions = max_early_collisions
        self.min_samples = min_samples

        self._samples: List[Sample] = []
        self._collisions: int = 0
        self._t0: Optional[float] = None
        self._verdict: Optional[SimHealthVerdict] = None

    def add_sample(self, t_s: float, z_ned: float) -> None:
        """Add one ``(t_s, z_ned)`` flight sample. No-op for a non-finite/bad
        reading. Samples past the window (relative to the first) are dropped so
        the buffer stays bounded; once :meth:`evaluate` has run, further samples
        are ignored (the verdict is fired once)."""
        if self._verdict is not None:
            return
        try:
            t = float(t_s)
            z = float(z_ned)
        except (TypeError, ValueError):
            return
        if not (np.isfinite(t) and np.isfinite(z)):
            return
        if self._t0 is None:
            self._t0 = t
        # Keep only what the window needs (bounded buffer); a tiny epsilon guards
        # float equality at the window edge.
        if t <= self._t0 + self.window_s + 1e-9:
            self._samples.append((t, z))

    def add_collisions(self, n: int) -> None:
        """Add ``n`` collisions observed since the last call (e.g.
        ``len(adapter.drain_collisions())``). No-op once evaluated."""
        if self._verdict is not None:
            return
        try:
            self._collisions += max(0, int(n))
        except (TypeError, ValueError):
            return

    def window_elapsed(self, t_now_s: float) -> bool:
        """True once ``t_now_s`` is at least ``window_s`` past the first sample.

        The runner uses this to decide WHEN to call :meth:`evaluate`. Returns
        False before any sample has arrived.
        """
        if self._t0 is None:
            return False
        try:
            return float(t_now_s) >= self._t0 + self.window_s
        except (TypeError, ValueError):
            return False

    @property
    def done(self) -> bool:
        """True once :meth:`evaluate` has produced a verdict (fire-once guard)."""
        return self._verdict is not None

    @property
    def verdict(self) -> Optional[SimHealthVerdict]:
        """The latched verdict, or None if :meth:`evaluate` has not run."""
        return self._verdict

    def evaluate(self) -> SimHealthVerdict:
        """Compute (and latch) the verdict from the buffered window. Idempotent —
        returns the same latched verdict on repeat calls. NEVER raises."""
        if self._verdict is not None:
            return self._verdict
        self._verdict = probe_health(
            self._samples,
            collisions=self._collisions,
            peak_climb_warn_z=self.peak_climb_warn_z,
            healthy_ref_z=self.healthy_ref_z,
            window_s=self.window_s,
            max_early_collisions=self.max_early_collisions,
            min_samples=self.min_samples,
        )
        return self._verdict
