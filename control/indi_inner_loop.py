"""Measured-angular-acceleration INDI inner loop with online-G (roadmap #2).

This is the *bench-ID-free* incremental rate-INDI controller the deep-research
report (docs/aigp/2026-06-16-deep-research-improvement-report.md) names as the
**crux discriminator** for the 0.53x roll-attenuation ceiling. It is a fully
self-contained, numpy-only, stateful class with NO MAVLink / adapter imports so
it is unit-testable offline (and embeddable inside the AIGP MAVLink adapter).

What it computes
----------------
The AIGP interface commands a BODY-RATE setpoint via SET_ATTITUDE_TARGET rates
mode; the sim has its OWN black-box rate loop (where the 0.53 roll attenuation
lives). So ``u`` here is the *body-rate setpoint we send*, NOT a motor command.

Per axis (roll, pitch, yaw), incremental rate-INDI:

    u_k = u_{k-1} + Ginv_i * (alpha_des_i - alpha_meas_i)

where ``alpha`` is angular acceleration (d(rate)/dt). ``u_{k-1}`` is the
PREVIOUS rate command and ``alpha_meas`` is the filtered gyro derivative.

CRITICAL INDI detail — MATCHED FILTERING. INDI inverts the relation between the
*increment* in command and the *increment* in measured acceleration. The gyro
derivative is noisy, so it must be low-pass filtered; but if only the measured
signal is filtered, ``u_{k-1}`` (the un-filtered command) leads the filtered
measurement and the inversion sees a lag mismatch -> bias / oscillation. The
fix (Smeur 2016): apply the SAME low-pass filter to BOTH the measured-accel
signal AND the command signal used in the increment, so both carry identical
phase. This module filters the gyro derivative AND keeps a matched-filtered copy
of the command; the increment ``u_{k-1}`` it adds to is the matched-filtered
command, exactly matching ``alpha_meas``'s phase.

alpha_des (desired angular acceleration)
----------------------------------------
Reuses the SAME quaternion error vector as
``competition.aigp_mavlink._attitude_error_body_rates``:
``q_err = conj(q_cur) (x) q_des``, vector part, shortest-path (w >= 0). The PD on
angular acceleration is:

    alpha_des_i = Kp_att_i * (2 * q_err_vec_i) - Kd_att_i * omega_i

The factor of 2 on the error vector matches the rate-PD law (its
``2.0 * kp * ex`` convention) so the attitude-error scaling is identical; the
gains here act on angular ACCELERATION rather than rate.

Online-G (diagonal control effectiveness)
------------------------------------------
``Ghat = diag(g_roll, g_pitch, g_yaw)``; each ``g_i`` tracks ``d(alpha_i)/d(u_i)``
via scalar RLS-with-forgetting, mirroring ``OnlineDroneCalibrator``'s math and
numerical guards (covariance-windup cap, finite checks). ``g_i`` is SEEDED from
the bench-measured per-axis amplification (roll ~1.0, pitch/yaw ~2.1; see
``competition/aigp_mavlink.py`` rate-ID comments) so it NEVER starts blind — the
report explicitly refuted "only a coarse init needed", so a sane prior is kept.
``Ginv`` floors ``|g_i|`` so a near-zero estimate cannot blow up the increment.

Anti-windup / excitation gating
-------------------------------
``u`` is clamped to the same ``+/- max_rate`` envelope as the PD path. ``g_i`` is
NOT updated when that axis is saturated (the increment was clipped, so the
measured response no longer reflects the commanded one) OR when ``|delta u_i|``
is below an excitation threshold (a smooth racing line may under-excite roll;
RLS would divide near-zero by near-zero and corrupt the estimate). FREEZE, do
not corrupt.

THE DISCRIMINATOR READ-OUT
--------------------------
Drive this loop closed-loop against the unknown sim rate loop and watch the
achieved rate/attitude:

  * **recovered => model mismatch.** If a static effectiveness/gain error (e.g.
    achieved = 0.53 * commanded, no bandwidth limit), online-G grows g_i toward
    the true ratio, the inversion pre-distorts the command, and the achieved
    rate is driven back to the commanded value. The deficit was RECOVERABLE.
  * **still clamped => true bandwidth limit.** If the sim loop is a genuine
    first-order lag / slew-rate cap, the measured-accel inversion is *correct*
    yet the achieved rate stays attenuated/clamped — no command pre-distortion
    beats a bandwidth wall. The deficit is a TRUE actuator/bandwidth limit.

See ``control/tests/test_indi_inner_loop.py`` for the synthetic discriminator
harness that asserts exactly this (model (a) recovers, model (b) does not).

Numerics: a fixed handful of scalar ops per axis per step — numpy-only, no
allocation growth, comfortably inside the <1 ms / >100 Hz control budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Bench-measured per-axis rate-channel amplification used as the ONLINE-G prior
# (competition/aigp_mavlink.py:135-143 rate-ID: ~1.0x roll, ~2.1x pitch/yaw).
# These are the d(alpha)/d(u) effectiveness seeds — never start blind.
DEFAULT_G_SEED: Tuple[float, float, float] = (1.0, 2.1, 2.1)

_AXES = ("roll", "pitch", "yaw")


def _as3(v, name: str) -> Tuple[float, float, float]:
    """Coerce a scalar or length-3 sequence to a 3-tuple of floats."""
    if isinstance(v, (int, float)):
        return (float(v), float(v), float(v))
    seq = tuple(float(x) for x in v)
    if len(seq) != 3:
        raise ValueError(f"{name} must be a scalar or length-3; got {v!r}")
    return seq


@dataclass
class IndiConfig:
    """Tuning knobs for :class:`IndiInnerLoop`. Kept minimal."""

    # Desired-angular-acceleration PD (attitude error -> alpha_des). Per-axis or
    # scalar. These act on angular ACCELERATION (rad/s^2), so they are NOT the
    # FALSIFIED rate-PD gains — the INDI inversion converts alpha_des to a rate
    # increment via Ginv.
    kp_att: Tuple[float, float, float] = (18.0, 18.0, 12.0)
    kd_att: Tuple[float, float, float] = (3.0, 3.0, 2.0)

    # Body-rate clamp (rad/s) — MUST match the PD path's envelope so the INDI
    # branch never commands outside the validated rate range. Scalar or per-axis.
    max_rate: Tuple[float, float, float] = (0.8, 0.8, 0.8)

    # Matched low-pass cutoff (Hz) applied to BOTH the gyro derivative and the
    # command increment. 2nd-order critically-damped biquad when order==2.
    filter_cutoff_hz: float = 20.0
    filter_order: int = 2  # 1 (single-pole) or 2 (critically-damped biquad)

    # --- Online-G (per-axis scalar RLS with forgetting) ---
    g_seed: Tuple[float, float, float] = DEFAULT_G_SEED
    forgetting_factor: float = 0.995  # lambda in (0, 1]; <1 tracks drift.
    covariance_init: float = 1.0e2
    max_covariance: float = 1.0e6  # covariance-windup cap (mirror OnlineDroneCalibrator)
    g_inv_floor: float = 0.2  # floor on |g_i| used to form Ginv (never divide by ~0)
    g_clip: Tuple[float, float] = (0.05, 20.0)  # hard bound on the internal g estimate

    # Excitation gate: skip the G update when |delta u_i| (matched-filtered) is
    # below this (rad/s). A smooth racing line may under-excite roll; freeze.
    excitation_min_du: float = 1.0e-3

    enable_online_g: bool = True  # when False, g_i stays at the seed (fixed-G INDI)

    def __post_init__(self) -> None:
        self.kp_att = _as3(self.kp_att, "kp_att")
        self.kd_att = _as3(self.kd_att, "kd_att")
        self.max_rate = _as3(self.max_rate, "max_rate")
        self.g_seed = _as3(self.g_seed, "g_seed")
        if not (0.0 < self.forgetting_factor <= 1.0):
            raise ValueError(
                f"forgetting_factor must be in (0, 1]; got {self.forgetting_factor!r}"
            )
        if self.covariance_init <= 0.0:
            raise ValueError("covariance_init must be positive")
        if self.max_covariance <= self.covariance_init:
            raise ValueError("max_covariance must exceed covariance_init")
        if self.g_inv_floor <= 0.0:
            raise ValueError("g_inv_floor must be positive")
        if self.filter_order not in (1, 2):
            raise ValueError("filter_order must be 1 or 2")
        if self.filter_cutoff_hz <= 0.0:
            raise ValueError("filter_cutoff_hz must be positive")
        lo, hi = self.g_clip
        if not (0.0 < lo < hi):
            raise ValueError("g_clip must be (lo, hi) with 0 < lo < hi")


@dataclass
class IndiDebug:
    """Per-step snapshot for recording / analysis (the discriminator read-out)."""

    alpha_des: Tuple[float, float, float]
    alpha_meas: Tuple[float, float, float]
    ghat: Tuple[float, float, float]
    saturated: Tuple[bool, bool, bool]
    g_updated: Tuple[bool, bool, bool]
    u: Tuple[float, float, float]
    q_err_vec: Tuple[float, float, float]
    dt: float


class _MatchedLowPass:
    """Discrete low-pass filter, identical instance applied to two signals.

    Two filters constructed with the same config + dt have identical phase, so
    filtering the gyro-derivative with one and the command with another gives
    the matched filtering INDI requires. First order is a single pole; second
    order is a critically-damped (real repeated pole) biquad for a steeper
    roll-off without ringing.
    """

    def __init__(self, cutoff_hz: float, order: int):
        self.cutoff_hz = float(cutoff_hz)
        self.order = int(order)
        self._y1: Optional[float] = None  # previous output
        self._y2: Optional[float] = None  # output two steps back (order 2)

    def reset(self) -> None:
        self._y1 = None
        self._y2 = None

    def apply(self, x: float, dt: float) -> float:
        # Single-pole coefficient from the cutoff and dt (guarded dt elsewhere).
        wc = 2.0 * math.pi * self.cutoff_hz
        a = wc * dt / (1.0 + wc * dt)  # in (0, 1)
        if self._y1 is None:
            # Seed both states with the first sample so there is no startup
            # transient that would bias the steady state.
            self._y1 = x
            self._y2 = x
            return x
        if self.order == 1:
            y = self._y1 + a * (x - self._y1)
            self._y1 = y
            return y
        # Order 2: cascade of two identical single-pole stages (real repeated
        # pole -> critically damped, no overshoot, matched phase to itself).
        s1 = self._y1 + a * (x - self._y1)
        y = self._y2 + a * (s1 - self._y2)
        self._y1 = s1
        self._y2 = y
        return y


class _ScalarRLS:
    """Per-axis scalar RLS-with-forgetting for g_i = d(alpha_i)/d(u_i).

    Mirrors ``OnlineDroneCalibrator``'s recursion and guards, specialised to a
    single scalar parameter and a single scalar regressor (delta_u):

        e = delta_alpha - g * delta_u             (a-priori error)
        K = P*delta_u / (lambda + delta_u^2 * P)
        g <- g + K e
        P <- (P - K*delta_u*P) / lambda           (forget old data)

    P is capped (covariance-windup guard); g is hard-clipped to a physical band.
    """

    def __init__(self, g_init: float, lam: float, p_init: float,
                 max_cov: float, g_clip: Tuple[float, float]):
        self.g = float(g_init)
        self.lam = float(lam)
        self.P = float(p_init)
        self.max_cov = float(max_cov)
        self.g_lo, self.g_hi = float(g_clip[0]), float(g_clip[1])

    def update(self, du: float, dalpha: float) -> None:
        if not (math.isfinite(du) and math.isfinite(dalpha)):
            return
        e = dalpha - self.g * du
        denom = self.lam + du * du * self.P
        if denom <= 0.0 or not math.isfinite(denom):
            return
        k = self.P * du / denom
        g_new = self.g + k * e
        if math.isfinite(g_new):
            self.g = min(self.g_hi, max(self.g_lo, g_new))
        p_new = (self.P - k * du * self.P) / self.lam
        if math.isfinite(p_new) and p_new > 0.0:
            self.P = min(p_new, self.max_cov)


class IndiInnerLoop:
    """Stateful incremental rate-INDI inner loop with online-G.

    Call :meth:`compute` once per control tick with the current/desired
    orientation quaternions, the measured body-rate gyro, and the timestep. It
    returns the body-rate setpoint ``(roll_rate, pitch_rate, yaw_rate)`` to send
    (the CALLER still applies the sim's per-axis ``_rate_sign``). The latest
    :class:`IndiDebug` snapshot is available on :attr:`last_debug`.

    Guards: ``dt <= 0`` or non-finite ``dt`` / inputs hold the previous command
    and skip all state updates (no NaN propagation). The first call has no valid
    derivative, so ``alpha_meas`` is treated as zero and G is not updated.
    """

    def __init__(self, config: Optional[IndiConfig] = None):
        self.config = config or IndiConfig()
        c = self.config

        # Matched filters: one per axis for the gyro derivative, one per axis
        # for the command increment. Same config => identical phase response.
        self._accel_filt = [
            _MatchedLowPass(c.filter_cutoff_hz, c.filter_order) for _ in range(3)
        ]
        self._cmd_filt = [
            _MatchedLowPass(c.filter_cutoff_hz, c.filter_order) for _ in range(3)
        ]
        # Per-axis online-G estimators (RLS), seeded from the bench prior.
        self._rls = [
            _ScalarRLS(c.g_seed[i], c.forgetting_factor, c.covariance_init,
                       c.max_covariance, c.g_clip)
            for i in range(3)
        ]

        # State carried between ticks.
        self._u_prev = np.zeros(3, dtype=np.float64)         # last raw command
        self._u_filt_prev = np.zeros(3, dtype=np.float64)    # matched-filtered cmd (k-1)
        self._omega_prev: Optional[np.ndarray] = None        # last gyro (for derivative)
        self._alpha_filt_prev = np.zeros(3, dtype=np.float64)  # matched alpha (k-1)
        self.last_debug: Optional[IndiDebug] = None

    # -- public API ---------------------------------------------------------
    @property
    def ghat(self) -> Tuple[float, float, float]:
        """Current per-axis control-effectiveness estimate (roll, pitch, yaw)."""
        return tuple(r.g for r in self._rls)  # type: ignore[return-value]

    def reset(self) -> None:
        """Clear filter/derivative state (e.g. on a sim reset). Keeps Ghat."""
        for f in self._accel_filt:
            f.reset()
        for f in self._cmd_filt:
            f.reset()
        self._u_prev[:] = 0.0
        self._u_filt_prev[:] = 0.0
        self._omega_prev = None
        self._alpha_filt_prev[:] = 0.0

    def compute(
        self,
        q_cur,
        q_des,
        omega: Sequence[float],
        dt: float,
    ) -> Tuple[float, float, float]:
        """Return the body-rate setpoint (roll,pitch,yaw) for this tick.

        Args:
            q_cur, q_des: current / desired orientation. Anything with
                ``.w/.x/.y/.z`` attributes (e.g. ``competition.adapter.Quaternion``)
                or a 4-sequence ``(w, x, y, z)``.
            omega: measured body-rate gyro (rad/s), FRD — same convention as the
                PD path's ``omega``.
            dt: control timestep (s). ``dt <= 0`` / non-finite holds the command.
        """
        c = self.config
        omega_arr = np.asarray([float(o) for o in omega], dtype=np.float64)

        # --- guards: bad dt or non-finite inputs => hold last command -------
        if (not math.isfinite(dt)) or dt <= 0.0 or not np.all(np.isfinite(omega_arr)):
            u = self._u_prev
            self._set_debug(
                alpha_des=(0.0, 0.0, 0.0), alpha_meas=(0.0, 0.0, 0.0),
                saturated=(False, False, False), g_updated=(False, False, False),
                u=u, q_err_vec=(0.0, 0.0, 0.0), dt=float(dt) if math.isfinite(dt) else 0.0,
            )
            return float(u[0]), float(u[1]), float(u[2])

        q_err_vec = _quat_error_vec(q_cur, q_des)
        if not np.all(np.isfinite(q_err_vec)):
            u = self._u_prev
            self._set_debug(
                alpha_des=(0.0, 0.0, 0.0), alpha_meas=(0.0, 0.0, 0.0),
                saturated=(False, False, False), g_updated=(False, False, False),
                u=u, q_err_vec=(0.0, 0.0, 0.0), dt=dt,
            )
            return float(u[0]), float(u[1]), float(u[2])

        kp = np.asarray(c.kp_att, dtype=np.float64)
        kd = np.asarray(c.kd_att, dtype=np.float64)
        max_rate = np.asarray(c.max_rate, dtype=np.float64)

        # --- desired angular acceleration (attitude-error PD) ---------------
        # factor of 2 matches _attitude_error_body_rates' 2*kp*ex convention.
        alpha_des = kp * (2.0 * q_err_vec) - kd * omega_arr

        # --- measured angular acceleration (filtered gyro derivative) -------
        if self._omega_prev is None:
            alpha_meas_raw = np.zeros(3, dtype=np.float64)
            have_deriv = False
        else:
            alpha_meas_raw = (omega_arr - self._omega_prev) / dt
            have_deriv = True
        alpha_meas = np.array([
            self._accel_filt[i].apply(float(alpha_meas_raw[i]), dt) for i in range(3)
        ], dtype=np.float64)

        # --- online-G update (RLS) on the matched-filtered increments ------
        # delta_u is the change in the MATCHED-FILTERED command (phase-aligned
        # with alpha_meas); delta_alpha the change in filtered measured accel.
        g_updated = [False, False, False]
        # Saturation of the PREVIOUS command (the one that produced this tick's
        # measured response). If it was clipped, the response is not the true
        # effectiveness -> freeze.
        prev_saturated = np.abs(self._u_prev) >= (max_rate - 1e-9)
        if have_deriv and c.enable_online_g:
            du = self._u_filt_prev - self._u_filt_prev2()
            dalpha = alpha_meas - self._alpha_filt_prev
            for i in range(3):
                if prev_saturated[i]:
                    continue  # anti-windup: don't learn from a clipped command
                if abs(du[i]) < c.excitation_min_du:
                    continue  # under-excitation: freeze, don't corrupt
                self._rls[i].update(float(du[i]), float(dalpha[i]))
                g_updated[i] = True

        # --- INDI inversion: u_k = u_{k-1} + Ginv * (alpha_des - alpha_meas) -
        # Increment added to the MATCHED-FILTERED previous command (matched
        # filtering: the term we add to has the same phase as alpha_meas).
        ginv = np.array([self._ginv(i) for i in range(3)], dtype=np.float64)
        u_raw = self._u_filt_prev + ginv * (alpha_des - alpha_meas)

        # --- anti-windup clamp to the +/- max_rate envelope ----------------
        u = np.clip(u_raw, -max_rate, max_rate)
        saturated = np.abs(u_raw) > max_rate

        # --- roll state forward --------------------------------------------
        # Matched-filtered copy of the command we are actually issuing this
        # tick (feeds next tick's increment + the next RLS delta_u).
        u_filt = np.array([
            self._cmd_filt[i].apply(float(u[i]), dt) for i in range(3)
        ], dtype=np.float64)
        self._u_filt_prev2_store = self._u_filt_prev.copy()
        self._u_filt_prev = u_filt
        self._u_prev = u.copy()
        self._omega_prev = omega_arr.copy()
        self._alpha_filt_prev = alpha_meas.copy()

        self._set_debug(
            alpha_des=tuple(alpha_des), alpha_meas=tuple(alpha_meas),
            saturated=tuple(bool(s) for s in saturated),
            g_updated=tuple(g_updated), u=u,
            q_err_vec=tuple(q_err_vec), dt=dt,
        )
        return float(u[0]), float(u[1]), float(u[2])

    # -- internals ----------------------------------------------------------
    def _ginv(self, i: int) -> float:
        """Inverse effectiveness for axis i, with a magnitude floor."""
        g = self._rls[i].g
        floored = math.copysign(max(abs(g), self.config.g_inv_floor), g if g != 0 else 1.0)
        return 1.0 / floored

    def _u_filt_prev2(self) -> np.ndarray:
        """Matched-filtered command two ticks back (for the RLS increment)."""
        return getattr(self, "_u_filt_prev2_store", np.zeros(3, dtype=np.float64))

    def _set_debug(self, *, alpha_des, alpha_meas, saturated, g_updated, u,
                   q_err_vec, dt) -> None:
        self.last_debug = IndiDebug(
            alpha_des=tuple(float(x) for x in alpha_des),
            alpha_meas=tuple(float(x) for x in alpha_meas),
            ghat=self.ghat,
            saturated=tuple(bool(x) for x in saturated),
            g_updated=tuple(bool(x) for x in g_updated),
            u=tuple(float(x) for x in u),
            q_err_vec=tuple(float(x) for x in q_err_vec),
            dt=float(dt),
        )

    def debug_dict(self) -> Optional[Dict]:
        """The latest debug snapshot as a plain dict (for JSONL recording)."""
        d = self.last_debug
        if d is None:
            return None
        return {
            "alpha_des": list(d.alpha_des),
            "alpha_meas": list(d.alpha_meas),
            "ghat": list(d.ghat),
            "saturated": list(d.saturated),
            "g_updated": list(d.g_updated),
            "u": list(d.u),
            "q_err_vec": list(d.q_err_vec),
            "dt": d.dt,
        }


def _quat_error_vec(q_cur, q_des) -> np.ndarray:
    """Body-frame quaternion error vector part, shortest-path (w >= 0).

    EXACTLY the ``conj(q_cur) (x) q_des`` computation from
    ``competition.aigp_mavlink._attitude_error_body_rates`` (vector part only).
    Accepts objects with ``.w/.x/.y/.z`` or a 4-sequence ``(w, x, y, z)``.
    """
    qc = _quat_wxyz(q_cur)
    qd = _quat_wxyz(q_des)
    cw, cx, cy, cz = qc[0], -qc[1], -qc[2], -qc[3]  # conj(qc)
    ew = cw * qd[0] - cx * qd[1] - cy * qd[2] - cz * qd[3]
    ex = cw * qd[1] + cx * qd[0] + cy * qd[3] - cz * qd[2]
    ey = cw * qd[2] - cx * qd[3] + cy * qd[0] + cz * qd[1]
    ez = cw * qd[3] + cx * qd[2] - cy * qd[1] + cz * qd[0]
    if ew < 0:  # shortest path: q and -q are the same rotation
        ex, ey, ez = -ex, -ey, -ez
    return np.array([ex, ey, ez], dtype=np.float64)


def _quat_wxyz(q) -> Tuple[float, float, float, float]:
    if hasattr(q, "w"):
        return (float(q.w), float(q.x), float(q.y), float(q.z))
    seq = tuple(float(x) for x in q)
    if len(seq) != 4:
        raise ValueError(f"quaternion must have w/x/y/z or be length-4; got {q!r}")
    return seq  # type: ignore[return-value]
