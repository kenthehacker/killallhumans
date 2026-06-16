"""Offline unit tests for the measured-accel INDI inner loop (roadmap #2).

These tests are numpy-only and import NO MAVLink — the IndiInnerLoop is a pure
control object. They cover:

  * THE DISCRIMINATOR (the point of the whole task): two synthetic sim rate-loop
    models driven closed-loop by the IndiInnerLoop —
      (a) pure static-gain attenuation (achieved = 0.53 * commanded, fast loop):
          a recoverable MISMATCH;
      (b) a TRUE rate-bandwidth limit (slew-rate cap + lag): NOT recoverable by
          command pre-distortion.
    With online-G enabled the loop RECOVERS the commanded rate for (a) (fast
    slew, peak achieved rate ~= demanded) but for (b) the achieved rate stays
    CLAMPED at the bandwidth wall despite a correct measured-accel inversion.
    READ-OUT: recovered => model mismatch; still clamped => true bandwidth limit.
  * online-G convergence to the true per-axis effectiveness under excitation;
  * matched filtering introduces no steady-state lag/DC bias;
  * anti-windup freezes G under saturation and under-excitation;
  * dt<=0 and non-finite guards hold the command and never emit NaN.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from competition.adapter import Quaternion
from control.indi_inner_loop import (
    DEFAULT_G_SEED,
    IndiConfig,
    IndiInnerLoop,
    _MatchedLowPass,
    _quat_error_vec,
)

DT = 0.0025  # 400 Hz — representative of the >100 Hz control budget


# ---------------------------------------------------------------------------
# Synthetic "sim rate-loop" models — the two halves of the discriminator.
# ---------------------------------------------------------------------------

class StaticGainRateLoop:
    """Model (a): a RECOVERABLE static-gain attenuation.

    The sim's hidden rate loop tracks the commanded rate ``u`` with a static DC
    gain ``K`` and a FAST first-order response (small ``tau`` => high bandwidth):

        achieved_rate_dot = (K * u - achieved_rate) / tau

    Steady state ``achieved_rate = K * u``. With ``K = 0.53`` this is exactly the
    handoff's "achieved roll ~= 0.53x commanded" deficit, but there is NO
    bandwidth limit (tau is small). INDI's online-G learns the effective gain and
    inflates ``u`` so the achieved rate is restored — the deficit is RECOVERABLE.
    """

    def __init__(self, K, tau: float = 0.01):
        self.K = np.asarray(K, dtype=np.float64)
        self.tau = float(tau)
        self.rate = np.zeros(3, dtype=np.float64)

    def step(self, u, dt: float) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        self.rate = self.rate + (self.K * u - self.rate) / self.tau * dt
        return self.rate.copy()


class BandwidthLimitedRateLoop:
    """Model (b): a NON-recoverable true rate-bandwidth / slew-rate limit.

    DC gain is unity, but the achieved rate's slew is HARD-CAPPED at
    ``slew_cap`` rad/s^2 (a true actuator-bandwidth wall):

        target_dot = (u - achieved_rate) / tau
        achieved_rate_dot = clip(target_dot, -slew_cap, +slew_cap)

    No amount of command pre-distortion lets the achieved rate slew faster than
    ``slew_cap`` allows — INDI's measured-accel inversion is correct yet the
    achieved rate stays clamped. The deficit is a TRUE bandwidth limit.
    """

    def __init__(self, slew_cap: float, tau: float = 0.012):
        self.slew = float(slew_cap)
        self.tau = float(tau)
        self.rate = np.zeros(3, dtype=np.float64)

    def step(self, u, dt: float) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        target_dot = (u - self.rate) / self.tau
        target_dot = np.clip(target_dot, -self.slew, self.slew)
        self.rate = self.rate + target_dot * dt
        return self.rate.copy()


def _closed_loop_step_response(
    plant,
    *,
    warmup_s: float = 8.0,
    hold_s: float = 4.0,
    step_angle_rad: float = 0.4,
    cfg: IndiConfig = None,
):
    """Drive ``plant`` closed-loop with an IndiInnerLoop and return step metrics.

    Phase 1 (warm-up): a small roll sine excites the roll axis so online-G can
    converge BEFORE the step (the report warns a smooth line under-excites roll).
    Phase 2 (step): command a +``step_angle_rad`` roll step and measure the PEAK
    achieved body rate and the time to reach 90 % of the step. Both models
    eventually reach the commanded attitude (the attitude integrator wins), so
    the discriminator is the *rate / speed* of recovery, not the final angle.
    """
    if cfg is None:
        cfg = IndiConfig(
            kp_att=(25.0, 25.0, 18.0), kd_att=(5.0, 5.0, 4.0),
            max_rate=(8.0, 8.0, 8.0), filter_cutoff_hz=55.0,
            excitation_min_du=5.0e-4,
        )
    loop = IndiInnerLoop(cfg)
    rate = np.zeros(3, dtype=np.float64)
    roll = 0.0

    n_warm = int(warmup_s / DT)
    for k in range(n_warm):
        t = k * DT
        roll_des = 0.12 * math.sin(2 * math.pi * 1.1 * t)
        u = loop.compute(
            Quaternion.from_euler(roll, 0.0, 0.0),
            Quaternion.from_euler(roll_des, 0.0, 0.0),
            omega=tuple(rate), dt=DT,
        )
        rate = plant.step(u, DT)
        roll += rate[0] * DT
    g_warm = loop.ghat[0]

    n_hold = int(hold_s / DT)
    roll0 = roll
    target = roll + step_angle_rad
    peak_rate = 0.0
    t90 = None
    for k in range(n_hold):
        u = loop.compute(
            Quaternion.from_euler(roll, 0.0, 0.0),
            Quaternion.from_euler(target, 0.0, 0.0),
            omega=tuple(rate), dt=DT,
        )
        rate = plant.step(u, DT)
        roll += rate[0] * DT
        peak_rate = max(peak_rate, abs(rate[0]))
        if t90 is None and (roll - roll0) >= 0.9 * step_angle_rad:
            t90 = k * DT
    return {
        "peak_rate": peak_rate,
        "t90": t90,
        "reached": roll - roll0,
        "g_warm": g_warm,
    }


# ---------------------------------------------------------------------------
# THE HEADLINE DELIVERABLE — discriminator with a negative control.
# ---------------------------------------------------------------------------

def test_discriminator_static_gain_is_recovered():
    """Model (a): a static-gain (0.53) attenuation MUST be recovered.

    With online-G, the achieved rate is restored: the drone slews fast (high
    peak rate, reaches the step quickly). READ-OUT: recovered => model mismatch.
    """
    res = _closed_loop_step_response(StaticGainRateLoop([0.53, 1.0, 1.0], tau=0.01))
    # The slew reaches the commanded attitude...
    assert res["t90"] is not None, "static-gain model never reached 90% of the step"
    assert res["reached"] == pytest.approx(0.4, abs=0.05)
    # ...and does so FAST, with a high peak achieved rate (the deficit is
    # recovered — the achieved rate is NOT pinned near the 0.53 attenuation).
    assert res["peak_rate"] > 0.8, (
        f"static-gain peak achieved rate {res['peak_rate']:.3f} too low — "
        "the 0.53 deficit was NOT recovered"
    )
    assert res["t90"] < 0.8, f"static-gain slew too slow: t90={res['t90']:.3f}s"


def test_discriminator_bandwidth_limit_is_not_recovered():
    """Model (b): a TRUE slew-rate/bandwidth limit must NOT be recovered.

    The achieved rate stays CLAMPED at the slew cap despite a correct
    measured-accel inversion; the slew is far slower. READ-OUT: still clamped
    => true bandwidth limit.
    """
    slew_cap = 0.5
    res = _closed_loop_step_response(BandwidthLimitedRateLoop(slew_cap, tau=0.012))
    # The achieved peak rate is pinned at the bandwidth wall (~slew_cap), NOT the
    # high rate INDI would need to slew fast — command pre-distortion can't beat
    # the slew cap.
    assert res["peak_rate"] <= slew_cap * 1.2, (
        f"bandwidth-limited peak rate {res['peak_rate']:.3f} exceeded the slew "
        f"cap {slew_cap} — the wall did not hold"
    )
    assert res["peak_rate"] < 0.7, "bandwidth-limited achieved rate was not clamped"


def test_discriminator_separates_the_two_models():
    """The harness ACTUALLY discriminates: (a) recovers a strictly faster slew /
    higher achieved rate than (b). This is the crux read-out in one assertion.

    recovered (a) => model mismatch;  still clamped (b) => true bandwidth limit.
    """
    a = _closed_loop_step_response(StaticGainRateLoop([0.53, 1.0, 1.0], tau=0.01))
    b = _closed_loop_step_response(BandwidthLimitedRateLoop(0.5, tau=0.012))
    # Achieved peak rate: (a) is markedly higher than (b) — the deficit is
    # recovered in (a), clamped in (b).
    assert a["peak_rate"] > 1.5 * b["peak_rate"], (
        f"discriminator failed to separate: static-gain peak rate "
        f"{a['peak_rate']:.3f} vs bandwidth-limited {b['peak_rate']:.3f}"
    )
    # Time-to-target: (a) reaches the step; (b) is much slower (or never within
    # the window). Either way (a) is strictly faster.
    assert a["t90"] is not None
    if b["t90"] is not None:
        assert a["t90"] < b["t90"], (
            f"static-gain t90 {a['t90']:.3f}s not faster than bandwidth "
            f"t90 {b['t90']:.3f}s"
        )


# ---------------------------------------------------------------------------
# Online-G convergence (textbook INDI plant: alpha = g_true * u).
# ---------------------------------------------------------------------------

def _drive_textbook_indi(g_true, *, n: int = 20000, cfg: IndiConfig = None):
    """Closed loop where the plant is the textbook INDI relation
    ``angular_accel = g_true * u`` (per axis), with rich multi-axis excitation
    so every g_i is well-excited. Returns the final Ghat."""
    if cfg is None:
        cfg = IndiConfig(
            g_seed=(1.0, 2.1, 2.1), kp_att=(20.0, 20.0, 15.0),
            kd_att=(4.0, 4.0, 3.0), max_rate=(12.0, 12.0, 12.0),
            filter_cutoff_hz=60.0, forgetting_factor=0.999,
            excitation_min_du=1.0e-4,
        )
    g_true = np.asarray(g_true, dtype=np.float64)
    loop = IndiInnerLoop(cfg)
    rate = np.zeros(3, dtype=np.float64)
    ang = np.zeros(3, dtype=np.float64)
    for k in range(n):
        t = k * DT
        des = np.array([
            0.25 * math.sin(2 * math.pi * 0.9 * t),
            0.20 * math.sin(2 * math.pi * 1.3 * t + 1.0),
            0.18 * math.sin(2 * math.pi * 0.7 * t + 2.0),
        ])
        u = np.asarray(loop.compute(
            Quaternion.from_euler(*ang), Quaternion.from_euler(*des),
            omega=tuple(rate), dt=DT,
        ))
        alpha = g_true * u            # plant: alpha = g_true * u
        rate = rate + alpha * DT
        ang = ang + rate * DT
    return np.asarray(loop.ghat)


def test_online_g_converges_to_true_per_axis_gain():
    """Under excitation, online-G converges to the true per-axis effectiveness,
    even from a seed that is wrong on two of three axes."""
    g_true = np.array([1.4, 2.1, 0.7])
    ghat = _drive_textbook_indi(g_true)
    for i, axis in enumerate(("roll", "pitch", "yaw")):
        rel = abs(ghat[i] - g_true[i]) / g_true[i]
        assert rel < 0.05, (
            f"{axis} online-G did not converge: learned {ghat[i]:.3f}, "
            f"true {g_true[i]:.3f} (rel err {rel:.3f})"
        )


def test_online_g_seed_is_a_sane_prior_not_blind():
    """G must start at the bench-measured prior (roll ~1.0, pitch/yaw ~2.1),
    NEVER blind — the report refuted 'only a coarse init needed'."""
    loop = IndiInnerLoop(IndiConfig())
    assert loop.ghat == pytest.approx(DEFAULT_G_SEED)
    assert DEFAULT_G_SEED[0] == pytest.approx(1.0)
    assert DEFAULT_G_SEED[1] == pytest.approx(2.1)
    assert DEFAULT_G_SEED[2] == pytest.approx(2.1)


def test_online_g_disabled_keeps_seed():
    """With online-G disabled, G stays exactly at the seed (fixed-G INDI)."""
    loop = IndiInnerLoop(IndiConfig(enable_online_g=False, g_seed=(1.0, 2.1, 2.1)))
    rate = np.zeros(3)
    ang = np.zeros(3)
    for k in range(2000):
        t = k * DT
        des = np.array([0.3 * math.sin(2 * math.pi * 1.0 * t), 0.0, 0.0])
        u = np.asarray(loop.compute(
            Quaternion.from_euler(*ang), Quaternion.from_euler(*des),
            omega=tuple(rate), dt=DT,
        ))
        rate = rate + (np.array([1.4, 2.1, 2.1]) * u) * DT
        ang = ang + rate * DT
    assert loop.ghat == pytest.approx((1.0, 2.1, 2.1))


# ---------------------------------------------------------------------------
# Matched filtering — no steady-state lag / DC bias.
# ---------------------------------------------------------------------------

def test_matched_filters_are_phase_identical():
    """Two filters with the same config on the SAME signal produce IDENTICAL
    output — the matched-filter property INDI needs (no relative lag between the
    filtered command and the filtered measured accel)."""
    f1 = _MatchedLowPass(20.0, 2)
    f2 = _MatchedLowPass(20.0, 2)
    xs = [math.sin(0.1 * k) + 0.3 for k in range(500)]
    o1 = [f1.apply(x, DT) for x in xs]
    o2 = [f2.apply(x, DT) for x in xs]
    assert np.allclose(o1, o2)


def test_matched_filter_no_dc_bias():
    """A constant input settles to exactly the input (unity DC gain — no
    steady-state lag bias). Holds for both filter orders."""
    for order in (1, 2):
        f = _MatchedLowPass(20.0, order)
        y = 0.0
        for _ in range(5000):
            y = f.apply(2.5, DT)
        assert y == pytest.approx(2.5, abs=1e-9), f"order {order} DC bias: {y}"


def test_matched_filter_no_steady_state_command_bias():
    """End-to-end: with online-G OFF and seeds == true gain, a CONSTANT desired
    rate yields a constant steady command with no drift/bias from the matched
    filters (the filtered alpha and filtered command stay phase-aligned, so the
    INDI increment goes to zero in steady state rather than accumulating)."""
    # Plant alpha = g_true * u; with g_seed == g_true and online-G off, the
    # inversion is exact. Hold a steady small desired rate via a tiny constant
    # attitude error and check the command stops drifting once settled.
    # Plant alpha = g_true * u is a DOUBLE integrator to angle, so the attitude
    # PD needs damping (kd_att) to be stable — that is a property of the plant,
    # not the filter. With damping the loop settles; we then check the SETTLED
    # command is steady (matched filters add no drift / lag bias keeping the
    # INDI increment non-zero forever).
    g_true = (1.0, 2.1, 2.1)
    loop = IndiInnerLoop(IndiConfig(
        g_seed=g_true, enable_online_g=False,
        kp_att=(20.0, 20.0, 15.0), kd_att=(8.0, 8.0, 6.0),
        filter_cutoff_hz=30.0, max_rate=(8.0, 8.0, 8.0),
    ))
    rate = np.zeros(3)
    ang = np.zeros(3)
    us = []
    for k in range(8000):
        # Constant desired attitude => steady state has zero rate; once settled
        # the command must be steady (no integrator windup from filter lag).
        u = np.asarray(loop.compute(
            Quaternion.from_euler(*ang), Quaternion.from_euler(0.05, 0.0, 0.0),
            omega=tuple(rate), dt=DT,
        ))
        rate = rate + (np.asarray(g_true) * u) * DT
        ang = ang + rate * DT
        if k > 7000:
            us.append(u[0])
    us = np.asarray(us)
    # Settled command is steady (tiny spread) and the attitude converged to the
    # target — no runaway bias from a filter lag mismatch.
    assert np.std(us) < 1e-3, f"command not steady (filter lag bias?): std={np.std(us):.2e}"
    assert ang[0] == pytest.approx(0.05, abs=2e-3)


# ---------------------------------------------------------------------------
# Anti-windup / excitation gating — FREEZE, don't corrupt.
# ---------------------------------------------------------------------------

def test_g_frozen_under_saturation():
    """When an axis saturates the rate clamp, its G must NOT update (the clipped
    command no longer reflects the true effectiveness)."""
    cfg = IndiConfig(max_rate=(0.3, 0.3, 0.3), filter_cutoff_hz=40.0,
                     g_seed=(1.0, 2.1, 2.1))
    loop = IndiInnerLoop(cfg)
    g0 = loop.ghat[0]
    rate = np.zeros(3)
    saw_saturation = False
    for k in range(3000):
        # Huge constant roll error; keep measured rate ~0 so the error (and thus
        # the demanded rate) stays huge => the roll command pins the clamp.
        loop.compute(
            Quaternion.from_euler(0.0, 0.0, 0.0),
            Quaternion.from_euler(1.2, 0.0, 0.0),
            omega=tuple(rate), dt=DT,
        )
        if loop.last_debug.saturated[0]:
            saw_saturation = True
        assert loop.last_debug.g_updated[0] is False
    assert saw_saturation, "test did not actually drive the roll axis into saturation"
    assert loop.ghat[0] == pytest.approx(g0, abs=1e-12)


def test_g_frozen_under_excitation_threshold():
    """A steady target (|delta u| below the excitation floor) freezes G — the
    report's flagged risk that a smooth racing line under-excites roll."""
    loop = IndiInnerLoop(IndiConfig(excitation_min_du=1e-3, filter_cutoff_hz=40.0))
    g0 = loop.ghat
    for k in range(2000):
        loop.compute(Quaternion(), Quaternion(), omega=(0.0, 0.0, 0.0), dt=DT)
        assert loop.last_debug.g_updated == (False, False, False)
    assert loop.ghat == pytest.approx(g0)


def test_command_clamped_to_max_rate_envelope():
    """The emitted command never exceeds the +/- max_rate envelope (same
    envelope as the PD path), and the saturation flag is raised when it tries."""
    max_rate = 0.8
    loop = IndiInnerLoop(IndiConfig(max_rate=(max_rate, max_rate, max_rate)))
    rate = np.zeros(3)
    over = False
    for k in range(500):
        u = loop.compute(
            Quaternion(), Quaternion.from_euler(1.5, -1.5, 1.0),
            omega=tuple(rate), dt=DT,
        )
        assert all(abs(x) <= max_rate + 1e-9 for x in u)
        if any(loop.last_debug.saturated):
            over = True
    assert over, "expected saturation under a large attitude error"


# ---------------------------------------------------------------------------
# dt and non-finite guards.
# ---------------------------------------------------------------------------

def test_dt_nonpositive_holds_command():
    loop = IndiInnerLoop()
    u_prev = loop.compute(Quaternion(), Quaternion.from_euler(0.1, 0.0, 0.0),
                          omega=(0.0, 0.0, 0.0), dt=DT)
    u_zero = loop.compute(Quaternion(), Quaternion.from_euler(0.1, 0.0, 0.0),
                          omega=(0.0, 0.0, 0.0), dt=0.0)
    u_neg = loop.compute(Quaternion(), Quaternion.from_euler(0.1, 0.0, 0.0),
                         omega=(0.0, 0.0, 0.0), dt=-1.0)
    assert u_zero == u_prev
    assert u_neg == u_prev


def test_nonfinite_inputs_hold_command_and_stay_finite():
    loop = IndiInnerLoop()
    u_prev = loop.compute(Quaternion(), Quaternion.from_euler(0.1, 0.0, 0.0),
                          omega=(0.0, 0.0, 0.0), dt=DT)
    # non-finite gyro
    u_nan_w = loop.compute(Quaternion(), Quaternion.from_euler(0.1, 0.0, 0.0),
                           omega=(float("nan"), 0.0, 0.0), dt=DT)
    assert u_nan_w == u_prev
    assert all(math.isfinite(x) for x in u_nan_w)
    # non-finite dt
    u_inf_dt = loop.compute(Quaternion(), Quaternion.from_euler(0.1, 0.0, 0.0),
                            omega=(0.0, 0.0, 0.0), dt=float("inf"))
    assert u_inf_dt == u_prev
    assert all(math.isfinite(x) for x in u_inf_dt)


def test_ginv_floor_prevents_blowup_at_near_zero_g():
    """A near-zero effectiveness estimate must not blow up the increment: Ginv
    is floored, so the per-tick command change stays bounded."""
    loop = IndiInnerLoop(IndiConfig(g_inv_floor=0.2, g_seed=(1e-9, 2.1, 2.1),
                                    enable_online_g=False, max_rate=(8.0, 8.0, 8.0)))
    # With a near-zero g, 1/g would be enormous; the floor caps it at 1/0.2 = 5.
    u = loop.compute(Quaternion(), Quaternion.from_euler(0.1, 0.0, 0.0),
                     omega=(0.0, 0.0, 0.0), dt=DT)
    assert all(math.isfinite(x) for x in u)
    # The roll increment magnitude is bounded by ginv_floor^-1 * |alpha_des| and
    # then clamped to max_rate — finite and within envelope.
    assert abs(u[0]) <= 8.0 + 1e-9


def test_quat_error_vec_matches_attitude_error_law():
    """The INDI alpha_des reuses the SAME quaternion error vector as the PD law
    (conj(q_cur) (x) q_des, shortest-path). Cross-check against an independent
    computation and confirm shortest-path (w>=0) handling."""
    q_cur = Quaternion.from_euler(0.1, -0.2, 0.3)
    q_des = Quaternion.from_euler(-0.3, 0.4, 1.0)
    vec = _quat_error_vec(q_cur, q_des)
    # Independent: conj(qc) * qd via numpy, take vector part, flip if w<0.
    qc = np.array([q_cur.w, q_cur.x, q_cur.y, q_cur.z])
    qd = np.array([q_des.w, q_des.x, q_des.y, q_des.z])
    cw, cx, cy, cz = qc[0], -qc[1], -qc[2], -qc[3]
    ew = cw * qd[0] - cx * qd[1] - cy * qd[2] - cz * qd[3]
    ex = cw * qd[1] + cx * qd[0] + cy * qd[3] - cz * qd[2]
    ey = cw * qd[2] - cx * qd[3] + cy * qd[0] + cz * qd[1]
    ez = cw * qd[3] + cx * qd[2] - cy * qd[1] + cz * qd[0]
    if ew < 0:
        ex, ey, ez = -ex, -ey, -ez
    assert vec == pytest.approx([ex, ey, ez])
    # Identical orientation => zero error.
    assert _quat_error_vec(q_cur, q_cur) == pytest.approx([0.0, 0.0, 0.0])
