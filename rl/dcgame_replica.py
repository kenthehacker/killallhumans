"""Gray-box composite REPLICA of the DCGame sim's drone response.

WHAT THIS IS (read the CRITICAL REFRAME in the task / rl/README.md):
We do NOT command or observe the drone's true rigid-body physics. We send
body-rate setpoints + a normalized thrust over MAVLink to DCGame's CLOSED-SOURCE
inner autopilot, and we observe the resulting state. Mass / inertia / motor /
aero are a black box and are NOT separately identifiable. What IS observable —
and what RL trains on — is the COMPOSITE map

    (rate/attitude setpoint + thrust, current state) -> next state.

This module replicates THAT map, fit to telemetry, not first-principles physics.

STRUCTURE OF THE COMPOSITE MAP (the gray box)
---------------------------------------------
ROTATION (the part that carries the *bandwidth wall*):
  The official `aigp_mavlink.send_attitude` takes a desired attitude, converts
  it to a body-rate setpoint via the EXACT PD law `_attitude_error_body_rates`
  (kp=(1.0,0.5,0.5), kd=(0.4,0.2,0.2), per-axis clamp +/-0.8 rad/s), applies the
  sim sign convention `_rate_sign=(-1,+1,-1)`, and sends SET_ATTITUDE_TARGET in
  rate mode. DCGame's inner loop then *tracks* that rate setpoint imperfectly.

  We model the closed inner loop at the ATTITUDE level as a per-axis FIRST-ORDER
  LAG with a DC gain:

      d(att_i)/dt  =  ( clamp(rate_cmd_i, +/-w_max) * eff_i  -  achieved_rate_i )
                       ... integrated, but realised as a lag toward the
                       commanded attitude with time constant tau_i and gain g_i.

  Equivalently (and this is what we fit, because attitude is what telemetry
  observes directly and what the champion outputs):

      att_i[k+1] = att_i[k] + dt * ( g_i * att_cmd_i[k] - att_i[k] ) / tau_i.

  This is a closed-loop rate-tracking model: a rate command proportional to the
  attitude error, integrated, with a per-axis bandwidth 1/tau_i and DC gain g_i.
  The achieved BODY RATE is the achieved attitude's time-derivative, and the
  +/-0.8 rate clamp of the PD law is applied to the rate the loop actually
  commands, so the clamp still bounds how fast the bank can slew.

  THE 0.53 ROLL ATTENUATION + THE ~2 m/s DESCENT WALL EMERGE from tau_roll.
  Fitted tau_roll ~= 0.47 s. During a fast slalom turn (~1 s/turn) the roll
  command reverses before the slow roll lag reaches the commanded amplitude, so
  achieved/commanded roll amplitude collapses to ~0.49-0.53x (matched below) —
  the measured wall. A steep-descent leg needs the thrust vector banked/pitched
  to trade vertical for forward; the same lag caps how fast that vector can
  change, capping the sustainable vertical rate near ~2 m/s. Neither is hard-
  coded; both fall out of the rate-lag limiting the bank/thrust-vector slew.

TRANSLATION (the reused calibration model):
      accel_NED = R(att) @ [0, 0, -thrust_norm * k_t]  +  g_NED  -  k_d * vel
  with k_t = max_thrust/mass (accel per unit normalized thrust), k_d = drag/mass,
  fit by the SAME least-squares regression as competition/calibration.py. attitude
  is integrated from the achieved rate; vel and pos are integrated from accel.

Pure numpy, deterministic, fast. No torch, no PyBullet.

CONVENTIONS
-----------
* Frame: NED (x-North, y-East, z-Down). Gravity is +9.81 along +z.
* Attitude is FRD euler (roll, pitch, yaw), the same convention as the telemetry
  `roll/pitch/yaw` fields and `competition.adapter.Quaternion.from_euler/to_euler`.
* `thrust_norm` is the [0,1] offboard thrust command (cmd_thrust in telemetry).
* Body rate is rad/s, FRD, the same convention as the `gyro` telemetry field
  (NOTE: telemetry `gyro` carries a sign relative to the *commanded* rate that we
  deliberately do NOT reason about from first principles — we fit attitude, which
  is sign-unambiguous, and report body rate as the achieved attitude rate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence, Tuple

import numpy as np

# Gravity in NED (z-down): gravity acts along +z.
G_NED = np.array([0.0, 0.0, 9.81], dtype=float)
GRAVITY = 9.81

# The EXACT inner-loop PD gains / clamp the live adapter uses
# (competition/aigp_mavlink.py). Reproduced here so the attitude->rate wrapper is
# byte-identical to `_attitude_error_body_rates` and the replica's rate clamp
# matches the real signal chain.
ATT_RATE_KP: Tuple[float, float, float] = (1.0, 0.5, 0.5)   # (roll, pitch, yaw)
ATT_RATE_KD: Tuple[float, float, float] = (0.4, 0.2, 0.2)
ATT_RATE_MAX: float = 0.8                                   # rad/s per-axis clamp
RATE_SIGN: Tuple[float, float, float] = (-1.0, 1.0, -1.0)   # sim applies this


# --------------------------------------------------------------------------- #
# Attitude <-> body-rate wrapper (reuses the EXACT _attitude_error_body_rates) #
# --------------------------------------------------------------------------- #
def _quat_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """(w,x,y,z) from FRD euler — identical math to adapter.Quaternion.from_euler."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def attitude_error_body_rates(
    q_cur: Tuple[float, float, float, float],
    q_des: Tuple[float, float, float, float],
    omega: Sequence[float] = (0.0, 0.0, 0.0),
    kp=ATT_RATE_KP,
    kd=ATT_RATE_KD,
    max_rate=ATT_RATE_MAX,
) -> Tuple[float, float, float]:
    """PD body-rate command (FRD) driving q_cur -> q_des.

    BYTE-FOR-BYTE the same math as competition.aigp_mavlink._attitude_error_body_rates
    (the validated champion inner loop): error quat conj(q_cur) (x) q_des, vector
    part is sin(theta/2)*axis, 2*kp*vec is the proportional body rate, -kd*omega
    damps; shortest path via w>=0; per-axis clamp. q_cur/q_des are (w,x,y,z).
    Returns (roll_rate, pitch_rate, yaw_rate) BEFORE the sim _rate_sign (the
    caller / replica applies that, exactly as the adapter does).
    """
    kpx, kpy, kpz = (kp, kp, kp) if isinstance(kp, (int, float)) else kp
    kdx, kdy, kdz = (kd, kd, kd) if isinstance(kd, (int, float)) else kd
    qcw, qcx, qcy, qcz = q_cur
    qdw, qdx, qdy, qdz = q_des
    # conj(qc) (x) qd
    cw, cx, cy, cz = qcw, -qcx, -qcy, -qcz
    ew = cw * qdw - cx * qdx - cy * qdy - cz * qdz
    ex = cw * qdx + cx * qdw + cy * qdz - cz * qdy
    ey = cw * qdy - cx * qdz + cy * qdw + cz * qdx
    ez = cw * qdz + cx * qdy - cy * qdx + cz * qdw
    if ew < 0:
        ex, ey, ez = -ex, -ey, -ez
    rates = (
        2.0 * kpx * ex - kdx * omega[0],
        2.0 * kpy * ey - kdy * omega[1],
        2.0 * kpz * ez - kdz * omega[2],
    )
    mxx, mxy, mxz = (max_rate, max_rate, max_rate) if isinstance(
        max_rate, (int, float)) else max_rate
    return (
        max(-mxx, min(mxx, rates[0])),
        max(-mxy, min(mxy, rates[1])),
        max(-mxz, min(mxz, rates[2])),
    )


def attitude_to_body_rate(
    cur_euler: Sequence[float],
    des_euler: Sequence[float],
    omega: Sequence[float] = (0.0, 0.0, 0.0),
    kp=ATT_RATE_KP,
    kd=ATT_RATE_KD,
    max_rate=ATT_RATE_MAX,
    apply_rate_sign: bool = True,
) -> Tuple[float, float, float]:
    """Euler-in convenience wrapper around `attitude_error_body_rates`.

    Converts current/desired FRD euler to quaternions, runs the EXACT champion
    PD law, and (by default) applies the sim `_rate_sign=(-1,+1,-1)` — i.e. it
    reproduces the full body-rate vector that `aigp_mavlink.send_attitude` puts
    on the wire. Set ``apply_rate_sign=False`` to get the pre-sign PD output.
    """
    q_cur = _quat_from_euler(*cur_euler)
    q_des = _quat_from_euler(*des_euler)
    rr, pr, yr = attitude_error_body_rates(q_cur, q_des, omega, kp, kd, max_rate)
    if apply_rate_sign:
        sx, sy, sz = RATE_SIGN
        return (sx * rr, sy * pr, sz * yr)
    return (rr, pr, yr)


# --------------------------------------------------------------------------- #
# Parameters                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class ReplicaParams:
    """Fitted composite-map parameters. Persisted to rl/dcgame_params.json.

    The rotation block is the per-axis closed-loop attitude lag (tau) + DC gain
    (eff). The translation block is the reused calibration regression (k_t, k_d).
    Defaults are the values fit from captures/rel_1..15 (see rl/fit_dynamics.py);
    fit_dynamics overwrites them and re-saves.
    """
    # Translation (composite thrust/drag, accel per unit thrust & per unit vel).
    k_t: float = 31.0          # accel produced per unit normalized thrust (1/s^2 * ...)
    k_d: float = 0.10          # drag deceleration per unit velocity (1/s)

    # Rotation: per-axis closed-loop attitude lag (s) and DC gain (-).
    # tau_i = bandwidth of the inner loop tracking the commanded attitude;
    # eff_i = steady-state achieved/commanded attitude ratio.
    tau_roll: float = 0.47
    tau_pitch: float = 0.28
    tau_yaw: float = 0.08
    eff_roll: float = 1.05
    eff_pitch: float = 1.04
    eff_yaw: float = 1.0

    # Inner-loop PD gains / clamp the rate command is formed with (fixed — these
    # are the real adapter constants, not fit). Kept here so a single object
    # fully specifies the signal chain and tests can perturb them.
    rate_kp: Tuple[float, float, float] = ATT_RATE_KP
    rate_kd: Tuple[float, float, float] = ATT_RATE_KD
    rate_max: float = ATT_RATE_MAX

    # Provenance / fit diagnostics (informational).
    n_samples: int = 0
    translation_rmse: float = 0.0   # m/s^2 (calibration regression residual)
    roll_rate_rmse: float = 0.0     # rad/s (1-step attitude-rate residual)
    pitch_rate_rmse: float = 0.0
    yaw_rate_rmse: float = 0.0

    def tau(self) -> np.ndarray:
        return np.array([self.tau_roll, self.tau_pitch, self.tau_yaw], dtype=float)

    def eff(self) -> np.ndarray:
        return np.array([self.eff_roll, self.eff_pitch, self.eff_yaw], dtype=float)

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuples -> lists for clean JSON
        d["rate_kp"] = list(self.rate_kp)
        d["rate_kd"] = list(self.rate_kd)
        return d

    @staticmethod
    def from_dict(d: dict) -> "ReplicaParams":
        kw = dict(d)
        if "rate_kp" in kw:
            kw["rate_kp"] = tuple(kw["rate_kp"])
        if "rate_kd" in kw:
            kw["rate_kd"] = tuple(kw["rate_kd"])
        # Drop any unknown keys so older/newer JSON still loads.
        fields = set(ReplicaParams.__dataclass_fields__.keys())
        kw = {k: v for k, v in kw.items() if k in fields}
        return ReplicaParams(**kw)


# --------------------------------------------------------------------------- #
# State                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class ReplicaState:
    """Full replica state. NED position/velocity (m, m/s), FRD euler attitude
    (rad), FRD body rate (rad/s)."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    att: np.ndarray = field(default_factory=lambda: np.zeros(3))   # roll,pitch,yaw
    rate: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body rate

    def copy(self) -> "ReplicaState":
        return ReplicaState(self.pos.copy(), self.vel.copy(),
                            self.att.copy(), self.rate.copy())


# --------------------------------------------------------------------------- #
# Rotation matrix (FRD body -> NED world), euler ZYX                          #
# --------------------------------------------------------------------------- #
def rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R such that v_ned = R @ v_body, for FRD body / NED world, ZYX euler.

    Matches the standard aerospace 3-2-1 (yaw-pitch-roll) used throughout the
    stack (e.g. the thrust-vector extraction in minimal_controller). Body z (down)
    is the 3rd column; thrust along body -z therefore points up when level.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=float)


# --------------------------------------------------------------------------- #
# The replica                                                                 #
# --------------------------------------------------------------------------- #
class DCGameReplica:
    """Deterministic gray-box composite simulator of DCGame's drone response.

    Two entry points:
      * step(rate_cmd, thrust_norm, dt)      — body-rate-setpoint interface
                                               (what RL / MAVLink actually sends)
      * step_attitude(att_cmd, thrust_norm, dt) — attitude-setpoint convenience
                                               that runs the EXACT champion PD law
                                               internally, so the champion (which
                                               outputs attitude+thrust) drives the
                                               replica UNCHANGED for fidelity.

    Both share `_step_core`. Pure numpy; no hidden global state; same params +
    same inputs => same trajectory (deterministic).
    """

    def __init__(self, params: Optional[ReplicaParams] = None,
                 state: Optional[ReplicaState] = None):
        self.params = params or ReplicaParams()
        self.state = state.copy() if state is not None else ReplicaState()

    # -- construction helpers ------------------------------------------------ #
    def reset(self, pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
              att=(0.0, 0.0, 0.0), rate=(0.0, 0.0, 0.0)) -> "DCGameReplica":
        self.state = ReplicaState(
            np.asarray(pos, float).copy(), np.asarray(vel, float).copy(),
            np.asarray(att, float).copy(), np.asarray(rate, float).copy())
        return self

    # -- the composite map --------------------------------------------------- #
    def _rotation_update(self, rate_cmd: np.ndarray, dt: float) -> None:
        """Advance attitude + achieved body rate from a desired-rate setpoint.

        rate_cmd is the DESIRED ACHIEVED body rate in the attitude-error-reducing
        direction (FRD): either a raw RL rate command, or the PD law's output that
        drives the achieved attitude toward the commanded attitude. The achieved
        rate tracks rate_cmd through the per-axis first-order lag with DC gain
        eff_i and time constant tau_i, after the +/-rate_max clamp. Attitude
        integrates the achieved rate.

        IMPORTANT SIGN NOTE (the composite-map reframe): we model the OBSERVED map
        commanded-attitude -> achieved-attitude, which the telemetry shows is
        correct-direction tracking with DC gain ~1. The adapter's `_rate_sign`
        =(-1,+1,-1) exists so that AFTER DCGame's own internal sign flip the loop
        is negative feedback; that flip is already absorbed into the observed map.
        So we drive the lag with the attitude-error-reducing rate directly and do
        NOT re-apply `_rate_sign` here (doing so would double-flip and diverge).
        `_rate_sign` / attitude_to_body_rate(apply_rate_sign=True) remain available
        to reproduce the exact on-the-wire vector for the wrapper-equivalence test.

        The DC gain eff_i scales the rate target; with the champion's rate command
        proportional to attitude error the closed loop converges to ~commanded
        attitude (eff~1). tau_i is the bandwidth that makes the 0.53 roll
        attenuation and the ~2 m/s descent wall EMERGE (verified, not hard-coded).
        """
        p = self.params
        w_max = p.rate_max
        clamped = np.clip(rate_cmd, -w_max, w_max)
        target = p.eff() * clamped
        tau = np.maximum(p.tau(), 1e-4)
        # Exact first-order step (stable for any dt): rate -> target with tau.
        alpha = 1.0 - np.exp(-dt / tau)
        new_rate = self.state.rate + alpha * (target - self.state.rate)
        # Integrate attitude with the trapezoidal mean rate (more accurate than
        # either endpoint for a lagging first-order response).
        self.state.att = self.state.att + 0.5 * (self.state.rate + new_rate) * dt
        self.state.rate = new_rate
        # Wrap yaw to (-pi, pi] so it never winds up.
        self.state.att[2] = math.atan2(math.sin(self.state.att[2]),
                                       math.cos(self.state.att[2]))

    def _translation_update(self, thrust_norm: float, dt: float) -> None:
        """Advance velocity + position from the calibration thrust/drag model.

        DCGAME ROLL-SIGN CONVENTION (composite map, telemetry-confirmed): in the
        real sim a +roll (the value reported in telemetry, which is what the
        controller reads back) produces +Y (East) thrust-accel at yaw=pi — the
        OPPOSITE of the standard aerospace ZYX convention (corr(roll, a_Y)=+0.49
        across the captures; the minimal controller carries a matching `roll=-roll`
        inversion calibrated to exactly this). So for the thrust-vector mapping we
        negate roll, reproducing the sim's frame convention. This keeps the replica
        sign-consistent with the unmodified champion controller (the fidelity-gate
        requirement) and with the logged dynamics. Pitch/yaw use the standard sign
        (pitch's separate inner-loop sign lives in `_rate_sign`, already handled).
        """
        p = self.params
        roll, pitch, yaw = self.state.att
        R = rotation_body_to_ned(-roll, pitch, yaw)
        thrust_body = np.array([0.0, 0.0, -thrust_norm * p.k_t])  # along body -z
        accel = R @ thrust_body + G_NED - p.k_d * self.state.vel
        # Semi-implicit (symplectic) Euler: update vel, then pos with new vel.
        self.state.vel = self.state.vel + accel * dt
        self.state.pos = self.state.pos + self.state.vel * dt

    def _step_core(self, rate_cmd: np.ndarray, thrust_norm: float, dt: float) -> ReplicaState:
        dt = float(dt)
        if not (dt > 0.0 and math.isfinite(dt)):
            return self.state.copy()
        thrust_norm = float(np.clip(thrust_norm, 0.0, 1.0))
        self._rotation_update(np.asarray(rate_cmd, float), dt)
        self._translation_update(thrust_norm, dt)
        return self.state.copy()

    def step(self, rate_cmd: Sequence[float], thrust_norm: float, dt: float
             ) -> ReplicaState:
        """Body-rate-setpoint step (the native RL / MAVLink interface).

        rate_cmd is the (roll,pitch,yaw) body-rate setpoint in FRD, interpreted as
        the DESIRED ACHIEVED rate (RL emits the rate it wants the airframe to
        track). The achieved rate follows it through the per-axis lag/clamp; this
        is the composite rotation map. The sim `_rate_sign` is the on-the-wire
        convention only and is intentionally NOT applied here (see
        `_rotation_update`'s sign note) — for the exact wire vector use
        `attitude_to_body_rate(..., apply_rate_sign=True)`.
        """
        return self._step_core(np.asarray(rate_cmd, float), thrust_norm, dt)

    def step_attitude(self, att_cmd: Sequence[float], thrust_norm: float,
                      dt: float) -> ReplicaState:
        """Attitude-setpoint step — runs the EXACT champion PD inner loop.

        att_cmd is the desired FRD euler (roll,pitch,yaw), i.e. exactly what
        `control.minimal_controller.MinimalController.compute` / the champion
        emit. Internally this calls the byte-identical `_attitude_error_body_rates`
        PD law to produce the body-rate that drives achieved attitude toward
        att_cmd (the attitude-error-reducing direction), then advances the
        composite map. The PD law / gains / clamp are identical to
        `aigp_mavlink.send_attitude`; the only thing not applied is the on-the-wire
        `_rate_sign` (already absorbed into the observed map — see the sign note).
        This lets the champion drive the replica UNCHANGED — the fidelity gate.
        """
        rate = attitude_to_body_rate(
            self.state.att, att_cmd, omega=self.state.rate,
            kp=self.params.rate_kp, kd=self.params.rate_kd,
            max_rate=self.params.rate_max, apply_rate_sign=False,
        )
        return self._step_core(np.asarray(rate, float), thrust_norm, dt)
