"""Continuous global racing line + curvature-limited speed profile.

A self-contained, pure-math (numpy + scipy) replacement for gate-by-gate
pure-pursuit. Instead of chasing one gate at a time, we fit ONE smooth C2
spline through every gate (a global racing line), re-parameterize it by true
arc length, and precompute a speed profile that:

  * caps speed by local curvature (so we can survive the lateral accel of a
    turn), and
  * smooths that cap forward/backward so the longitudinal accel needed to get
    between adjacent speeds is feasible (the cheap MPCC analog: brake BEFORE a
    turn, accelerate out of it).

Coordinate frame: NED (x north, y east, z down), metres.

No simulator, no MAVLink, no project imports -- just numpy and scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


class RacingSpline:
    """A global racing line through ordered waypoints with a speed profile.

    The curve is C2 (cubic spline) and is queried by ARC LENGTH ``s`` in metres,
    ``s`` in ``[0, self.length]``. All grid quantities (position, unit tangent,
    curvature, speed) are precomputed on a uniform ``s`` grid for cheap lookup.
    """

    def __init__(
        self,
        waypoints,
        v_max,
        a_lat_max,
        a_long_max,
        v_min=4.0,
        samples=2000,
        v_descent_max=None,
        v_final_cap=None,
        final_region_m=0.0,
    ):
        """Fit the spline and precompute the arc-length grid + speed profile.

        Parameters
        ----------
        waypoints : array_like, shape (N, 3)
            Points the curve must pass through, IN ORDER (NED metres).
        v_max : float
            Maximum allowed speed (m/s).
        a_lat_max : float
            Maximum lateral (centripetal) accel (m/s^2). Sets the curvature cap.
        a_long_max : float
            Maximum longitudinal accel/decel (m/s^2). Sets how fast speed may
            change along the path.
        v_min : float, optional
            Floor speed in tight turns (m/s). Default 4.0.
        samples : int, optional
            Number of uniform-arc-length grid points. Default 2000.
        v_descent_max : float, optional
            Cap on the VERTICAL descent RATE (m/s). On a descending course the
            quadrotor destabilises if it descends too fast WHILE moving fast
            horizontally (the thrust vector swings and the roll extraction blows
            up), so steep legs must be flown slower: speed is capped so
            v * |tangent_z| <= v_descent_max. None = no descent cap (pure
            curvature limiting). Default None.
        """
        self.waypoints = np.asarray(waypoints, dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("waypoints must have shape (N, 3)")
        if self.waypoints.shape[0] < 2:
            raise ValueError("need at least 2 waypoints")

        self.v_max = float(v_max)
        self.a_lat_max = float(a_lat_max)
        self.a_long_max = float(a_long_max)
        self.v_min = float(v_min)
        self.samples = int(samples)
        self.v_descent_max = None if v_descent_max is None else float(v_descent_max)
        self.v_final_cap = None if v_final_cap is None else float(v_final_cap)
        self.final_region_m = float(final_region_m)

        # ------------------------------------------------------------------
        # 1) Fit a C2 cubic spline parameterized by a CHORD-LENGTH proxy.
        #    Chord length (cumulative straight-line distance between waypoints)
        #    is a good monotone parameter; the true arc length comes later.
        # ------------------------------------------------------------------
        chord = self._chord_param(self.waypoints)
        # One independent cubic spline per axis, sharing the chord parameter.
        # natural BCs ("natural") are fine and stable for an open path.
        self._spline_u = CubicSpline(chord, self.waypoints, axis=0, bc_type="natural")
        self._u_max = float(chord[-1])

        # ------------------------------------------------------------------
        # 2) RE-PARAMETERIZE by true arc length.
        #    Densely sample the chord-parameterized curve, integrate |dP/du|
        #    to get cumulative arc length s(u), then build s -> u so that we
        #    can evaluate the curve at uniform arc-length steps.
        # ------------------------------------------------------------------
        dense_n = max(20 * self.samples, 20000)
        u_dense = np.linspace(0.0, self._u_max, dense_n)
        p_dense = self._spline_u(u_dense)                 # (dense_n, 3)
        seg = np.linalg.norm(np.diff(p_dense, axis=0), axis=1)
        s_dense = np.concatenate([[0.0], np.cumsum(seg)])  # (dense_n,)
        self.length = float(s_dense[-1])

        # Map s -> u (monotone). Used only to build the arc-length-indexed
        # interpolators below.
        u_of_s = np.interp  # closure-free; we interp inline

        # Arc-length-indexed position interpolators x(s), y(s), z(s).
        # Build by sampling the chord curve at the u that corresponds to each
        # uniform s, giving us position-vs-arclength, then a CubicSpline on s.
        s_grid = np.linspace(0.0, self.length, self.samples)
        u_grid = u_of_s(s_grid, s_dense, u_dense)         # u for each uniform s
        p_grid = self._spline_u(u_grid)                   # (samples, 3)

        # x(s), y(s), z(s) as a single vector-valued natural cubic spline on s.
        self._spline_s = CubicSpline(s_grid, p_grid, axis=0, bc_type="natural")

        # ------------------------------------------------------------------
        # 3) Precompute grid quantities on the uniform s grid.
        # ------------------------------------------------------------------
        self.s_grid = s_grid
        self.ds = self.length / (self.samples - 1) if self.samples > 1 else 0.0

        d1 = self._spline_s(s_grid, 1)                    # P'(s),  (samples, 3)
        d2 = self._spline_s(s_grid, 2)                    # P''(s), (samples, 3)

        self.P = p_grid                                   # position
        speed_param = np.linalg.norm(d1, axis=1)          # |P'(s)| (~1 by arc len)
        speed_param = np.maximum(speed_param, 1e-9)
        self.T = d1 / speed_param[:, None]                # unit tangent

        # curvature kappa = |P' x P''| / |P'|^3
        cross = np.cross(d1, d2)
        self.kappa = np.linalg.norm(cross, axis=1) / speed_param**3

        # ------------------------------------------------------------------
        # 4) Speed profile v(s).
        # ------------------------------------------------------------------
        self.v = self._build_speed_profile()

    # ----------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _chord_param(pts):
        """Cumulative chord length used as the spline's fitting parameter."""
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    def _build_speed_profile(self):
        """Curvature cap + forward/backward longitudinal-accel smoothing."""
        ds = self.ds

        # 1) curvature-limited cap: v_curve = sqrt(a_lat / kappa), clipped.
        kappa_safe = np.maximum(self.kappa, 1e-6)
        v = np.sqrt(self.a_lat_max / kappa_safe)
        v = np.clip(v, self.v_min, self.v_max)

        # 1b) descent-rate cap: keep the VERTICAL speed v*|T_z| <= v_descent_max
        #     so steep descending legs are flown slowly enough that the thrust
        #     vector stays near-vertical (a fast horizontal + fast descent swings
        #     it and destabilises). Clipped to [v_min, v_max] so it never drives
        #     below the turn floor.
        if self.v_descent_max is not None:
            tz = np.abs(self.T[:, 2])
            v_desc = self.v_descent_max / np.maximum(tz, 1e-6)
            v = np.minimum(v, np.clip(v_desc, self.v_min, self.v_max))

        # 1c) FINAL-REGION brake: the closing slalom reversal (last gates) has
        #     GENTLE spline curvature so the curvature cap doesn't slow it, but
        #     the drone can't bank the final lateral move fast enough at speed
        #     (undershoot). Cap the speed over the last `final_region_m` metres
        #     (mirrors the gate-by-gate --final-brake-band). The fwd/bwd pass
        #     below then makes the decel INTO it gradual (bounds the balloon).
        if self.v_final_cap is not None and self.final_region_m > 0.0:
            in_final = self.s_grid >= (self.length - self.final_region_m)
            v[in_final] = np.minimum(v[in_final], max(self.v_final_cap, self.v_min))

        # 2a) BACKWARD pass: limit how fast we may be going at i given that we
        #     must be able to DECELERATE to v[i+1] within ds. (Brake into turn.)
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * self.a_long_max * ds))

        # 2b) FORWARD pass: limit how fast we may be at i given we could only
        #     have ACCELERATED from v[i-1] over ds. (Accelerate out of turn.)
        for i in range(1, len(v)):
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * self.a_long_max * ds))

        return v

    # ----------------------------------------------------------------------
    # Query API
    # ----------------------------------------------------------------------
    def _clamp_s(self, s):
        return float(min(max(s, 0.0), self.length))

    def point_at(self, s):
        """Position (3,) at arc length s (clamped to [0, length])."""
        return self._spline_s(self._clamp_s(s))

    def tangent_at(self, s):
        """Unit tangent (3,) at arc length s (clamped)."""
        d1 = self._spline_s(self._clamp_s(s), 1)
        n = np.linalg.norm(d1)
        return d1 / n if n > 1e-12 else d1

    def curvature_at(self, s):
        """Curvature kappa (float) at arc length s (clamped)."""
        sc = self._clamp_s(s)
        d1 = self._spline_s(sc, 1)
        d2 = self._spline_s(sc, 2)
        n1 = np.linalg.norm(d1)
        if n1 < 1e-12:
            return 0.0
        return float(np.linalg.norm(np.cross(d1, d2)) / n1**3)

    def speed_at(self, s):
        """Profiled target speed (float) at arc length s (clamped)."""
        return float(np.interp(self._clamp_s(s), self.s_grid, self.v))

    def project(self, pos):
        """Arc length of the nearest grid point to ``pos``, with a refinement.

        A single distance argmin over the precomputed position grid finds the
        closest sample; we then refine within the neighbouring +/- ds window by
        projecting onto the local tangent for sub-grid accuracy.
        """
        pos = np.asarray(pos, dtype=float)
        d2 = np.sum((self.P - pos) ** 2, axis=1)
        i = int(np.argmin(d2))
        s0 = self.s_grid[i]

        # Sub-grid refinement: project (pos - P[i]) onto the unit tangent T[i]
        # and step by that scalar (clamped to one grid cell to stay local).
        delta = float(np.dot(pos - self.P[i], self.T[i]))
        delta = max(-self.ds, min(self.ds, delta))
        return self._clamp_s(s0 + delta)

    def aim(self, pos, lookahead_m):
        """Pure-pursuit-style aim point + target speed.

        Returns the point a ``lookahead_m`` arc-length ahead of the current
        projection (the geometric target to fly toward) and the profiled speed
        for the CURRENT progress -- so the drone slows BEFORE entering a turn
        rather than after.

        Returns
        -------
        aim_point : np.ndarray shape (3,)
        target_speed : float
        """
        s0 = self.project(pos)
        s_aim = min(s0 + float(lookahead_m), self.length)
        return self.point_at(s_aim), self.speed_at(s0)


# --------------------------------------------------------------------------
# Quick sanity dump
# --------------------------------------------------------------------------
if __name__ == "__main__":
    GATES = np.array(
        [
            [0.0, 0.0, 0.0],       # start
            [-23.3, -0.4, -0.9],   # g0
            [-46.9, -2.5, 4.2],    # g1
            [-74.6, 1.2, 12.8],    # g2
            [-111.5, -5.1, 23.7],  # g3
            [-135.5, -0.8, 24.5],  # g4
            [-159.2, -4.4, 25.1],  # g5
        ]
    )

    # a_lat_max=5.0 (not the originally suggested 12): the VQ1 course is gentle
    # (min turn radius ~30 m), so a_lat_max>=~8.5 leaves the curvature cap above
    # v_max and the profile flat at v_max. 5.0 makes the limiter visibly engage.
    rs = RacingSpline(GATES, v_max=16.0, a_lat_max=5.0, a_long_max=12.0, v_min=6.0)

    chord_sum = float(np.sum(np.linalg.norm(np.diff(GATES, axis=0), axis=1)))
    i_min = int(np.argmin(rs.v))

    print("=== RacingSpline sanity dump (VQ1 gate set) ===")
    print(f"waypoints           : {GATES.shape[0]}")
    print(f"straight chord sum  : {chord_sum:8.2f} m")
    print(f"total spline length : {rs.length:8.2f} m  ({rs.length / chord_sum:.3f}x chord)")
    print(f"speed profile  min  : {rs.v.min():6.2f} m/s")
    print(f"speed profile  max  : {rs.v.max():6.2f} m/s")
    print(f"min speed at s      : {rs.s_grid[i_min]:8.2f} m  "
          f"(fraction {rs.s_grid[i_min] / rs.length:.3f} along course)")
    print(f"  position there    : {np.round(rs.point_at(rs.s_grid[i_min]), 2)}")
    print(f"  curvature there   : {rs.kappa[i_min]:.4f} 1/m")

    # A few low-speed (turn) regions for context.
    order = np.argsort(rs.v)
    seen = []
    print("lowest-speed s locations (deduped ~5m):")
    for idx in order:
        s = rs.s_grid[idx]
        if all(abs(s - p) > 5.0 for p in seen):
            seen.append(s)
            print(f"    s={s:7.2f} m  v={rs.v[idx]:5.2f} m/s  kappa={rs.kappa[idx]:.4f}")
        if len(seen) >= 4:
            break
