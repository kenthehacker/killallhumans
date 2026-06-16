"""Gate-map integrity / corruption monitor (VQ2 robustness item 3).

The AIGP sim delivers the gate map at runtime over a chunked SIM_RESET track
transfer (:func:`competition.aigp_messages.parse_track_data`). After ~25
runs/session the DCGame process degrades and that transfer starts returning
GARBAGE — observed signatures (see
``docs/aigp/2026-06-16-speed-and-spline-handoff.md`` "Operational notes"):

* gate positions ~1 km out of bounds (already caught by the old check),
* **sign-flipped X** (the healthy course runs x in ~[-160, 0]; a flip puts the
  gates at x in ~[0, 160]),
* **Z ≈ −350** (the healthy course runs z in ~[-1, 27]; a flip+offset drives z
  far negative),
* a **uniform offset** / drift of the whole map that stays inside the generous
  bounds and so slips past a pure bounding-box check.

The deep-research report (2026-06-16, Part 2, "(3)+(4) Map / state integrity")
flags this as the team's #1 race-day reliability failure and notes there is NO
published quadrotor-racing precedent for it — it is a purpose-built consistency
test adapted from aerospace integrity practice. This module is that test.

Design constraints (CLAUDE.md): pure-python / numpy-only, **no new
dependencies, and NO MAVLink import** so it unit-tests fully offline. It takes
anything with ``.position`` (a 3-tuple/sequence NED) — real ``GateSpec``s or a
minimal stand-in — and NEVER raises on bad input; it *diagnoses* it.

The single entry point is :func:`check_gate_map`, which runs, in order:

1. empty / non-finite checks,
2. an OUTER bounds floor that is a strict superset of the legacy
   ``_gate_map_is_sane`` box (so behaviour can only get stricter, never looser),
3. expected-signed-region + single-axis sign-flip detection (no reference
   needed) — this is what nails "sign-flipped X" and "Z ≈ −350",
4. self-consistency invariants (gate count, inter-gate spacing, polyline
   length, z-span) derived from the real VQ1 course geometry,
5. optional reference comparison (a known-good map) — catches the *uniform
   offset / drift* the bounds miss, and a single-gate outlier.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Bounds & expected-region constants
# ---------------------------------------------------------------------------
# OUTER bounds floor — byte-identical to the legacy ``_gate_map_is_sane`` box so
# this module is a strict superset of the existing check (it can only reject
# MORE, never fewer, maps). Generous: only true ~1 km-out corruption trips it.
DEFAULT_BOUNDS: "Bounds"  # forward ref; defined after the dataclass below.

# EXPECTED SIGNED REGION — the *healthy* VQ1 course (FakeAdapter.VQ1_GATES /
# the live first-contact capture): x in [-159.2, -23.3], y in [-5.1, 1.2],
# z in [-0.03, 25.97]. The signed region is deliberately a little looser than
# the observed extent (course variation / the baked aim-z offset shift z) but
# MUCH tighter than the outer bounds, so a sign flip on x or z lands well
# outside it. Rationale per axis:
#   x: healthy gates are all <= ~0 (course runs out along -X). A sign flip makes
#      them all >= ~0 -> violates the [-, ~+margin] band.
#   z: healthy gates are all >= ~-1 (start near ground, climb to +26 in -Z-up
#      NED... note z here is the raw NED value, +Z down, so the course numbers
#      are POSITIVE-down/​"below origin" — the point is only the SIGN/RANGE).
#      Z ≈ -350 lands far below the floor.
# These bound the SIGNED region used for flip detection; the looser numeric pad
# (EXPECTED_*_PAD) keeps legitimate jitter / aim-offset inside.
EXPECTED_X_MIN: float = -200.0   # course max-extent ~-159; pad to -200
EXPECTED_X_MAX: float = 5.0      # healthy gates <= ~0; +5 m pad for jitter/offset
EXPECTED_Z_MIN: float = -10.0    # healthy gates >= ~-1 (NED); -10 m pad
EXPECTED_Z_MAX: float = 40.0     # healthy gates <= ~26; +14 m pad for aim-z / climb

# SELF-CONSISTENCY ranges, derived from the real VQ1 course (6 gates, ~136 m
# x-span). Measured consecutive-gate spacings on the healthy map are
# [24.2, 29.2, 39.0, 24.4, 24.0] m, polyline length ~140.8 m, z-span ~26.0 m.
# Ranges are padded generously around those measurements so legitimate course
# variation passes but a scrambled/teleported gate (neighbour absurdly far or
# coincident) is caught:
#   spacing in [2, 60] m  — floor 2 m (gates can't be ~coincident; the tightest
#       real leg is 24 m, 2 m leaves vast margin yet still catches a collapsed
#       duplicate); ceiling 60 m (~1.5x the widest real 39 m leg, well under a
#       200 m teleport).
#   polyline in [40, 320] m — ~0.3x..2.3x the real 141 m; catches both a
#       collapsed map and one stretched by a far-flung gate.
#   z-span in [3, 120] m — the real course climbs 26 m; <3 m would mean a flat
#       map (lost the climb), >120 m a wildly displaced gate.
MIN_GATE_SPACING_M: float = 2.0
MAX_GATE_SPACING_M: float = 60.0
MIN_POLYLINE_LEN_M: float = 40.0
MAX_POLYLINE_LEN_M: float = 320.0
MIN_Z_SPAN_M: float = 3.0
MAX_Z_SPAN_M: float = 120.0

# REFERENCE-COMPARISON tolerances. A healthy re-fetch of the same map differs
# only by float jitter (sub-metre); a uniform offset / drift is metres. We call
# the per-gate residual field a "uniform offset" when every gate is displaced by
# nearly the SAME vector (low spread) by a meaningful amount (mean above a
# floor) — that is the drift signature the bounds miss. A single large outlier
# with the rest tight is a per-gate "reference_mismatch".
REF_JITTER_TOL_M: float = 0.5        # residuals at/below this are "no change"
REF_UNIFORM_SPREAD_TOL_M: float = 1.0  # max deviation from the mean offset to
#                                        still call the field "uniform"
REF_OUTLIER_FACTOR: float = 4.0      # a gate this many x the median residual
#                                      (and absolutely large) is an outlier


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned NED bounding box used as the hard outer floor."""
    x_min: float = -300.0
    x_max: float = 20.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = -50.0
    z_max: float = 60.0

    def contains(self, p: Sequence[float]) -> bool:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


# Default outer floor == the legacy box. Defined here now that ``Bounds`` exists.
DEFAULT_BOUNDS = Bounds()


# Allowed diagnosis codes (kept as a frozenset so callers/tests can assert
# against the canonical set without importing string literals piecemeal).
DIAGNOSES = frozenset(
    {
        "ok",
        "empty",
        "non_finite",
        "out_of_bounds",
        "sign_flip_x",
        "sign_flip_z",
        "uniform_offset",
        "gate_count",
        "spacing_anomaly",
        "reference_mismatch",
    }
)


@dataclass
class GateMapVerdict:
    """Result of a gate-map integrity check.

    Never carries an exception — a bad map is *diagnosed*, not raised. ``ok`` is
    the single bool the existing runner contract needs; ``diagnosis`` is one of
    :data:`DIAGNOSES`; ``message`` is human-readable; ``suggested_correction``
    is set only when the fix is UNAMBIGUOUS (e.g. ``"negate_x"`` for a clean
    single-axis flip, or an offset vector string for a uniform drift).
    """

    ok: bool
    diagnosis: str
    message: str
    suggested_correction: Optional[str] = None
    # Optional structured diagnostics (offset vector, residual stats, etc.) for
    # logging — never load-bearing for the bool contract.
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.diagnosis not in DIAGNOSES:
            # Defensive: a typo in a diagnosis code is itself a bug, but this
            # monitor must never raise from the happy path. Surface it in the
            # message rather than crashing the runner.
            self.message = f"[bad diagnosis code {self.diagnosis!r}] " + self.message


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _positions(gates: Sequence) -> Optional[np.ndarray]:
    """Extract an (N, 3) float array of NED positions from a gate sequence.

    Accepts anything with a ``.position`` attribute that is a length-3
    sequence (real ``GateSpec`` or a minimal stand-in). Returns ``None`` if the
    structure is unusable (so the caller diagnoses it rather than crashing).
    """
    try:
        rows = []
        for g in gates:
            p = getattr(g, "position", None)
            if p is None:
                p = g  # allow a raw (x, y, z) sequence too
            x, y, z = p[0], p[1], p[2]
            rows.append((float(x), float(y), float(z)))
        if not rows:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError, IndexError):
        return None


def _in_expected_signed_region(pos: np.ndarray) -> bool:
    """True iff every gate sits in the healthy SIGNED region (x<=~0, z in range).

    This is the sign/region test the bounding box can't do: a sign flip stays
    inside the (sign-agnostic) outer bounds but leaves this signed band.
    """
    if pos.size == 0:
        return False
    x, _, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return bool(
        np.all(x >= EXPECTED_X_MIN)
        and np.all(x <= EXPECTED_X_MAX)
        and np.all(z >= EXPECTED_Z_MIN)
        and np.all(z <= EXPECTED_Z_MAX)
    )


def _try_sign_flips(pos: np.ndarray) -> Optional[Tuple[str, str]]:
    """If a single-axis (or both-axis) negation brings the map back INTO the
    expected signed region, return ``(diagnosis, suggested_correction)``.

    Tries, in priority order, negate-x, negate-z, negate-both. Returns the first
    that lands the WHOLE map back in the signed region. Priority is deliberate:
    a pure X flip and a pure Z flip are the two named live failures; "both" is a
    rarer combined corruption. Returns ``None`` if no single negation recovers
    the map (then it is generic out-of-bounds / a worse corruption).
    """
    candidates = (
        ("sign_flip_x", "negate_x", np.array([-1.0, 1.0, 1.0])),
        ("sign_flip_z", "negate_z", np.array([1.0, 1.0, -1.0])),
        # both-axis flip is reported under sign_flip_x with a combined
        # correction so the diagnosis still names an axis it fixes; callers key
        # off ``suggested_correction`` for the exact operation.
        ("sign_flip_x", "negate_x_z", np.array([-1.0, 1.0, -1.0])),
    )
    for diagnosis, correction, mult in candidates:
        if _in_expected_signed_region(pos * mult):
            return diagnosis, correction
    return None


def _match_to_reference(
    pos: np.ndarray, ref: np.ndarray
) -> Optional[np.ndarray]:
    """Return per-gate residual vectors (received - reference), shape (N, 3).

    Matches received gate ``i`` to reference gate ``i`` (gates are delivered in
    pass order). If counts differ, returns ``None`` (count handled separately).
    As a robustness fallback for a *reordered* map, also tries nearest-by-id
    matching (greedy nearest reference per received gate) and keeps whichever
    pairing yields the smaller total residual — so a benign reshuffle is matched
    on geometry, not index.
    """
    if pos.shape[0] != ref.shape[0] or pos.shape[0] == 0:
        return None

    # Index-aligned residuals (the common case: same order).
    resid_idx = pos - ref
    cost_idx = float(np.sum(np.linalg.norm(resid_idx, axis=1)))

    # Nearest-neighbour pairing (greedy) as a fallback for reordering.
    n = pos.shape[0]
    used = [False] * n
    resid_nn = np.empty_like(pos)
    cost_nn = 0.0
    ok_nn = True
    for i in range(n):
        best_j, best_d = -1, math.inf
        for j in range(n):
            if used[j]:
                continue
            d = float(np.linalg.norm(pos[i] - ref[j]))
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0:
            ok_nn = False
            break
        used[best_j] = True
        resid_nn[i] = pos[i] - ref[best_j]
        cost_nn += best_d

    if ok_nn and cost_nn < cost_idx:
        return resid_nn
    return resid_idx


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def check_gate_map(
    gates: Sequence,
    *,
    reference: Optional[Sequence] = None,
    expected_count: int = 6,
    bounds: Bounds = DEFAULT_BOUNDS,
) -> GateMapVerdict:
    """Diagnose a transferred gate map. NEVER raises — returns a verdict.

    Args:
        gates: sequence of objects with a ``.position`` NED 3-tuple (real
            ``GateSpec`` or a stand-in), or raw ``(x, y, z)`` sequences.
        reference: optional known-good map (same shape) to compare against —
            enables uniform-offset / drift and single-gate-outlier detection.
        expected_count: expected number of gates (VQ1 = 6).
        bounds: outer hard floor (defaults to the legacy generous box).

    Returns:
        A :class:`GateMapVerdict`. ``ok`` is the bool the runner keys off.

    Order of checks (first failure wins, so the most fundamental corruption is
    reported): empty -> non-finite -> outer bounds -> sign-flip/region ->
    self-consistency -> reference drift/outlier.
    """
    pos = _positions(gates)

    # 1a. Unusable structure or empty.
    if pos is None:
        return GateMapVerdict(
            ok=False,
            diagnosis="empty",
            message="gate map could not be read (no usable .position fields)",
        )
    n = pos.shape[0]
    if n == 0:
        return GateMapVerdict(
            ok=False, diagnosis="empty", message="gate map is empty (0 gates)"
        )

    # 1b. Non-finite (NaN / inf) anywhere.
    if not np.all(np.isfinite(pos)):
        bad = int(np.count_nonzero(~np.all(np.isfinite(pos), axis=1)))
        return GateMapVerdict(
            ok=False,
            diagnosis="non_finite",
            message=f"gate map has non-finite coordinates in {bad} gate(s)",
        )

    # 2. OUTER bounds floor (strict superset of the legacy box).
    oob = [i for i in range(n) if not bounds.contains(pos[i])]
    if oob:
        # A sign flip can ALSO be out of bounds (e.g. Z ≈ -350 < z_min). Prefer
        # the more SPECIFIC sign-flip diagnosis when a single negation recovers
        # the map, so the runner gets an actionable correction instead of a
        # generic "out of bounds".
        flip = _try_sign_flips(pos)
        if flip is not None:
            diagnosis, correction = flip
            return GateMapVerdict(
                ok=False,
                diagnosis=diagnosis,
                message=(
                    f"{len(oob)} gate(s) out of bounds, but a single-axis "
                    f"negation ({correction}) restores the expected course "
                    f"region — likely a sign-flipped transfer"
                ),
                suggested_correction=correction,
                details={"out_of_bounds_indices": oob},
            )
        worst = pos[oob[0]]
        # Name which axis/axes are violated so a uniform z corruption (e.g.
        # Z ≈ -350) is reported as a Z fault, not a faceless "out of bounds".
        axes = []
        if np.any(pos[:, 0] < bounds.x_min) or np.any(pos[:, 0] > bounds.x_max):
            axes.append("x")
        if np.any(pos[:, 1] < bounds.y_min) or np.any(pos[:, 1] > bounds.y_max):
            axes.append("y")
        if np.any(pos[:, 2] < bounds.z_min) or np.any(pos[:, 2] > bounds.z_max):
            axes.append("z")
        axis_str = "/".join(axes) if axes else "?"
        return GateMapVerdict(
            ok=False,
            diagnosis="out_of_bounds",
            message=(
                f"{len(oob)} gate(s) out of bounds on {axis_str} "
                f"(e.g. gate {oob[0]} at "
                f"({worst[0]:.1f}, {worst[1]:.1f}, {worst[2]:.1f}))"
            ),
            details={"out_of_bounds_indices": oob, "axes": axes},
        )

    # 3. Expected signed region + sign-flip (in-bounds but wrong sign/region).
    #    The classic "sign-flipped X" can stay inside the symmetric outer box
    #    (x in [-300, 20] admits +x up to 20) yet leave the signed region.
    if not _in_expected_signed_region(pos):
        flip = _try_sign_flips(pos)
        if flip is not None:
            diagnosis, correction = flip
            return GateMapVerdict(
                ok=False,
                diagnosis=diagnosis,
                message=(
                    f"map is in-bounds but outside the expected course region; "
                    f"a single-axis negation ({correction}) restores it — "
                    f"likely a sign-flipped transfer"
                ),
                suggested_correction=correction,
            )
        # In bounds, wrong region, no clean negation recovers it. Report which
        # axis looks wrong for a useful message.
        x, z = pos[:, 0], pos[:, 2]
        axis = "x" if (np.any(x < EXPECTED_X_MIN) or np.any(x > EXPECTED_X_MAX)) else "z"
        return GateMapVerdict(
            ok=False,
            diagnosis="out_of_bounds",
            message=(
                f"map sits outside the expected course region on {axis} but no "
                f"single-axis negation recovers it — unrecognised corruption"
            ),
        )

    # 4. Self-consistency invariants.
    #    4a. Gate count.
    if n != expected_count:
        return GateMapVerdict(
            ok=False,
            diagnosis="gate_count",
            message=f"expected {expected_count} gates, got {n}",
            details={"count": n},
        )

    #    4b. Inter-gate spacing (consecutive, in pass order).
    diffs = np.diff(pos, axis=0)
    spac = np.linalg.norm(diffs, axis=1)
    if spac.size:
        smin, smax = float(spac.min()), float(spac.max())
        if smin < MIN_GATE_SPACING_M or smax > MAX_GATE_SPACING_M:
            j = int(np.argmin(spac)) if smin < MIN_GATE_SPACING_M else int(np.argmax(spac))
            return GateMapVerdict(
                ok=False,
                diagnosis="spacing_anomaly",
                message=(
                    f"inter-gate spacing out of range [{MIN_GATE_SPACING_M}, "
                    f"{MAX_GATE_SPACING_M}] m: gates {j}->{j + 1} are "
                    f"{spac[j]:.1f} m apart"
                ),
                details={"spacings_m": [round(float(s), 2) for s in spac]},
            )

    #    4c. Polyline length + z-span.
    poly = float(spac.sum())
    if poly < MIN_POLYLINE_LEN_M or poly > MAX_POLYLINE_LEN_M:
        return GateMapVerdict(
            ok=False,
            diagnosis="spacing_anomaly",
            message=(
                f"total course polyline length {poly:.1f} m out of expected "
                f"range [{MIN_POLYLINE_LEN_M}, {MAX_POLYLINE_LEN_M}] m"
            ),
            details={"polyline_m": round(poly, 2)},
        )
    z_span = float(pos[:, 2].max() - pos[:, 2].min())
    if z_span < MIN_Z_SPAN_M or z_span > MAX_Z_SPAN_M:
        return GateMapVerdict(
            ok=False,
            diagnosis="spacing_anomaly",
            message=(
                f"course z-span {z_span:.1f} m out of expected range "
                f"[{MIN_Z_SPAN_M}, {MAX_Z_SPAN_M}] m"
            ),
            details={"z_span_m": round(z_span, 2)},
        )

    # 5. Reference comparison (uniform-offset / drift + single-gate outlier).
    if reference is not None:
        ref = _positions(reference)
        if ref is None or ref.shape[0] == 0:
            # A broken reference must not break a healthy run — treat as "no
            # reference" and pass the self-consistent map.
            return GateMapVerdict(
                ok=True,
                diagnosis="ok",
                message=(
                    "gate map passes all self-consistency checks "
                    "(reference unusable — skipped comparison)"
                ),
            )
        if ref.shape[0] != n:
            return GateMapVerdict(
                ok=False,
                diagnosis="gate_count",
                message=(
                    f"gate count {n} does not match reference count "
                    f"{ref.shape[0]}"
                ),
                details={"count": n, "reference_count": int(ref.shape[0])},
            )

        resid = _match_to_reference(pos, ref)
        if resid is not None:
            mags = np.linalg.norm(resid, axis=1)
            max_mag = float(mags.max())
            if max_mag <= REF_JITTER_TOL_M:
                # Sub-tolerance everywhere: legitimate float jitter -> PASS.
                return GateMapVerdict(
                    ok=True,
                    diagnosis="ok",
                    message=(
                        f"gate map matches the reference within tolerance "
                        f"(max residual {max_mag:.3f} m)"
                    ),
                    details={"max_residual_m": round(max_mag, 4)},
                )

            mean_offset = resid.mean(axis=0)
            spread = float(np.linalg.norm(resid - mean_offset, axis=1).max())
            mean_mag = float(np.linalg.norm(mean_offset))
            median_mag = float(np.median(mags))

            # Uniform offset / drift: every gate displaced by ~the same vector
            # (low spread) by a meaningful amount. This is the drift the bounds
            # miss and the reason the session reference exists.
            if spread <= REF_UNIFORM_SPREAD_TOL_M and mean_mag > REF_JITTER_TOL_M:
                ox, oy, oz = mean_offset
                return GateMapVerdict(
                    ok=False,
                    diagnosis="uniform_offset",
                    message=(
                        f"every gate is offset from the reference by ~the same "
                        f"vector ({ox:+.2f}, {oy:+.2f}, {oz:+.2f}) m "
                        f"(spread {spread:.2f} m) — uniform map drift, not "
                        f"per-gate jitter"
                    ),
                    suggested_correction=(
                        f"subtract_offset({ox:+.3f},{oy:+.3f},{oz:+.3f})"
                    ),
                    details={
                        "offset_m": [round(float(c), 4) for c in mean_offset],
                        "spread_m": round(spread, 4),
                    },
                )

            # Single-gate outlier: one gate far off while the rest are tight.
            if max_mag > REF_JITTER_TOL_M and (
                median_mag <= REF_JITTER_TOL_M
                or max_mag > REF_OUTLIER_FACTOR * max(median_mag, 1e-6)
            ):
                worst = int(np.argmax(mags))
                return GateMapVerdict(
                    ok=False,
                    diagnosis="reference_mismatch",
                    message=(
                        f"gate {worst} deviates {max_mag:.1f} m from the "
                        f"reference while the others stay within "
                        f"{median_mag:.2f} m — single-gate displacement"
                    ),
                    details={
                        "worst_gate": worst,
                        "max_residual_m": round(max_mag, 4),
                        "median_residual_m": round(median_mag, 4),
                    },
                )

            # Residuals are large but neither cleanly uniform nor a lone
            # outlier (e.g. several gates shifted differently). Flag as a
            # reference mismatch rather than silently passing.
            return GateMapVerdict(
                ok=False,
                diagnosis="reference_mismatch",
                message=(
                    f"map disagrees with the reference (max residual "
                    f"{max_mag:.1f} m, spread {spread:.1f} m) but the pattern "
                    f"is neither a clean uniform offset nor a single outlier"
                ),
                details={
                    "max_residual_m": round(max_mag, 4),
                    "spread_m": round(spread, 4),
                },
            )

    # All checks passed.
    return GateMapVerdict(
        ok=True,
        diagnosis="ok",
        message=f"gate map passes all integrity checks ({n} gates)",
        details={"count": n},
    )


# ---------------------------------------------------------------------------
# Reference-map JSON persistence (mirrors competition.calibration.py style)
# ---------------------------------------------------------------------------

_REF_SCHEMA_VERSION = 1


def write_reference_json(
    gates: Sequence, path: Union[str, Path]
) -> None:
    """Persist a known-good gate map as the session reference.

    Stores ONLY what the integrity check needs — the ordered NED positions —
    plus a gate_id when available, mirroring ``calibration.write_calibration_json``
    (flat JSON, indent=2, explicit schema version). Writes the parent dir if it
    does not exist so a default ``captures/`` path works on a fresh checkout.
    """
    pos = _positions(gates)
    if pos is None or pos.shape[0] == 0:
        raise ValueError("refusing to write an empty/unreadable reference map")
    if not np.all(np.isfinite(pos)):
        raise ValueError("refusing to write a reference map with non-finite coords")

    ids: List[Optional[str]] = []
    for g in gates:
        gid = getattr(g, "gate_id", None)
        ids.append(None if gid is None else str(gid))

    payload = {
        "schema_version": _REF_SCHEMA_VERSION,
        "count": int(pos.shape[0]),
        "gates": [
            {
                "gate_id": ids[i],
                "position_ned": [float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])],
            }
            for i in range(pos.shape[0])
        ],
    }
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


@dataclass
class ReferenceGate:
    """A minimal gate stand-in carrying just what the integrity check reads.

    Has a ``.position`` (NED 3-tuple) and an optional ``.gate_id`` so it is a
    drop-in for the ``check_gate_map`` / ``write_reference_json`` duck type
    WITHOUT importing ``GateSpec`` (keeps this module MAVLink-free and offline).
    """

    position: Tuple[float, float, float]
    gate_id: Optional[str] = None


def read_reference_json(path: Union[str, Path]) -> List[ReferenceGate]:
    """Load a reference map written by :func:`write_reference_json`.

    Validates the schema (required keys present, finite numeric positions, each
    gate a length-3 NED triple) the same way ``calibration.read_calibration_json``
    does, raising ``ValueError`` on a malformed file. Returns a list of
    :class:`ReferenceGate` suitable to pass straight to ``check_gate_map`` as
    ``reference=``.
    """
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("reference JSON must be an object")
    if "gates" not in d:
        raise ValueError("reference JSON missing required key 'gates'")
    raw_gates = d["gates"]
    if not isinstance(raw_gates, list) or not raw_gates:
        raise ValueError("reference JSON 'gates' must be a non-empty list")

    out: List[ReferenceGate] = []
    for i, g in enumerate(raw_gates):
        if not isinstance(g, dict) or "position_ned" not in g:
            raise ValueError(f"reference gate {i} missing 'position_ned'")
        p = g["position_ned"]
        if not isinstance(p, (list, tuple)) or len(p) != 3:
            raise ValueError(f"reference gate {i} position_ned must be length-3")
        try:
            triple = (float(p[0]), float(p[1]), float(p[2]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reference gate {i} has non-numeric position") from exc
        if not all(math.isfinite(c) for c in triple):
            raise ValueError(f"reference gate {i} has non-finite position {triple!r}")
        gid = g.get("gate_id")
        out.append(ReferenceGate(position=triple, gate_id=None if gid is None else str(gid)))

    declared = d.get("count")
    if declared is not None and int(declared) != len(out):
        raise ValueError(
            f"reference JSON count {declared} != number of gates {len(out)}"
        )
    return out
