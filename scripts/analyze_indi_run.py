"""Read-out tool for the roadmap-#2 INDI experiment (the 0.53x roll crux).

Given a run captured by ``scripts/aigp_vq1_run.py --indi``, this prints an
unambiguous VERDICT on whether the documented 0.53x roll-attenuation deficit is
a RECOVERABLE model/effectiveness mismatch or a TRUE rate/bandwidth wall.

It is the OFFLINE companion to ``control/indi_inner_loop.py`` (which proves the
discriminator at the control-object level) and a sibling of
``scripts/iter36_compare.py`` (same gzip/plain JSONL open, same record-iteration
+ printed-report shape). It is numpy-/stdlib-only and touches NO control path.

The read-out logic (from docs/aigp/2026-06-16-deep-research-improvement-report.md
and the IndiInnerLoop docstring):

    recovered  => model mismatch (achieved roll driven back toward commanded);
    still clamped => true bandwidth limit (a correct inversion can't beat it).

Usage:
    python -m scripts.analyze_indi_run captures/indi.jsonl.gz [label]
    python scripts/analyze_indi_run.py captures/indi.jsonl.gz [label]

Exit codes: 0 verdict printed (any of RECOVERED / BANDWIDTH-LIMITED /
INCONCLUSIVE); 2 the capture has NO INDI telemetry (flown without --indi) — the
message tells you to re-run with --indi; 1 the file is unreadable / empty.

What lands in the JSONL per tick (recorded by aigp_vq1_run.py's iter-59
callback, written by ``_write_telem_log`` — gzip if .gz else plain):
  * ``roll``     achieved roll ANGLE (rad), from telem.orientation.to_euler()[0];
  * ``cmd_roll`` COMMANDED roll ANGLE (rad), the AttitudeCommand the tracker set;
  * ``gyro``     achieved body rates (rad/s); gyro[0] is the achieved roll RATE;
  * ``indi``     the IndiDebug snapshot (only when --indi): per-axis (roll,pitch,
                 yaw) lists ``ghat`` / ``u`` / ``saturated`` / ``alpha_des`` /
                 ``alpha_meas`` / ``q_err_vec`` / ``g_updated`` + scalar ``dt``.
The 0.53 lives in achieved-angle vs commanded-angle, so the discriminator ratio
is ``achieved_roll / commanded_roll`` over the fast-turn (high-|cmd_roll|)
segments. If ``cmd_roll`` is absent we reconstruct the commanded roll from the
INDI ``q_err_vec`` applied to the achieved roll (see ``_commanded_roll``).
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from typing import Dict, List, Optional, Tuple

ROLL = 0  # per-axis index convention in every INDI debug list (roll,pitch,yaw)

# --- Verdict thresholds (see module docstring for the rationale) -------------
# g_roll is "converged" when its tail is stable to within this relative spread.
G_CONVERGE_REL = 0.05          # (max-min)/|mean| over the settle tail
G_SETTLE_TAIL_FRAC = 0.25      # fraction of the (post-warmup) run used as "tail"
G_WARMUP_FRAC = 0.10           # ignore the first 10% of ticks for convergence

# A tick counts as "high-demand / fast-turn" on roll when |cmd_roll| is both in
# the top quartile AND above this absolute floor (a smooth straight under-excites
# roll — the report's flagged risk; small commands give a noisy, meaningless
# ratio). 0.15 rad ~= 8.6 deg of commanded bank.
HIGH_DEMAND_ROLL_RAD = 0.15
HIGH_DEMAND_QUANTILE = 0.75
MIN_HIGH_DEMAND_TICKS = 15      # too few fast turns => INCONCLUSIVE (under-excited)
RATIO_DENOM_FLOOR = 0.05        # ignore per-tick ratios with |cmd_roll| below this

# achieved/commanded roll ratio bands. The deficit anchor is 0.53; 0.65 sits
# safely above it (a clamped run reads ~0.53 <= 0.65), 0.85 marks "driven toward
# 1.0". The 0.65..0.85 mid-band is honestly ambiguous -> INCONCLUSIVE.
RECOVERED_RATIO = 0.85
DEFICIT_RATIO = 0.65

# Roll-rate saturation: if the INDI roll command pins its clamp this often during
# the fast turns, that is itself bandwidth-limit evidence.
SAT_HIGH_FRAC = 0.30


# ---------------------------------------------------------------------------
# IO — match iter36_compare.py: open .jsonl.gz (gzip) or plain .jsonl, one
# JSON object per line.
# ---------------------------------------------------------------------------
def load_rows(path: str) -> List[Dict]:
    """Read a capture into a list of per-tick dicts (gzip if .gz, else plain)."""
    opener = gzip.open if path.endswith(".gz") else open
    rows: List[Dict] = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------
def _quat_roll(q) -> Optional[float]:
    """Roll (rad) from a quaternion stored as [w,x,y,z] or {w,x,y,z}.

    Identical math to competition.adapter.Quaternion.to_euler (roll component)
    so the achieved-roll signal here matches what the recorder logged as ``roll``;
    used only as a FALLBACK when a row lacks the precomputed ``roll`` field.
    """
    if q is None:
        return None
    if isinstance(q, dict):
        w, x, y, z = q.get("w"), q.get("x"), q.get("y"), q.get("z")
    else:
        try:
            w, x, y, z = q[0], q[1], q[2], q[3]
        except (TypeError, IndexError, KeyError):
            return None
    if None in (w, x, y, z):
        return None
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    return math.atan2(sinr_cosp, cosr_cosp)


def _achieved_roll(row: Dict) -> Optional[float]:
    """Achieved roll ANGLE (rad): prefer the recorded ``roll``; else from a
    quaternion field if one was logged."""
    r = row.get("roll")
    if isinstance(r, (int, float)):
        return float(r)
    for key in ("orientation", "q", "quat"):
        if key in row:
            rr = _quat_roll(row[key])
            if rr is not None:
                return rr
    return None


def _commanded_roll(row: Dict, achieved_roll: Optional[float]) -> Optional[float]:
    """Commanded roll ANGLE (rad), the discriminator's denominator.

    Best signal: the recorded ``cmd_roll`` (the AttitudeCommand.roll_rad the
    tracker set). FALLBACK (when no command was logged): reconstruct the desired
    roll from the INDI body-frame quaternion-error roll component. For small
    angles the body-x error ~= 0.5*(roll_des - roll_cur), so
    roll_des ~= roll_cur + 2*q_err_vec[0]; that matches the IndiInnerLoop's own
    ``2 * q_err_vec`` convention, so it reproduces the controller's notion of the
    commanded roll without any runner change.
    """
    c = row.get("cmd_roll")
    if isinstance(c, (int, float)):
        return float(c)
    indi = row.get("indi")
    if isinstance(indi, dict) and achieved_roll is not None:
        qev = indi.get("q_err_vec")
        if isinstance(qev, (list, tuple)) and len(qev) >= 1 and qev[ROLL] is not None:
            return achieved_roll + 2.0 * float(qev[ROLL])
    return None


def _roll_rate(row: Dict) -> Optional[float]:
    g = row.get("gyro")
    if isinstance(g, (list, tuple)) and len(g) >= 1 and isinstance(g[ROLL], (int, float)):
        return float(g[ROLL])
    return None


def _indi_axis(indi: Dict, field: str, axis: int = ROLL):
    v = indi.get(field)
    if isinstance(v, (list, tuple)) and len(v) > axis:
        return v[axis]
    return None


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def g_convergence(indi_rows: List[Tuple[int, Dict]]) -> Dict:
    """Trajectory + convergence of ghat per axis (roll is the one that matters).

    ``indi_rows`` is the list of (row_index, indi_dict) for ticks that carried
    an INDI snapshot. Returns final per-axis g, whether roll-g converged, the
    relative tail spread, and the tick at which roll-g settled.
    """
    g_series = {ax: [] for ax in ("roll", "pitch", "yaw")}
    idxs: List[int] = []
    for ridx, indi in indi_rows:
        gh = indi.get("ghat")
        if not (isinstance(gh, (list, tuple)) and len(gh) >= 3):
            continue
        try:
            vals = [float(gh[0]), float(gh[1]), float(gh[2])]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in vals):
            continue
        idxs.append(ridx)
        for ax, v in zip(("roll", "pitch", "yaw"), vals):
            g_series[ax].append(v)

    n = len(g_series["roll"])
    out: Dict = {
        "n": n,
        "g_series": g_series,
        "idxs": idxs,
        "final": {ax: (g_series[ax][-1] if g_series[ax] else float("nan"))
                  for ax in g_series},
    }
    if n < 4:
        out.update(converged=False, rel_spread=float("nan"), settle_idx=None,
                   reason="too few INDI ticks to judge g convergence")
        return out

    roll = g_series["roll"]
    warm = int(G_WARMUP_FRAC * n)
    tail_start = max(warm, int((1.0 - G_SETTLE_TAIL_FRAC) * n))
    tail = roll[tail_start:]
    mean = sum(tail) / len(tail)
    spread = (max(tail) - min(tail))
    rel = spread / abs(mean) if abs(mean) > 1e-9 else float("inf")
    converged = rel < G_CONVERGE_REL

    # When did roll-g settle: earliest tick after warmup beyond which every
    # later sample stays within G_CONVERGE_REL of the final value.
    final = roll[-1]
    band = G_CONVERGE_REL * abs(final) if abs(final) > 1e-9 else 1e-6
    settle_i = None
    for i in range(warm, n):
        if all(abs(roll[j] - final) <= band for j in range(i, n)):
            settle_i = i
            break
    out.update(
        converged=converged,
        rel_spread=rel,
        settle_idx=(idxs[settle_i] if settle_i is not None else None),
        settle_series_i=settle_i,
        reason="",
    )
    return out


def roll_discriminator(rows: List[Dict]) -> Dict:
    """achieved-vs-commanded roll over the fast-turn segments (THE read-out).

    Returns the high-demand sample count, the median per-tick achieved/commanded
    ratio AND the magnitude ratio (sum|achieved|/sum|commanded|), the achieved
    roll-rate stats, the roll-saturation fraction over those segments, and
    whether the commanded roll came from ``cmd_roll`` or the q_err reconstruction.
    """
    # Per tick gather commanded/achieved roll (+ rate, saturation).
    samples: List[Dict] = []
    used_reconstruction = False
    have_cmd = False
    for row in rows:
        ach = _achieved_roll(row)
        cmd = _commanded_roll(row, ach)
        if ach is None or cmd is None:
            continue
        if "cmd_roll" in row:
            have_cmd = True
        elif row.get("indi"):
            used_reconstruction = True
        indi = row.get("indi") or {}
        sat = _indi_axis(indi, "saturated") if isinstance(indi, dict) else None
        samples.append({
            "cmd": cmd,
            "ach": ach,
            "rate": _roll_rate(row),
            "sat": bool(sat) if sat is not None else None,
        })

    out: Dict = {
        "n_total": len(samples),
        "used_reconstruction": used_reconstruction and not have_cmd,
        "cmd_source": "cmd_roll" if have_cmd else
                      ("q_err_vec reconstruction" if used_reconstruction else "none"),
    }
    if not samples:
        out.update(n_high=0, ratio_median=float("nan"), ratio_mag=float("nan"),
                   sat_frac=float("nan"), peak_rate=float("nan"),
                   thresh=float("nan"))
        return out

    # High-demand selection: |cmd| in the top quartile AND above the floor.
    abscmd = [abs(s["cmd"]) for s in samples]
    q = _quantile(abscmd, HIGH_DEMAND_QUANTILE)
    thresh = max(q, HIGH_DEMAND_ROLL_RAD)
    high = [s for s in samples if abs(s["cmd"]) >= thresh]

    # Per-tick signed ratio (achieved/commanded), guarding tiny denominators.
    ratios = [s["ach"] / s["cmd"] for s in high if abs(s["cmd"]) >= RATIO_DENOM_FLOOR]
    ratio_median = (sorted(ratios)[len(ratios) // 2] if ratios else float("nan"))
    # Magnitude ratio is robust to per-tick sign/phase noise: total achieved
    # bank vs total commanded bank over the fast turns.
    sum_ach = sum(abs(s["ach"]) for s in high)
    sum_cmd = sum(abs(s["cmd"]) for s in high)
    ratio_mag = (sum_ach / sum_cmd) if sum_cmd > 1e-9 else float("nan")

    sat_vals = [s["sat"] for s in high if s["sat"] is not None]
    sat_frac = (sum(1 for s in sat_vals if s) / len(sat_vals)) if sat_vals else float("nan")
    rates = [abs(s["rate"]) for s in high if s["rate"] is not None]
    peak_rate = max(rates) if rates else float("nan")

    out.update(
        n_high=len(high),
        thresh=thresh,
        ratio_median=ratio_median,
        ratio_mag=ratio_mag,
        sat_frac=sat_frac,
        peak_rate=peak_rate,
        max_abs_cmd=max(abscmd),
    )
    return out


def classify(gconv: Dict, disc: Dict) -> Tuple[str, str]:
    """Return (verdict, explanation). One of RECOVERED / BANDWIDTH-LIMITED /
    INCONCLUSIVE, with the numbers + read-out logic behind it."""
    converged = gconv.get("converged", False)
    n_high = disc.get("n_high", 0)
    ratio = disc.get("ratio_mag", float("nan"))
    ratio_med = disc.get("ratio_median", float("nan"))
    sat_frac = disc.get("sat_frac", float("nan"))

    # Precondition 1: enough fast-turn excitation to even form the ratio.
    if n_high < MIN_HIGH_DEMAND_TICKS:
        return ("INCONCLUSIVE",
                f"only {n_high} high-demand roll tick(s) (need >= "
                f"{MIN_HIGH_DEMAND_TICKS}); the roll axis was UNDER-EXCITED. "
                f"Re-run with more roll excitation / a higher cruise / a longer "
                f"run so there are real fast turns to measure.")

    # Precondition 2: a converged g is required to TRUST either verdict.
    if not converged:
        return ("INCONCLUSIVE",
                f"g_roll did NOT converge (tail rel-spread "
                f"{gconv.get('rel_spread', float('nan')):.3f} >= {G_CONVERGE_REL}); "
                f"a non-converged effectiveness estimate means the inversion is "
                f"not yet trustworthy. Run longer / with more roll excitation so "
                f"online-G settles before reading the ratio.")

    if not math.isfinite(ratio):
        return ("INCONCLUSIVE",
                "could not form the achieved/commanded roll ratio (missing "
                "achieved or commanded roll signal).")

    # g converged AND we have a ratio -> the verdict is real.
    if ratio >= RECOVERED_RATIO:
        return ("RECOVERED",
                f"g_roll converged (final {gconv['final']['roll']:.3f}) and the "
                f"achieved/commanded roll ratio is {ratio:.2f} (>= {RECOVERED_RATIO}) "
                f"over {n_high} fast-turn ticks, with roll saturated only "
                f"{_pct(sat_frac)} of them. The achieved roll was driven back "
                f"toward the commanded value: the 0.53 deficit was a RECOVERABLE "
                f"model/effectiveness mismatch -- INDI is a real speed lever.")

    if ratio <= DEFICIT_RATIO or (math.isfinite(sat_frac) and sat_frac >= SAT_HIGH_FRAC):
        why_sat = (f" and roll pinned the rate clamp {_pct(sat_frac)} of the "
                   f"fast-turn ticks (>= {_pct(SAT_HIGH_FRAC)})"
                   if (math.isfinite(sat_frac) and sat_frac >= SAT_HIGH_FRAC) else "")
        return ("BANDWIDTH-LIMITED",
                f"g_roll converged (final {gconv['final']['roll']:.3f}) yet the "
                f"achieved/commanded roll ratio stayed {ratio:.2f} "
                f"(<= {DEFICIT_RATIO}, near the documented 0.53){why_sat}, over "
                f"{n_high} fast-turn ticks. A correct measured-accel inversion did "
                f"NOT restore the roll: this is a TRUE rate/bandwidth wall -- pursue "
                f"the trajectory-side mitigation (roadmap #3 bandwidth-constrained "
                f"re-timing), not more inner-loop gain.")

    # Converged, excited, but the ratio sits in the ambiguous mid-band.
    return ("INCONCLUSIVE",
            f"g_roll converged (final {gconv['final']['roll']:.3f}) but the "
            f"achieved/commanded roll ratio {ratio:.2f} is in the ambiguous band "
            f"({DEFICIT_RATIO}..{RECOVERED_RATIO}) -- neither clearly restored (~1.0) "
            f"nor clearly clamped (~0.53), and roll saturated only {_pct(sat_frac)} "
            f"of the fast-turn ticks. Push harder turns / a longer run to drive the "
            f"ratio out of the mid-band before declaring mismatch vs bandwidth.")


def _pct(x: float) -> str:
    return "n/a" if not math.isfinite(x) else f"{100.0 * x:.0f}%"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def analyze(path: str, label: str = "") -> int:
    """Analyze one capture; print the report + VERDICT. Returns an exit code."""
    try:
        rows = load_rows(path)
    except FileNotFoundError:
        print(f"ERROR: capture not found: {path}", file=sys.stderr)
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read {path}: {exc}", file=sys.stderr)
        return 1

    header = label or path
    print(f"\n=== INDI roll-crux read-out: {header}  ({len(rows)} ticks) ===")

    if not rows:
        print("  (empty capture -- no ticks)")
        return 1

    # Collect the INDI snapshots. If there are NONE, the run was flown without
    # --indi: there is nothing to read out. Say so clearly and exit non-zero.
    indi_rows = [(i, r["indi"]) for i, r in enumerate(rows)
                 if isinstance(r.get("indi"), dict)]
    if not indi_rows:
        print("  NO INDI telemetry in this capture -- the run was flown WITHOUT "
              "--indi.")
        print("  There is no online-G / measured-accel read-out to judge the "
              "0.53 crux.")
        print("  => Re-run with --indi:  python -m scripts.aigp_vq1_run "
              "--indi --record <capture>.jsonl.gz")
        return 2

    print(f"  INDI ticks: {len(indi_rows)}/{len(rows)} carried an INDI snapshot.")

    # (a) g convergence
    gconv = g_convergence(indi_rows)
    fin = gconv["final"]
    print("\n  (a) ONLINE-G convergence (precondition for trusting the verdict):")
    print(f"      ghat final: roll={fin['roll']:.3f}  pitch={fin['pitch']:.3f}  "
          f"yaw={fin['yaw']:.3f}  (seed roll=1.0, pitch/yaw=2.1)")
    if gconv["converged"]:
        si = gconv.get("settle_idx")
        where = f"around tick {si}" if si is not None else "early"
        print(f"      g_roll CONVERGED: tail rel-spread {gconv['rel_spread']:.3f} "
              f"< {G_CONVERGE_REL}, settled {where}.")
    else:
        rs = gconv.get("rel_spread", float("nan"))
        extra = gconv.get("reason") or (
            f"tail rel-spread {rs:.3f} >= {G_CONVERGE_REL}")
        print(f"      g_roll did NOT converge ({extra}).")

    # (b) achieved-vs-commanded roll discriminator
    disc = roll_discriminator(rows)
    print("\n  (b) ACHIEVED-vs-COMMANDED roll (THE discriminator):")
    print(f"      commanded-roll source: {disc['cmd_source']}"
          + ("  [reconstructed from q_err_vec -- no cmd_roll logged]"
             if disc.get("used_reconstruction") else ""))
    if disc["n_total"] == 0:
        print("      no usable achieved/commanded roll samples in the capture.")
    else:
        print(f"      fast-turn ticks: {disc['n_high']}/{disc['n_total']} "
              f"(|cmd_roll| >= {disc.get('thresh', float('nan')):.3f} rad; "
              f"max |cmd_roll| {disc.get('max_abs_cmd', float('nan')):.3f} rad)")
        print(f"      achieved/commanded roll  magnitude-ratio={_fmt(disc['ratio_mag'])}"
              f"  median-ratio={_fmt(disc['ratio_median'])}  (1.0=restored, "
              f"~0.53=the documented deficit)")
        print(f"      achieved roll-rate peak over fast turns: "
              f"{_fmt(disc['peak_rate'])} rad/s")

    # (c) roll saturation
    print("\n  (c) ROLL-RATE SATURATION over the fast turns:")
    print(f"      roll command pinned the clamp {_pct(disc.get('sat_frac', float('nan')))} "
          f"of fast-turn ticks  (high saturation during the deficit is itself "
          f"bandwidth evidence).")

    # VERDICT
    verdict, explanation = classify(gconv, disc)
    print("\n  " + "-" * 68)
    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")
    print("  read-out logic: g must converge to trust the inversion; then "
          "achieved/commanded")
    print("  roll ~1.0 => recoverable mismatch, ~0.53 (and/or clamped) => true "
          "bandwidth wall.")
    print("  " + "-" * 68)
    return 0


def _fmt(x: float) -> str:
    return "n/a" if (x is None or not math.isfinite(x)) else f"{x:.3f}"


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m scripts.analyze_indi_run "
              "<capture.jsonl[.gz]> [label]", file=sys.stderr)
        return 1
    path = argv[0]
    label = argv[1] if len(argv) > 1 else ""
    return analyze(path, label)


if __name__ == "__main__":
    raise SystemExit(main())
