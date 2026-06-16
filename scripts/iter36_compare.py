"""iter-36 kv-sweep comparator.

Computes the abort/comparison metrics the iter-36 workflow specified, for one
capture: per-gate plane-crossing lateral(Y)/vertical(Z) error + sign, closest
3D approach, gyro |w| p95/max, roll-rate gyro_p p95/max, per-gate-flip post-flip
0.35s window gyro_p max + cmd_roll-at-clamp fraction, and lateral-accel clamp
engagement (whole-run + startup).

Usage: python scripts/iter36_compare.py captures/foo.jsonl.gz [label]
"""
import gzip, json, math, sys

# Baked gate CENTRES (NED, with the -0.85 vertical opening offset applied, as
# the runner bakes into the gate map + sequencer). These are the opening centres.
GATES = {
    0: (-23.3, -0.4, -0.9), 1: (-46.9, -2.5, 4.2), 2: (-74.6, 1.2, 12.8),
    3: (-111.5, -5.1, 23.7), 4: (-135.5, -0.8, 24.5), 5: (-159.2, -4.4, 25.1),
}
CLAMP = 7.0          # g*tan(0.62)
ROLL_CLAMP = 0.619   # |cmd_roll| at/above this == tilt-saturated
HALF_OPENING = 0.75  # interior gate opening is 1.5m => half = 0.75m (frame edge).
                     # Frame clearance = 0.75 - max(|lat(Y)|,|vert(Z)|); the drone
                     # clips the bar when this hits ~0 (minus its own radius). This
                     # is the "how close to crashing" margin that bounds cruise.


def _breach_where(laty, vertz):
    """Which frame edge is the binding/breaching one (user: top/bottom/side?).
    The binding axis is whichever |error| is larger (it hits the 0.75 m edge
    first). NED Z is DOWN, so vert<0 => drone is ABOVE the centre => TOP edge;
    vert>0 => BOTTOM. Lateral +Y/-Y are the two side bars."""
    if abs(vertz) >= abs(laty):
        return "TOP (too high)" if vertz < 0 else "BOTTOM (too low)"
    return "+Y side" if laty > 0 else "-Y side"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def analyze(path, label=""):
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    n = len(rows)
    t0 = rows[0]["t_wall"]
    dur = rows[-1]["t_wall"] - t0
    print(f"\n=== {label or path}  ({n} rows, {dur:.1f}s) ===")

    # --- per-gate plane crossing (drone X passes gate X) + closest 3D approach
    worst_lat = 0.0
    min_clear = (99.0, None)  # (clearance, gate) -- the closest-to-crash gate
    print("  per-gate: plane-cross lat(Y)/vert(Z), frame-CLEARANCE, closest-3D")
    for gi, g in GATES.items():
        # plane crossing: first tick where (pos_x - gate_x) changes sign while
        # the drone is in this gate's neighbourhood (within 6m along X).
        cross = None
        for k in range(1, n):
            a, b = rows[k - 1]["pos"][0] - g[0], rows[k]["pos"][0] - g[0]
            if a == 0 or (a < 0) != (b < 0):  # sign change => crossed the plane
                # interpolate to the crossing
                f = abs(a) / (abs(a) + abs(b)) if (abs(a) + abs(b)) > 1e-9 else 0.0
                py = rows[k - 1]["pos"][1] + f * (rows[k]["pos"][1] - rows[k - 1]["pos"][1])
                pz = rows[k - 1]["pos"][2] + f * (rows[k]["pos"][2] - rows[k - 1]["pos"][2])
                if abs(rows[k]["pos"][0] - g[0]) < 8.0:
                    cross = (py - g[1], pz - g[2])
                    break
        closest = min(math.dist(r["pos"], g) for r in rows)
        if cross:
            laty, vertz = cross
            worst_lat = max(worst_lat, abs(laty))
            clear = HALF_OPENING - max(abs(laty), abs(vertz))  # frame clearance (binding axis)
            if clear < min_clear[0]:
                min_clear = (clear, gi, _breach_where(laty, vertz))
            # WHERE on the frame is the binding edge (user: top/bottom/side?).
            # NED Z is DOWN, so vert<0 => drone ABOVE centre => TOP edge.
            where = _breach_where(laty, vertz)
            flag = (f"  <-- {'BREACH' if clear < 0 else 'CLOSE TO'} FRAME ({where})"
                    if clear < 0.20 else "")
            print(f"    gate{gi}: lat(Y)={laty:+.3f} vert(Z)={vertz:+.3f}  "
                  f"clearance={clear:+.3f}m [{where}]  closest3D={closest:.3f}{flag}")
        else:
            print(f"    gate{gi}: (no plane crossing found)  closest3D={closest:.3f}")
    print(f"  WORST plane-cross |lat(Y)| = {worst_lat:.3f}")
    mc = min_clear if len(min_clear) == 3 else (min_clear[0], min_clear[1], "?")
    print(f"  >>> CLOSEST-TO-CRASH: {mc[0]:+.3f}m frame clearance at gate{mc[1]} "
          f"on the {mc[2]} edge (0=clip; race-limiting margin) <<<")

    # --- gyro stats (|w| magnitude, and roll-rate gyro[0])
    wmag = [math.sqrt(sum(c * c for c in (r.get("gyro") or [0, 0, 0]))) for r in rows]
    gp = [abs((r.get("gyro") or [0, 0, 0])[0]) for r in rows]
    print(f"  gyro |w|: p95={pct(wmag,.95):.2f} max={max(wmag):.2f}   "
          f"roll-rate gyro_p: p95={pct(gp,.95):.2f} max={max(gp):.2f}")

    # --- lateral-accel clamp engagement (whole + startup t<2s)
    def clamp_frac(sub):
        d = [r for r in sub if r.get("dbg")]
        if not d:
            return 0.0
        c = sum(1 for r in d if math.hypot(r["dbg"]["accel"][0], r["dbg"]["accel"][1]) >= CLAMP - 0.05)
        return 100 * c / len(d)
    startup = [r for r in rows if r["t_wall"] - t0 < 2.0]
    print(f"  lateral-clamp engaged: whole={clamp_frac(rows):.1f}%  startup(t<2s)={clamp_frac(startup):.1f}%")

    # --- per gate-flip: post-flip 0.35s window gyro_p max + cmd_roll@clamp frac
    flips = [k for k in range(1, n) if rows[k].get("target_gate") != rows[k - 1].get("target_gate")
             and rows[k].get("target_gate") is not None]
    print("  gate-flips (post-flip 0.35s window): gyro_p max, cmd_roll@clamp%")
    for k in flips:
        tg = rows[k]["target_gate"]
        tf = rows[k]["t_wall"]
        win = [r for r in rows[k:] if r["t_wall"] - tf <= 0.35]
        if not win:
            continue
        gpmax = max(abs((r.get("gyro") or [0, 0, 0])[0]) for r in win)
        rc = [r for r in win if "cmd_roll" in r]
        clampf = 100 * sum(1 for r in rc if abs(r["cmd_roll"]) >= ROLL_CLAMP) / len(rc) if rc else 0.0
        print(f"    -> target gate{tg}: gyro_p max={gpmax:.2f}, cmd_roll@clamp={clampf:.0f}%")


if __name__ == "__main__":
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
