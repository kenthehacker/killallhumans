# Adversarial review: iter-009i F9 fix (commit b926734)

## Executive summary

The change correctly stops **basin-switching** in racing-line selection when execution `max_velocity` drops (auto-derive), by scoring candidates with a fixed `select_velocity_mps` inside `_select_by_sim`. That matches the diagnosed failure mode (min-snap time scaling couples geometry ranking to `v_max`). The fix is **pragmatic and likely stabilizes `aigp_default`**, but it papers over deeper **oracle honesty** and **test-signal** gaps. Below are issues ordered by severity for a hostile reviewer.

---

## 1. Kinematic oracle still decoupled from `select_velocity_mps`

`_kinematic_eval` hardcodes `max_speed = 15.0` (and `max_accel = 15.0`) regardless of `RacingLineConfig`. Today `select_velocity_mps` defaults to 15, so the mismatch is latent. If anyone lowers `select_velocity_mps` to match a slow course, the BO inner loop would build slower references but the PD+clamp integrator could still chase at up to 15 m/s, **distorting** avg/worst-gate error and the COP-normalized pool. The commit message claims path–velocity decoupling; this is only half-wired: trajectory generation follows `select_velocity_mps`, the toy sim does not follow it automatically.

**Verdict:** Latent foot-gun; fix should thread `select_velocity_mps` (or `TrajectoryOptimizer`’s cap) into `_kinematic_eval` for a coherent oracle.

---

## 2. “Second diagnostic test” adds almost no regression signal

`test_select_velocity_DOES_change_geometry` explicitly avoids asserting that low vs high `select_velocity_mps` produce different offsets. It only checks shape and finiteness. A future refactor that accidentally pins selection velocity or no-ops the BO path could still pass. The primary test (`max_velocity_mps` sweep) is valuable; the companion is **documentation dressed as a test**.

**Verdict:** Either assert inequality on a layout known to split basins (with documented tolerance / xfails), or drop the test and keep prose in the module docstring.

---

## 3. Synthetic vs PyBullet wiring asymmetry

`run_synthetic_benchmark` **hardcodes** `select_velocity_mps=15.0` next to auto-derived `max_velocity`. `run_sim_benchmark` uses `RacingLineConfig` from `race_config.racing_line_overrides` only. Defaults align today, but a JSON override can make PyBullet select lines under a different `select_velocity_mps` than synthetic for the **same** logical track—reintroducing “platform honesty drift” the project has been careful to avoid elsewhere.

**Verdict:** Low probability if configs stay disciplined; worth a one-line invariant comment or shared helper that applies the same `RacingLineConfig` construction policy as synthetic.

---

## 4. `max_velocity_mps` is “informational” but unused in-module

Nothing in `racing_line.py` reads `max_velocity_mps` after this change. Callers must remember to pass execution velocity to `TrajectoryOptimizer` separately (benchmark does). That is fine but **fragile API semantics**: a caller constructing `RacingLineOptimizer(RacingLineConfig(max_velocity_mps=6))` might assume the racing line “knows” execution speed; it does not. Docstrings mitigate; static typing cannot catch this.

**Verdict:** Acceptable with docs; a `@property` or `assert` in debug builds could catch misuse.

---

## 5. Test geometry / helper semantics

`_offsets` returns raw `(dx,dy,dz)` in world frame while the docstring claims “lateral … in the gate’s lateral direction.” The toy gates share `normal=(1,0,0)` and zero yaw in the snippet, so the bug is masked. On pitched/yawed gates the assertion would not measure what the prose claims.

**Verdict:** Minor clarity debt; could confuse a future maintainer extending the test.

---

## 6. Frame / convention smell (NED vs positive-Z test layout)

`test_racing_line_velocity_invariance` uses `z = +2.0` for gates and start. Much of the stack is documented as NED with −z up. If the optimizer is agnostic this may still be consistent internally, but the test does not mirror `race_01` conventions; it optimizes a **different embedding** of the problem. Risk: passing tests while missing an interaction between vertical offset logic and sign conventions on real tracks.

**Verdict:** Low but non-zero; consider mirroring a slice of real gate data.

---

## 7. Magic number duplication

`15.0` is locked in `benchmark.py` synthetic path, defaulted in `RacingLineConfig`, and baked into `_kinematic_eval`. Drift if any one site changes.

**Verdict:** Single named constant (e.g. `RacingLineConfig.DEFAULT_SELECT_VELOCITY_MPS`) imported by benchmark.

---

## 8. Conceptual honesty: “geometry independent of velocity”

True **path–velocity decomposition** removes time / dynamics from the spatial argmin. This patch **freezes the sim oracle’s trajectory speed** while still ranking with **kinematic sim error** and **race_time** from that reference trajectory—so selection remains **dynamics-informed**, just not tied to the execution `v_max`. That is a defensible engineering trade; it is **not** the same as a pure geometric corridor solve. A reviewer could call the Heilmeier/Kapania citations **aspirational** rather than strict implementations.

**Verdict:** Wording in commit/docs could be toned to “reference-speed decoupling” vs full literature decomposition.

---

## 9. Cache key choice

Excluding `max_velocity_mps` from the cache key is intentional (good for determinism when only execution speed changes). If future code starts using `max_velocity_mps` inside selection again without updating the key, stale cache risk returns.

**Verdict:** Fine; add a short comment near `_compute_cache_key` listing fields that must trigger invalidation if behavior changes.

---

## Positive notes (balance)

- The **root-cause story** in `git show` matches the code: `TrajectoryOptimizer(... max_velocity=...)` in `_select_by_sim` was the coupling lever; switching to `select_velocity_mps` is minimal blast radius.
- **Primary regression test** (four `max_velocity_mps` values, identical offsets) directly encodes F9.
- Synthetic benchmark comment block clearly documents intent for downstream readers.

---

## Bottom line

Shippable as a **stability patch** with known technical debt: kinematic-eval velocity clamp, weak second test, duplicated 15 m/s, and slightly overstated “full” path–velocity decomposition. Addressing (1) and (2) would materially strengthen the next iteration.
