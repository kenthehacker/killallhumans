# AIGP — Deep-Research Improvement Report (2026-06-16)

Companion to `2026-06-16-speed-and-spline-handoff.md`. This report answers three
questions — (1) how to improve the code, (2) where ML pays off, (3) how to derive
parameters online from observables — centered on the **0.53× roll-attenuation
ceiling** documented in the handoff.

**Provenance.** Automated deep-research run (run id `wf_f5d2e12d-269`): 6 search
angles → **28 primary peer-reviewed sources** → 139 extracted claims → **25
adversarially 3-vote-verified → 20 confirmed, 5 refuted** → 12 synthesized
findings. Every surviving finding rests on IEEE T-CST / RA-L, AIAA JGCD, Elsevier
*Aerospace Science & Tech*, *Nature*, *Science Robotics*, or ICRA/IROS/RSS sources.
No blog/forum/marketing source survived verification.

---

## TL;DR — two verdicts

1. **The 0.53× crux is UNRESOLVED in the literature — make it an experiment, not a
   belief.** Every claim that would have *settled* whether `achieved_roll ≈ 0.53×
   commanded` is a true actuator/bandwidth wall vs. a recoverable
   control-effectiveness/model-mismatch error was **refuted 0-3** by the verifiers —
   including the tempting one that the 9 Hz limit cycle "is the signature of an
   effectiveness mismatch." What survives: INDI *tolerates* large effectiveness
   errors (Tal & Karaman), and bandwidth-aware INDI can *partially* compensate
   finite actuator bandwidth (ANDIa). **Resolution = a trial, not a citation:**
   > Run measured-angular-accel adaptive-INDI with online-G on DCGame. **If
   > commanded roll is restored → it was model mismatch (recoverable).** **If
   > achieved roll stays clamped despite a correct measured-accel inversion → it's
   > a true rate/bandwidth limit** no controller beats. Bandwidth and effectiveness
   > errors are *coupled*; no published telemetry-only test cleanly separates them.

2. **For ML: lightweight augmentation beats a full RL policy for this setting.**
   Decisive result (Song et al., *Science Robotics* 2023): RL's racing edge comes
   from optimizing a *better objective* (task-level gate progress), **not** from the
   network being a better approximator — forced onto the *same* objective as MPC, RL
   is **worse** (tracking loss 12.62±12.69 vs MPC 3.32±3.14). Swift (*Nature* 2023)
   shows the ceiling is high (champion-level, >100 km/h) but its reality-gap closure
   needs *target-sim* residual data that a black-box DCGame makes costly to gather.

---

## Roadmap (effort × payoff ranked)

| # | Action | Effort | Payoff | Source |
|---|--------|--------|--------|--------|
| **1** | Wire `competition/calibration.py` → an **online RLS/LMS** estimator **+ add drag-aware differential-flatness feedforward** to the SE(3) tracker | **Low** | **High** | Faessler RA-L 2018 (~50% RMS error cut); reuses existing FF path + drag-cal stub |
| **2** | **Measured-accel adaptive-INDI / ANDIa** with online-G from gyro+commands | **Med** | **High** | Smeur JGCD 2016 + ANDIa AST 2025. Bench-ID-free speed lever **and** the crux discriminator |
| **3** | Re-time the arc-length spline with **TOPPQuad** vs an identified per-motor thrust bound **+ a hand-added commanded-roll-rate/bandwidth constraint** | **Med** | **Med** | TOPPQuad IROS 2024 (models no drag/motor dynamics → needs the explicit bandwidth constraint) |
| **4** | Full RL policy | **High** | High-but-risky | **Not first.** Most edge is the objective (cheaper to capture); reality-gap data costly for a black-box sim |

---

## 1 — Improve the code

### (a) Past the ceiling — two complementary levers

**Controller side — measured-accel INDI (roadmap #2).** INDI gets its robustness
from *filtered measured-angular-accel feedback*, not a forward model: the moment
increment is `μ_c = μ_f + J(Ω̇_c − Ω̇_f)`, so inertia **J is the only model term**
and the **control-effectiveness matrix G is the single residual** an online scheme
must supply. Tal & Karaman validated at 12.9 m/s / 2.1 g / 6.6 cm RMSE with drag-plate
and rope disturbances, transferred across airframes with static-test G and *no
retuning*, and showed INDI reaches commanded acceleration "even for very large"
effectiveness discrepancies — the affirmative basis for a bench-ID-free *imperfect-G*
INDI. *Caveats:* rejection is reactive/lagged (noisy gyro differentiation + filter
delay), and mapping moment→motor commands still needs G1/G2. The robustness proof is
for a *scalar* per-axis error, not a full-matrix mis-ID with a closed-form stability
bound — so coarse G is workable, arbitrary online-G stability is **not** proven.

**Trajectory side — drag-aware differential-flatness feedforward (roadmap #1).** A
quadrotor with linear rotor drag is differentially flat in position+heading, so exact
feedforward thrust / orientation / body-rates / angular-accelerations come
*algebraically* from the reference trajectory. Adding this to a cascaded controller
cuts RMS tracking error **~50% (51–63%)** vs treating drag as an unknown disturbance,
independent of trajectory (Faessler RA-L 2018). Feeds directly into the existing SE(3)
tracker + dormant feedforward path, and pairs with the online drag estimate from
roadmap #1. *Caveats:* needs **linear** rotor-drag coefficients (offline or online-ID),
assumes a linear drag model (no wind, stiff props), validated only to **5 m/s** on one
platform — below the ~50 km/h peak, so expect partial transfer and re-validate on DCGame.

**TOPPQuad re-timing (roadmap #3).** Takes a *fixed* collision-free path (your
arc-length spline) and minimizes traversal time subject to per-**motor** thrust limits
+ full rigid-body dynamics. Offline (~30 s/solve via CasADi/IPOPT), fine for one fixed
track; non-convex (success 0.61–0.99), init-sensitive. **It models NO drag and NO
motor/rotor dynamics**, so it bounds steady-state thrust *magnitude* only and does **not**
capture the inner-loop bandwidth behind 0.53× — it must be augmented with an explicit
commanded-roll-rate / bandwidth constraint to respect the ceiling.

**TOGT (path-shortening only).** Gate-traversing time-optimal planning exploits gate
free-space rather than center waypoints (16.1 m shorter, 0.35 s faster than CPC on a
19-gate task). But the big win is on *dense* (19–31-gate) tracks; on a 6-gate VQ1 the
realistic benefit is **path-shortening through gate apertures**. The stronger claim that
TOGT enforces full 13-state + body-rate limits was **refuted (1-2)** — do **not** rely
on it as the bandwidth-aware planner.

### (b) Race-day robustness — UNDER-EVIDENCED in this run

This angle was searched and fetched (drift-corrected VIO [arXiv:2512.20475], MonoRace/
AlphaPilot-class perception, [RSS'16 Deep Drone Racing]) but **none of its claims
survived into the top-25 verified set** — verification budget concentrated on
control/trajectory/ML. **This was covered by a dedicated follow-up run — see Part 2
below**, which directly targets the handoff's #1 open item (sim degrades after ~25 runs;
gate-map transfer returns garbage / sign-flipped X or Z≈−350; uniform-shift drift not yet
caught).

---

## 2 — ML applicability, by ROI

**Verdict: lightweight ML augmentation** (extend the dormant 12→64→3 residual + a
learned/adaptive INDI) is the higher-ROI path; full RL is not first.

- **Why not full RL first:** Song et al. (*Science Robotics* 2023) — RL's advantage is
  "not that it optimizes its objective better but that it optimizes a *better
  objective*"; on the *same* objective RL underperforms nonlinear MPC. So a good
  objective + lightweight augmentation captures most of the benefit. (Nuance: the paper
  also credits domain randomization for robustness, so the RL gap is largest under model
  mismatch.)
- **Upper bound is real:** Swift (*Nature* 2023) — champion-level FPV, won 15/25
  head-to-head vs three world-class pilots, fastest lap by ~0.5 s, >100 km/h, onboard-only
  sensing *during races*. But mocap was used for training/system-ID/gate-labels; no crash
  recovery; track-specific pretraining; sensitive to wind/lighting. High ceiling, high
  engineering+data cost.
- **Transferable sim-to-(black-box-)sim recipe** (with one correction): Swift trained PPO
  in sim, did **not** randomize platform dynamics, and closed the gap by **fine-tuning on
  residuals from ~50 s (3 rollouts)** — which *beat* a domain-randomization baseline.
  Residual model: **KNN, k=5, on state + commanded mass-normalized collective thrust,
  800–1000 samples**; perception residuals via Gaussian processes. **Correction the
  verifiers insisted on:** Swift's residual ID was **OFFLINE** (vs mocap ground truth),
  *not* in-flight online. For this team the gap to close is **PyBullet → DCGame**, so the
  analog is harvesting residuals from **DCGame rollouts** (the deployable sim) — a
  blueprint for the dormant residual MLP and the thrust/drag stub.

---

## 3 — Dynamic parameters from observables (online / in-flight)

- **Online-G from telemetry is achievable.** From live gyro-derived angular-accel +
  commands, an LMS update increments G by `expected − measured` accel; Smeur et al. (JGCD
  2016, canonical adaptive-INDI) ran this onboard a real MAV at **512 Hz**, and with enough
  excitation online G converges to the offline value. They prefer LMS over finite-horizon
  RLS (which "forgets" outside its window) — but that's a *design rationale*, not a
  theorem; variable-forgetting RLS and RLS-on-racing-hardware also work. *(The stronger
  "eliminates all bench-ID, only a coarse init needed" claim was **refuted 1-2** → keep a
  sane initial G.)*
- **ANDIa (Feb 2025)** is the most on-point method: recovers unknown effectiveness **and**
  compensates finite actuator bandwidth from real-time data only (no test signals / no
  training), validated on a real Z410 quad. *(It uses a virtual control matrix + estimator,
  so "online-G" is functionally — not literally — a G read-out; and "ANDIa *significantly
  beats* baseline INDI under coupled uncertainty+bandwidth" was **refuted 0-3** → expect
  *partial* compensation.)*
- **⚠ Caution — LINDI (2025):** deliberately avoids online estimation of the effectiveness
  term, training a net **offline** to dodge instability "when online models are not fully
  converged." This condemns online **learned-NN** G specifically — it does **not** condemn
  classical **LMS/RLS** online-G. *Excitation risk:* a smooth racing line may under-excite
  the roll axis, slowing LMS convergence.

---

## Verified findings (vote · source)

1. **INDI robustness is from measured-accel feedback; G is the only online residual.**
   3-0 · Tal & Karaman, IEEE T-CST [arXiv:1809.04048].
2. **Bench-ID-free online-G via LMS, 512 Hz onboard.** 3-0 · Smeur et al., JGCD 2016
   [doi:10.2514/1.G001490].
3. **ANDIa recovers unknown effectiveness + compensates bandwidth from real-time data
   only.** 3-0 · *Aerospace Sci. & Tech.* Feb 2025 [S1270963825001075].
4. **LINDI cautions against online *learned* G; trains offline instead.** 3-0 / 2-1 ·
   [arXiv:2503.09441].
5. **TOGT = path-shortening gain (not a verified bandwidth-aware planner).** 3-0; stronger
   feasibility claim refuted 1-2 · [arXiv:2309.06837].
6. **TOPPQuad = drop-in time-allocation over a fixed path; models no drag/motor
   dynamics.** 3-0 · [arXiv:2309.11637].
7. **Drag-aware differential-flatness feedforward cuts RMS error ~50%.** 3-0 · Faessler
   RA-L 2018 [rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf].
8. **INDI tolerates LARGE control-effectiveness errors (scalar-axis proof).** 3-0 ·
   [arXiv:1809.04048].
9. **Swift = champion-level RL ceiling, onboard-only during races.** 3-0 · Kaufmann et
   al., *Nature* 2023 [s41586-023-06419-4].
10. **RL's edge is the OBJECTIVE, not the approximator (same-objective → RL loses to
    MPC).** 3-0 / 2-1 · Song et al., *Science Robotics* 2023 [scirobotics.adg1462].
11. **Swift sim-to-real recipe: PPO + offline KNN k=5 residual fine-tune (~50 s), no
    platform-DR.** 3-0 · *Nature* 2023.
12. **THE CRUX IS UNRESOLVED — run online-G/adaptive-INDI as the empirical
    discriminator.** medium · [S1270963825001075], [arXiv:1809.04048], [doi:10.2514/1.G001490].

## Refuted — do NOT chase (killed by 3-vote verification)

1. ✗ "Online-G adaptive-INDI **eliminates** bench-ID; only a coarse init needed." (1-2)
   → keep a sane initial G.
2. ✗ "Effectiveness mismatch has a **distinct telemetry signature** = the un-removable
   fast oscillation (your 9 Hz limit cycle)." (0-3)
3. ✗ "Online adaptation **recovers nominal roll** regardless of initial G / inertia
   change." (0-3) — this is *why* the crux stays open.
4. ✗ "ANDIa **significantly beats** baseline INDI under coupled uncertain-effectiveness +
   limited-bandwidth." (0-3)
5. ✗ "TOGT enforces full 13-state + body-rate limits." (1-2) — path-shortening only.

## Open questions

1. **The crux** — no telemetry-only test cleanly separates bandwidth-limit vs mismatch;
   answer by trial (above).
2. How many **DCGame laps** are needed to fit a stable PyBullet→DCGame residual model?
3. Can a non-learning online-G (LMS/RLS) converge on a *smooth single track* without
   enough roll-axis excitation — and without LINDI's online-model instability?
4. Which published planner (if any) enforces a **true commanded-roll-rate / bandwidth
   constraint** so the trajectory side can respect 0.53× directly?

---

## Mapping to this codebase

| Roadmap item | Files to touch |
|---|---|
| #1 online estimator | `competition/calibration.py` (add recursive RLS/LMS alongside the batch lstsq), wire from `competition/aigp_mavlink.py` telemetry; tests in `tests/test_calibration.py` |
| #1 drag-aware FF | `control/mpc_tracker.py` (GeometricTracker feedforward path, alongside the `learned_residual` hook), consume drag coeff from estimator / `competition/drone_spec.py`; tests in `control/tests/test_tracker.py` |
| #2 adaptive-INDI / online-G | `competition/aigp_mavlink.py` `_attitude_error_body_rates` (the inner PD loop where 0.53 lives) → add a measured-accel INDI mode with online-G; gyro derivative + filter |
| #3 TOPPQuad re-timing | `planning/racing_spline.py` speed profile; add a commanded-roll-rate/bandwidth constraint sourced from the identified inner-loop bandwidth |

## Sources by angle (28 primary; key cited)

- **INDI / online-G:** [arXiv:1809.04048] (Tal & Karaman), [doi:10.2514/1.G001490]
  (Smeur LMS), [arXiv:2503.09441] (LINDI), [S1270963825001075] (ANDIa).
- **Trajectory-side:** [arXiv:2309.06837] (TOGT), [arXiv:2309.11637] (TOPPQuad),
  [RAL18_Faessler.pdf] (drag-FF).
- **Full RL ROI:** [s41586-023-06419-4] (Swift / *Nature*), [scirobotics.adg1462] (Song
  et al. / *Science Robotics*), plus arXiv 2412.11764, 2508.21065, 2504.21586, 2103.08624.
- **Lightweight ML:** arXiv 2503.09441, 2601.02738, 2305.17254, 2203.07747.
- **Online sysID / adaptive:** arXiv 2406.11723, 2302.07208, scirobotics.abm6597,
  2508.17577, 2409.12949, 2510.03100.
- **Race-day robustness (under-evidenced — see follow-up):** arXiv:2512.20475
  (drift-corrected VIO), 2510.13644, 2601.15222, [RSS'16 p081].

## Methodology & caveats

- **Time-sensitivity:** methods span 2016 (adaptive-INDI LMS, still canonical) → 2025
  (ANDIa, LINDI — newest, least independently corroborated, single source each).
- **Retrieval caveats:** ANDIa's full PDF was 403-blocked; its abstract was confirmed via
  3 independent search renderings + the foundational Smeur 2022 bandwidth paper.
  Science.org returned 403; confirmed via the arXiv version (2310.10943).
- **The headline crux is NOT answered** — treat online-G/adaptive-INDI as a recommended
  *experiment*, not a validated fix.
- **Scope mismatches to our setting:** Swift's residual ID was offline-on-hardware (analog
  = DCGame rollouts); TOGT's wins are on dense tracks; TOPPQuad needs an added bandwidth
  constraint; Faessler drag-FF is linear-drag, validated only to 5 m/s.

---

# Part 2 — Race-day robustness & VQ2 perception (follow-up run)

**Provenance.** Separate deep-research run (run id `wf_2cb08b01-e9d`): 5 search angles →
**23 primary sources** → 113 claims → **25 adversarially 3-vote-verified → 23 confirmed,
2 refuted** → 9 synthesized findings. Scoped to *only* robustness/perception/estimation
integrity (told not to re-cover INDI/trajectory/RL).

## TL;DR — VQ2

- **The champion recipe is unanimous and implementable:** detect gate **corners** with a
  lightweight CNN (segmentation or keypoint heatmaps — **not** direct 6-DoF pose
  regression) → planar **PnP/IPPE** against the **known map** → associate each detection to
  the **closest map gate** → fuse the global position into a **small KF that corrects ONLY
  translational** VIO drift (attitude is left to VIO, which drifts far less). Independently
  reproduced by **Swift** (*Nature* 2023), **"On Your Own"** (RA-L 2026), and the **KAIST
  AI-Grand-Prix** system (arXiv:2512.20475 — the recipe you named). This is the **#1
  reliability investment**.
- **It's worth ~2 orders of magnitude:** gate-corrected drift goes from **~50–100 m → sub-
  meter** at race speed (Gate-IO 0.48 m vs OpenVINS 98.2 m / SVO 47.2 m at ~70 km/h,
  arXiv:2210.15287). Classical VIO *fails* at race speed specifically from motion blur /
  high optical flow.
- **Embedded inference is a solved problem:** 10–40 ms/frame in FP16/TensorRT on Jetson
  Xavier/TX2/Orin NX.
- **The gap that bites you most is uncited:** champion systems do online robustness via
  **covariance-scaled measurement gating + RANSAC + offline self-calibration** — there is
  **NO published quadrotor-racing precedent** for NIS/chi-square integrity monitoring,
  RAIM, CUSUM, or **uniform gate-map offset / sign-flip detection**. Your exact failure
  mode (corrupted transferred map) must be solved with a **purpose-built** consistency test
  adapted from aerospace integrity practice.

## (1) Onboard monocular gate detection — corners, not pose

Consensus across all champions: **detect the 4 gate corners, recover pose by classical
geometry.** Concretely:
- **AlphaPilot** (arXiv:2005.12813 / RSS'16): ~160k-param 5-level U-Net segments the 4
  inner corners as Gaussian confidence maps + Part-Affinity-Fields; Hungarian matching →
  handles arbitrary/partial/overlapping gates; corners feed the EKF as **reprojection-error
  measurements** ("we do not infer the relative pose … but instead segment the four
  corners"). **10.5 ms** FP16/TensorRT on Jetson Xavier @ 60 Hz.
- **Swift** (*Nature* 2023): 6-level U-Net segments corners on greyscale T265; **40 ms
  (~25 Hz throughput)** on TX2 (note: 30 Hz is the *camera* rate, not the detector rate).
- **"On Your Own"** (arXiv:2510.13644, RA-L 2026): two-stage **YOLOv8n** bbox →
  **MobileNetV3-Small** keypoint heatmaps; **24–30 ms** on Orin NX.
- **MonoRace** (arXiv:2601.15222, **TU Delft, Jan 2026**): even the newest champion uses a
  **GateSeg** U-Net mask → classical corner extraction (LSD + RANSAC) → PnP — neural
  *segmentation*, not a learned pose regressor. **Won 2025 A2RL, beat all AI teams + 3 human
  world champions, monocular + IMU, no external tracking, ~100 km/h, ~16-state EKF.**
- **Deep Drone Racing** (arXiv:1905.09727, T-RO 2020) is the exception that proves the rule:
  it regresses a **waypoint+speed** (not a gate pose), trained entirely in sim, deployed
  zero-shot via domain randomization.

## (2) Gate-based global drift correction — the two templates

- **Translational-drift KF (Swift / On-Your-Own / KAIST):** PnP/IPPE per gate → assign to
  closest map gate → KF with a **3-state position-drift** (and optional drift-velocity)
  state; **only translation corrected**; stack multiple visible gates into one update;
  estimate measurement covariance **R via Monte-Carlo**. Simplest to bolt onto your EKF.
- **VIO-origin-misalignment EKF (AlphaPilot):** estimate the **SE(3) misalignment of the
  VIO origin frame** jointly with gate positions/heading, treating the VIO pose as a
  *constant parameter* (not a filter state); fuse **corner pixels** via the pinhole
  reprojection model; data-associate by **argmin Σ reprojection-error** (nearest-neighbor in
  reprojection space — note the paper does *not* call this a chi-square gate).

## (3)+(4) Map / state integrity — the build-it-yourself zone (your #1 item)

What champions actually ship for integrity: (a) **covariance-scaled measurement gating** —
MonoRace accepts a PnP update **iff `‖x_pos − x_PnP‖₂ < 16·Nc²·trace(P_pos)`**; (b)
**RANSAC** outlier rejection on corner matches + segmentation-based rejection of corrupted
image regions; (c) **offline** known-geometry **self-calibration** of camera extrinsics
between runs (reproject map gates, IoU-match, Bayesian-optimize extrinsics).

**What nobody published (the gap):** online **NIS/chi-square** estimator-integrity tests,
**RAIM**-style fault detection/exclusion, **CUSUM** change-detection, covariance-consistency
checks, or **uniform-offset / sign-flip** detection on a transferred map. These are *sound,
standard* aerospace-integrity techniques — but their racing application is an **engineering
inference, not a cited result**, and must be validated in DCGame before you trust them.
Targeted to your observed failure (sign-flipped X / Z≈−350): a **received-map self-
consistency check** against track invariants (inter-gate spacing, bounding box, expected Z
range) catches gross corruption cheaply, and **per-gate re-observation residuals with a
consistent same-direction bias across many gates ⇒ map error (not vehicle drift) ⇒ prefer
vision / abort**. This is the principled upgrade to your existing `_gate_map_is_sane`
(which catches out-of-bounds but not uniform shifts).

## ⚠ Key grounding caveat (open question #4)

The perception-robustness findings — motion blur, rolling shutter, exposure/lighting shift,
domain randomization, catastrophic forgetting (arXiv:2405.01054: naive sequential fine-
tuning grows error ~3–4×) — are largely **real-hardware** effects. **Your deployable target
is the black-box DCGame sim, not hardware**, so these may not surface or transfer. **Do not
over-invest in blur/lighting detector-hardening (roadmap item 5) for a sim-only VQ.** Your
real, observed robustness problem is **map corruption + sim degradation**, which is items
2–3, not item 5.

## VQ2 robustness roadmap (effort × payoff)

| # | Action | Effort | Risk | Payoff | Note |
|---|--------|--------|------|--------|------|
| **1** | Gate-based **global drift correction** — position-only KF update vs known map (associate to closest gate, stack multi-gate, MC-estimate R) | Med | Low | **Highest** | Sub-meter vs ~50–100 m; the VQ2 core. *Only when onboard vision is in the loop.* |
| **2** | **Covariance-scaled gating + RANSAC** on every vision update | **Low** | Low | High | MonoRace recipe; prevents one bad detection / corrupt-map gate poisoning the filter |
| **3** | **Online integrity + map-corruption monitor** (NIS/chi-square gate; CUSUM on innovations; per-gate re-observation residual → uniform-offset/sign-flip) | Med | Med | High | **No racing precedent — must build.** Directly your #1 failure mode. Validate in DCGame |
| **4** | **Offline self-calibration** of camera extrinsics from logs between runs | Med | Low | Med | MonoRace; corrects systematic perception bias pre-run |
| **5** | Detector hardening vs blur/lighting via **joint** domain randomization (avoid sequential FT) | High | Med | Med | **Likely hardware-only — defer for a sim-only VQ** (see caveat) |

Items **2–3 are the minimum viable race-day reliability upgrade for your *current* (map-
transfer) setup**; item 1 is the VQ2 onboard-perception build.

## Refuted — do NOT cite (killed)
1. ✗ "MonoRace demonstrated specific camera-interference / IMU-saturation tolerance at
   100 km/h." (1-2)
2. ✗ "Swift's authors explicitly named appearance-shift as their #1 failure mode." (1-2)

## Open questions (VQ2)
1. Exact NIS statistic / threshold (dof, covariance-scaling `k`) for your **15-state EKF** —
   no cited racing value; set empirically in DCGame.
2. How to disambiguate a **uniform map offset vs a sign-flip** online from telemetry alone;
   what residual signature separates a corrupt map from genuine VIO drift / degraded platform.
3. **When to trust vision over the prior map vs ABORT** — champions always trust the known
   map and only gate/RANSAC-reject individual detections; none publish a corrupt-map handover
   policy.
4. **Does DCGame even exhibit** motion-blur / rolling-shutter / exposure failure modes, or
   are those real-hardware-only (won't transfer in sim)? Determines whether item 5 matters.

## Sources by angle (VQ2; 23 primary)
- **Gate detection:** arXiv:2005.12813 (AlphaPilot) + [RSS'16 p081], PMC10468397 (Swift),
  arXiv:2510.13644 (On-Your-Own), arXiv:2601.15222 (MonoRace), arXiv:1905.09727 (Deep Drone
  Racing), arXiv:2405.01054 (continual-learning gate detection), arXiv:2311.02667 (TII dataset).
- **Gate-corrected VIO/drift:** arXiv:2512.20475 (KAIST AI-GP), arXiv:2210.15287 (Gate-IO /
  Learned Inertial Odometry), + Swift / On-Your-Own / AlphaPilot above.
- **Aerospace integrity (to adapt):** Joerger chi-squared RAIM FDE, arXiv:1909.08537,
  Emerald innovation-based FDI, PHM Society, "Online tests of Kalman filter consistency",
  Mourikis consistency TR.

## VQ2 methodology notes
- Most relevant refs are *very* recent (MonoRace Jan 2026; KAIST Dec 2025; On-Your-Own
  RA-L 2026) — re-check against the latest A2RL/AI-GP entries.
- **Venue note:** strongest sources describe **A2RL × DCL**; independent sources confirm the
  same minimal single-camera + low-quality-IMU constraint, so recipes transfer, but verify
  against the actual competition spec.
- Weaker points: domain-randomization-as-THE-mechanism was 2-1 (it's the *combination* of
  factors that matters); Gate-IO is a *baseline* in arXiv:2210.15287 (headline method is
  vision-free IMO) — the recipe is real but not the paper's endorsed direction.
