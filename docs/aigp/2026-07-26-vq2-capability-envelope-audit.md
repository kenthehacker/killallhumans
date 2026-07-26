# Build-3385 control capability-envelope audit

This audit separates genuine FlightSim/transport safety requirements from
measured build-3385 plant evidence and from limits inherited from earlier
bounded development experiments.  It covers the production
`visual-course` body-rate/thrust path; detector confidence and lifecycle
admission thresholds are navigation policy, not plant capability.

## Simulator, protocol, and standing safety requirements

| Constraint | Classification | Production disposition |
| --- | --- | --- |
| IMU-only `SET_ATTITUDE_TARGET`, rates/thrust type mask, finite floats, wire thrust in `[0, 1]` | Simulator/MAVLink interface | Retain |
| Build-3385 controller-to-wire rate signs `(-1, -1, -1)` | Measured interface behavior | Retain |
| 50 Hz release pacing, missed-tick drop, below the simulator's 100 Hz command limit | Simulator plus standing safety policy | Retain |
| Reset epoch/rollback, countdown, GO+150 ms, fresh heartbeat/IMU/race/camera/actuator, authoritative race-only credit | Simulator plus standing safety policy | Retain |
| Exact race-gate ownership at wire start, hard attempt timeout, collision abort, host-wide live lease | Standing safety policy | Retain |
| Broad watchdog: roll `+/-25 deg`, pitch `-35/+10 deg`, body rate `2 rad/s` sustained or `3 rad/s` immediate | User-designated outer safety envelope | Retain unchanged |
| Best-effort zero, disarm, reset, final disarm and usable-next-run check | Standing cleanup policy | Retain; report separately from navigation |

## Measured build-3385 plant evidence

| Channel/fact | Evidence | What it proves |
| --- | --- | --- |
| Yaw at `+/-0.08 rad/s` | Three clean runs `20260725T060252Z-calibration-excite-0726702b`, `20260725T060328Z-calibration-excite-9e3562b1`, and `20260725T060354Z-calibration-excite-d78746e7` | Correct body/image sign; gyro gain `1.841-1.846`; gyro-delay upper bound `52.6-55.9 ms`; first-image delay `47.5-70.5 ms`; peak measured yaw `0.221-0.222 rad/s`; axial excursion about `0.025 rad` |
| Roll/pitch sign at small excitation | July-18 sign-ID handoff | Correct signs and a response at the tested point; not the maximum usable rate or attitude |
| Launch/collective | Repeated Gate-0 and hover runs | `0.32` provides launch margin; roughly `0.275-0.295` supports the current flight regime; neither is a maximum thrust capability |
| Free-flight yaw response | Recent visual-course traces | Course yaw actuation matches the accepted sign and approximate gain and has reached about `0.12 rad` excursion without the broad watchdog firing |
| Spawn attitude | July-18 handoff | Pitch is approximately `-0.31 rad`, so pad-loaded yaw calibration does not by itself identify the free-flight coupled plant |

The accepted yaw evidence supports production authority through
`+/-0.08 rad/s`.  It does not show that `0.08` is the plant maximum, that the
response remains linear beyond it, or that the pad-loaded coupling matches
free flight.

## Inherited bounded-experiment limits

| Current limit | Value | Audit conclusion |
| --- | ---: | --- |
| Generic runner command-rate clamp | `+/-0.25 rad/s` | Conservative development clamp, not a MAVLink or measured plant maximum; retain as the ceiling for the first progressive sweep, then reassess from evidence |
| Generic runner thrust clamp | `[0, 0.35]` | Conservative development envelope, not protocol capability; retain until thrust characterization |
| Visual-course yaw command cap | `+/-0.08 rad/s` | Accepted operating point but not maximum; production stays here until a new measured profile is adopted |
| Visual target roll | `+/-0.12 rad` | Controller experiment limit; not a demonstrated bank capability |
| Visual target pitch | `[-0.30, +0.10] rad` | Controller experiment/trajectory range; not the outer safety envelope |
| Visual thrust | `[0.21, 0.32]` | Stage tuning range; not the plant thrust envelope |
| Visual measured roll/pitch/rate | roll `+/-0.18`, pitch `[-0.35,+0.15]`, rate `0.50` | Narrow stage corridor duplicated inside the broad watchdog |
| Visual segment yaw excursion/soft stop | `0.65/0.60 rad` | Course-turn policy without direct calibration support; it must not be confused with a plant limit |
| Alignment/recovery caps | rate `0.12`, yaw `0.08`, thrust `0.285-0.30`, several `0.05-0.18 rad` attitude deltas | Historical stage experiment limits; they are not generic production capability |
| Corridor dwell, bbox/rate/scale/censor bounds, exact transition timers | Trace-derived navigation confidence | Keep only where causally needed for observation validity or crossing safety; never use them as actuator capability |

The controller currently applies several of these narrow limits in series:
the image servo clamps target attitude/yaw/thrust, the stage clamps body rates
and measured attitude again, the yaw profile adds a projected excursion stop,
and the generic runner applies another command and broad-watchdog layer.  The
future allocator should choose continuous authority inside one measured
channel envelope, while the runner retains only the broad outer watchdog and
protocol invariants.

## First capability characterization

The first code-owned free-flight sweep leaves production visual-course
authority unchanged.  It:

1. uses the proved hover launch to leave the pad and level the vehicle;
2. holds zero roll/pitch attitude target and `0.285` collective;
3. sends symmetric yaw pairs at `+/-0.08`, `+/-0.12`, `+/-0.16`, and
   `+/-0.20 rad/s`, with neutral dwells and exact 50 Hz pacing;
4. records exact controller and wire commands, body rates, attitude, target
   motion, response delay, gain, and roll/pitch coupling;
5. aborts on the unchanged broad watchdog, collision, race change, stale
   stream/target, non-finite command, lease loss, or hard deadline.

The sweep never jumps to MAVLink limits and remains below the existing
`+/-0.25 rad/s` conservative command clamp.  A successful result may justify
a new conservative yaw profile below the largest clean, responsive level.
The same method then characterizes roll, pitch, and thrust before those
historical stage caps are treated as production capability.

## Continuous authority-allocation target

The generic allocator should use visual bearing error and error rate,
apparent-scale/closure rate, current attitude/body rates, and observation
confidence:

- large off-axis error or rapid closure: allocate coordinated yaw and bank,
  reduce nose-down pitch and collective-driven closure;
- centered, converging target: increase forward progress;
- rapid expansion: brake closure before censorship;
- censored or uncertain geometry: reduce closure while retaining steering on
  observable axes and increasing uncertainty on censored axes.

Lifecycle and race authority remain separate.  Vision selects bounded
navigation commands; only authoritative race status credits a gate.
