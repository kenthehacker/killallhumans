export const meta = {
  name: 'iter36-kv-centering-lever',
  description: 'Adversarially verify the kv (velocity-tracking gain) lever for cross-track centering on the AIGP minimal controller, before a live sweep',
  phases: [
    { title: 'Dissect', detail: 'three independent lenses analyze the clean baseline capture + controller code' },
    { title: 'Synthesize', detail: 'decide the change + live sweep plan + abort criteria' },
  ],
}

const REPO = 'C:/Users/Kenichi/projects/killallhumans'
const CAP = 'captures/iter36_min70_baseline.jsonl.gz'

const FACTS = `
CONTEXT — AIGP VQ1 drone racing, minimal pure-pursuit controller (control/minimal_controller.py).
Live-sim baseline (clean single run, fresh SIM_RESET before flight): cruise 7.0, 6/6 gates SIM-CREDITED
(race_finished=True), 0 collisions, 26.26s (4.33 s/gate). Capture: ${CAP}.

THE BINDING CONSTRAINT = cross-track (lateral Y) UNDERSHOOT at the Y-staggered gates. Per-gate closest-approach
error decomposed (along=X, lat=Y, vert=Z), worst gates:
  gate2: |err|=0.47  lat(Y)=-0.33 vert(Z)=-0.33   (follows a +3.7m Y swing)
  gate4: |err|=0.49  lat(Y)=-0.46 vert(Z)=-0.13   (follows a +4.3m Y swing)
Half-opening ~0.75m, so worst margin ~0.25m. Collisions appear when offset approaches the frame at higher cruise.

KEY MEASUREMENTS in the gate-2/3/4 approach windows (~1.8s before crossing):
  - lateral-accel clamp (g*tan(0.62)=7.0 m/s^2) is engaged 0% of ticks near these gates; |ah| is only ~2-3.
  - drone Y-velocity LAGS desired: e.g. gate2 vdes_y=+0.71 while vel_y=+0.34; accel_y=+1.12 = kv*(vdes-vel)=3.0*0.37.
  - So there is LARGE accel headroom (|ah| 2-3 vs clamp 7) but commanded catch-up accel is small BECAUSE kv=3.0.
  - Overall lateral-accel clamp engagement across the whole run = 3.5% (only startup/transition spikes).

HYPOTHESIS to verify/refute: raising kv (velocity-tracking gain, currently 3.0) tightens lateral velocity
tracking at gate transitions and reduces the undershoot, WITHOUT losing pure-pursuit's natural cross-track
damping (because v_des itself decelerates cross-track as the drone nears the gate — kv only makes ACTUAL
velocity follow v_des faster). This is DISTINCT from iter-34's cross_gain term (a high-gain Y POSITION term
with NO velocity damping) which overshot and clipped frames (3-14 collisions). It is also distinct from
iter-33's dismissal of kv, which was scoped to STARTUP (clamp-saturated regime), not gate-transition centering.

PRIOR NEGATIVE RESULTS (do not re-propose): cross_gain decoupling (FAILED, overshoot); trajectory/racing-line
mode (FAILED live this iter: infeasible 27.5 m/s^2 reference, 1/6 + 2 collisions); variable speed profile.

INNER LOOP RISK: AIGPMavlinkAdapter.send_attitude converts attitude->body-rate via a PD; the sim amplifies
commanded rate ~2.1-2.5x. An earlier too-high inner gain caused a ~9Hz LIMIT CYCLE (gyro p95 4.5) that
rectified thrust into a runaway climb. Current measured gyro p95=0.65, max 0.06 rad/s yaw (clean). Raising
the OUTER kv increases accel/tilt transients at transitions which feed the inner loop -> watch gyro for any
limit-cycle onset.
`

const ANALYST_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'kv_verdict', 'recommended_change', 'risks', 'better_alternative', 'confidence'],
  properties: {
    lens: { type: 'string' },
    kv_verdict: { type: 'string', enum: ['raise-kv-is-right', 'raise-kv-wrong', 'raise-kv-but-with-caveats'] },
    kv_recommendation: { type: 'string', description: 'recommended kv value or sweep range, with reasoning' },
    recommended_change: {
      type: 'object', additionalProperties: false,
      required: ['param', 'from', 'to', 'rationale'],
      properties: { param: { type: 'string' }, from: { type: 'string' }, to: { type: 'string' }, rationale: { type: 'string' } },
    },
    risks: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['risk', 'watch_metric', 'abort_threshold'],
        properties: { risk: { type: 'string' }, watch_metric: { type: 'string' }, abort_threshold: { type: 'string' } },
      },
    },
    better_alternative: { type: 'string', description: 'a lever better than raising kv, or "none — kv is the right lever" with reasoning' },
    evidence_checked: { type: 'string', description: 'what you actually read/computed from the capture+code to ground this' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
}

const lenses = [
  { key: 'control-theory', prompt: `LENS: CONTROL THEORY. Verify the closed-loop effect of raising kv on cross-track lag and damping. Read ${REPO}/control/minimal_controller.py (desired_velocity + compute). Confirm/refute that v_des is a naturally-damped pursuit target (cross-track component -> 0 at the gate) so raising kv preserves damping. Estimate the closed-loop cross-track time constant at cruise 7 and what kv would null the ~0.4m undershoot. Recommend a kv value/range.` },
  { key: 'adversarial-risk', prompt: `LENS: ADVERSARIAL / RISK. Your job is to BREAK the raise-kv plan. Read the capture ${REPO}/${CAP} (python+gzip+json) and ${REPO}/control/minimal_controller.py. Find the failure mode: does higher kv push tilt/gyro toward the inner-loop limit cycle? Does it amplify the gate-4 overshoot-then-undershoot transient? Does the startup (clamp-saturated) regime get worse? Quantify from the data what kv would START engaging the 7.0 clamp at transitions. Specify exact watch-metrics + abort thresholds for the live sweep.` },
  { key: 'alternatives', prompt: `LENS: ALTERNATIVES. Is there a lever BETTER than raising kv for the lateral undershoot? Consider: (a) anticipatory/lookahead aim toward the next gate's Y, (b) a time-matched lateral feedforward, (c) per-axis kv (higher for cross-track), (d) raising max_lateral_accel. Read ${REPO}/control/minimal_controller.py and ${REPO}/scripts/aigp_vq1_run.py and ${REPO}/race_pipeline.py to see how the next gate could be threaded in. For the slalom Y pattern (gate Y: -0.4,-2.5,+1.2,-5.1,-0.8,-4.4) reason carefully about whether anticipatory aim would HELP or HURT each gate. Rank the levers; if kv is best, say so.` },
]

phase('Dissect')
const analyses = await parallel(lenses.map(l => () =>
  agent(`${FACTS}\n\n${l.prompt}\n\nGround every claim in the actual capture/code. Do NOT run any live flight, sim connection, or aigp_vq1_run — analyze on-disk artifacts only. Return structured output.`,
    { label: `dissect:${l.key}`, phase: 'Dissect', schema: ANALYST_SCHEMA, agentType: 'general-purpose' })
))

const valid = analyses.filter(Boolean)
log(`Dissect done: ${valid.length}/3 analysts returned. kv verdicts: ${valid.map(a => a.kv_verdict).join(', ')}`)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['decision', 'param_changes', 'live_sweep_plan', 'abort_criteria', 'expected_outcome', 'dissent'],
  properties: {
    decision: { type: 'string', description: 'the single change to implement this iteration' },
    param_changes: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['param', 'from', 'to'],
        properties: { param: { type: 'string' }, from: { type: 'string' }, to: { type: 'string' } },
      },
    },
    live_sweep_plan: { type: 'array', items: { type: 'string' }, description: 'ordered list of live runs (each gets a fresh SIM_RESET via the runner)' },
    abort_criteria: { type: 'array', items: { type: 'string' }, description: 'metrics + thresholds that abort/back-off the sweep' },
    expected_outcome: { type: 'string' },
    dissent: { type: 'string', description: 'strongest minority view worth keeping in mind' },
  },
}

const synthesis = await agent(
  `You are the synthesizer for an AIGP control-tuning iteration. Three analysts examined the raise-kv hypothesis from control-theory, adversarial-risk, and alternatives lenses. Their structured findings:\n\n${JSON.stringify(valid, null, 2)}\n\n${FACTS}\n\nDecide the SINGLE change to implement and test live this iteration, the ordered live sweep (the live sim is the ground truth; each run gets a fresh in-sim SIM_RESET; baseline to beat = cruise 7.0, worst offset ~0.5m, 0 collisions, gyro p95 0.65), and the abort criteria. Be decisive. Return structured output.`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA, agentType: 'general-purpose' })

return { analyses: valid, synthesis }
