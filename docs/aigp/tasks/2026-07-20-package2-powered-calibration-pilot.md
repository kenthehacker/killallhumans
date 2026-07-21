# Package 2 powered calibration discovery pilot

- Task ID: `vq2-package2-powered-calibration-pilot`
- Parent: `vq2-package2-production-calibration`
- State: `I0 contract frozen and independently cleared - implementation not started; no simulator contact`
- Starting main commit:
  `ccbea8ac9fa9b53c3f86324662f616041693277b`.
- R0 contract commit: `49b331f`.
- P0 scope-pivot commit: `76dbebb`.
- Branch: `package2-powered-calibration`.
- Worktree:
  `C:\Users\John\aigp-worktrees\wt-package2-powered-calibration`.
- Future integration-owner-controlled detached live worktree, created only from
  the exact T1-integrated local-main commit and designated as the sole powered
  integration worktree for this task:
  `C:\Users\John\aigp-worktrees\wt-package2-powered-calibration-live`.
- Owner and integration owner: `/root`.
- Heartbeat date: `2026-07-20`.
- Simulator target: FlightSim build 3385, Training mode.
- Private root, not created by the contract freeze:
  `C:\Users\John\aigp-evidence\2026-07-20-package2-powered-calibration-pilot`.
- Maximum live scope after every entry gate passes: one accepted powered
  discovery session; the first failed live attempt ends live work.

## Authority and non-transferable boundary

On 2026-07-20 the user approved preparation of the separately frozen Package 2
powered-calibration successor and asked that routine in-scope work be treated
as approved without repeated permission prompts. For this exact task, that is
recorded as fresh user authorization to implement and, only after every frozen
entry gate below passes, execute the single exact `calibration-excite` pilot.
The standing instruction removes repeated user-authorization prompts for later
in-scope work, but no later semantic change, second attempt, different stage,
Gate 0 or Gate 1 flight, or other package inherits this task's exact attempt or
safety contract. Any such work requires its own frozen scope and entry gates.

On 2026-07-20 the user then selected the active reference and data boundary:

- do not inspect, extract, or otherwise read the cooked PAK for this task;
- gate dimensions and similar target facts may be explicit, configurable
  nominal inputs whose values and applicability can change;
- camera, IMU, timing, resolution, and stream facts must come from the actual
  build-3385 JPEG, `HIGHRES_IMU`, heartbeat, and race-status boundary; and
- private decoded-frame capture and analysis are permitted for this simulation
  task only, not for a later physical, HIL, submitted, or public-release phase.

The approval does not create facts that are absent from the build, waive the
authoritative VQ2 safety contract, establish organizer media credentials, or
turn configured target geometry into simulator-observed calibration truth.
Missing reference, process, lease, test, cleanup, or evidence proof stops the
task without simulator contact or powered execution.

## Objective and exact exit

This pilot may establish only whether one bounded build-3385 session can
preserve exact camera/IMU lineage while applying a predeclared, nonparallel,
reversing roll/pitch command plan while viewing a hash-admitted nominal target
configuration whose build-3385 geometry linkage remains unverified. Its output
is discovery/fit-only acquisition evidence, not an independently admitted
Package 2 calibration reference.

The pilot does not fit or accept camera intrinsics, distortion, camera-to-body
rotation, clock skew, time offset, an absolute source-to-host map, covariance,
or empirical error limits. It does not open held-out or repeat data, complete
Package 2, wire calibration into estimation/runtime, select a transport
command, approach or pass a gate, or authorize another live stage.

A valid capture exits at one clean, independently reviewed discovery artifact.
It makes no structural-rank, conditioning, practical-identifiability, or
calibration-feasibility claim. Observed motion and image support are descriptive
only. A later immutable pre-fit design and a new hash-bound simulation-only
data-use authority naming the exact F00 artifact hashes must be admitted before
that successor may open or use F00. It must freeze the camera/distortion model,
parameter order/units/scaling, visual-rotation construction, gyro interpolation
and integration, nuisance model, covariance construction, rejection rules, and
profiled Jacobian before it may inspect this session for rank or propose the
full `F01-F04`, `L01-L02`, `H01-H02`, and `R01-R02` collection. Failure exits
at an immutable invalidation record and cleanup proof. Either exit returns to a
new reviewed task before further powered work.

## Superseding active phase gates

The former cooked-PAK route is retired from the active path. Commits `49b331f`
and `444a1b3` remain historical evidence that the original contract and the
synthetic-only parser were reviewed; no production PAK read occurred. The
parser and its tests may remain in history and in the worktree, but neither is
an active prerequisite or permitted production invocation. Reopening that route
requires a new explicit contract.

The active phases are strictly ordered:

1. `P0 scope pivot`: record the four user decisions above, independently review
   this correction, and commit it on the exact candidate. This supersedes only
   the former PAK-specific R2/R3 route; the powered safety contract is unchanged.
2. `P1 nominal protocol admission`: implement, validate, hash, and independently
   admit one explicit collection configuration defining the nominal target,
   observed-stream contract, and simulation-only data scope. This authorizes
   discovery collection only. It does not verify the configured geometry against
   build 3385 or accept any calibration value. No simulator process or port is
   touched before this gate.
3. `I0 powered implementation`: the remaining owned executable surfaces and
   direct tests are implemented from the frozen contract.
4. `T0 non-live`: direct, VQ2, fast, unit, full-non-live, and exact hash-pinned
   VQ2 evidence passes from the required clean worktrees.
5. `T1 integration`: the integration owner integrates the exact reviewed
   implementation into current local `main`, runs post-merge VQ2, and proves
   tracked main clean. Only that exact integrated commit may become live code.
6. `L0 live freeze`: the exact integrated commit, integration-owner-controlled
   detached live worktree created from it and designated as this task's sole
   integration live worktree,
   interpreter/environment/import inventory, launch command, child command,
   private paths, configuration and authorization hashes, excitation ID,
   attempt ID, build/process proof, lease, outbound allowlist, phase deadlines,
   cleanup fallback, and invalidation rules pass independent review.
7. `L1 pilot`: at most one powered discovery session is attempted.
8. `E0 review`: evidence is validated offline and independently reviewed
   before any fit or successor collection is proposed.
9. `F0 pre-fit successor`: before opening or using F00, admit a new hash-bound
   simulation-only data-use authority naming its exact hashes, then freeze a
   separate model/reference design, geometry-uncertainty treatment, nuisance
   ledger, splits, rank definition, and later held-out/repeat collection.
   Simulation artifacts cannot enter a physical phase without a new authority
   and calibration chain.

Failure or incompleteness at one phase cannot be repaired by skipping to a
later phase.

## Owned tracked surfaces

The implementation candidate may own only:

- this task record;
- `config/aigp_vq2_calibration_target_build3385.json` for the explicit nominal
  target and simulation-only collection policy; it is never an implicit default;
- `scripts/aigp_vq2_calibration_target.py` for strict offline configuration
  validation and identity;
- `competition/vq2_capture.py` for additive immutable receive/outbound schemas;
  `competition/aigp_mavlink.py` for their envelopes/receipts plus opt-in powered
  endpoint injection/source gate, atomic calibration reset boundary/collision
  diagnostics, guarded outbound dispatch, and deadline-aware connect/disconnect;
  defaults and passive callers remain unchanged; their direct tests;
- `competition/vq2_vision.py` only for the opt-in exclusive powered camera-bind,
  loopback source freeze/rejection, and source diagnostics seam; its direct
  test;
- `scripts/aigp_live_lease.py` for an opt-in powered-only `/2` lease/takeover
  protocol while preserving passive `/1` behavior; its direct test;
- `scripts/aigp_vq2_powered_runtime.py` for shared Windows process-handle,
  capability, QPC-deadline, exclusive-UDP, port-owner, and process-tree proof;
  its direct test;
- `scripts/aigp_vq2_powered_attempt.py` for strict plan, live-freeze, attempt,
  command-event, terminal, and split schemas; its direct test;
- `scripts/aigp_vq2_run.py` for one additive fixed
  `calibration-excite` stage;
- `scripts/aigp_vq2_powered_calibration_probe.py` for the exact live wrapper;
- `scripts/aigp_vq2_powered_calibration_analysis.py` for pure evidence
  validation and acquisition-support summaries;
- `scripts/aigp_vq2_powered_cleanup.py` for a cleanup-only out-of-process
  fallback after a confirmed child exit;
- direct tests named after those scripts;
- additive direct cases in `tests/test_aigp_vq2_runner.py`; and
- additive compatibility cases in `tests/test_aigp_loop_replay.py` proving the
  frozen replay `/1` core was not widened; and
- `scripts/dev.ps1` only if needed to include the new offline tests in the
  explicit VQ2 suite.

The already committed `scripts/aigp_vq2_build_reference.py`, its direct test,
and its VQ2 test registration are inert historical surfaces. This candidate
does not invoke, extend, or use them. Their presence grants no PAK authority and
no PAK identity may enter the active configuration, attempt, capture, or report.

The candidate does not own `aigp_loop/replay.py`, `competition/vq2_contracts.py`,
the passive collector/analyzer, or either frozen replay-record/bundle `/1`
schema. The behavior candidate does not own promotion policy or trusted-manifest
files. Any later integration-owner change to those files occurs only after
candidate acceptance and is reviewed separately. No powered command is added
to `dev.cmd` or any generic test task.

## Excluded surfaces and claims

The task must not change detector behavior, fitted gate geometry, guidance,
controller envelopes, estimator state, runtime selection, supervisor
authority, MAVLink rate signs, command masks, the passive timing collector,
replay acceptance, Package 1 data policy, or gate-passage logic. It must not
reuse VQ1 camera intrinsics, camera tilt, FOV, distortion, pose/map streams, the
pixel-square prior, bbox corners, fitted `/1` corners, synthetic identity
calibration, candidate-generated labels, or the completed static passive tranche
as reference truth. Public VQ1 gate dimensions are permitted only through the
explicit nominal configuration and must remain labeled unverified for build
3385 Training.

Private images, decoded frames, captures, annotations, invalidation records,
analysis reports, dependency inventories, and operational authorization records
remain outside Git. The public nominal configuration may be tracked. Generated
evidence is not organizer or license approval and is never physical-phase or
public-release authority.

## Active P1 nominal protocol admission

P1 uses only `scripts/aigp_vq2_calibration_target.py`, the tracked nominal
configuration, and a private simulation-capture authorization. The validator is
standard-library-only and offline. It has no simulator, socket, PAK, asset,
network, subprocess, or write-back path. It reads only paths explicitly supplied
on its command line and emits no derived camera or geometry value.

The tracked `aigp-vq2-sim-calibration-collection-config/1` document has exact
top-level keys `schema`, `config_id`, `revision`, `simulator`, `source`,
`applicability`, `geometry`, `observed_streams`, `calibration_status`, and
`data_scope`. All nested objects reject unknown or missing keys:

- `config_id` and `revision` are nonempty strings. A changed value, feature,
  stream contract, or use scope requires a new ID/revision and hash; existing
  files and results are never overwritten or relabeled.
- `simulator` is exact integer build `3385` and mode `Training`.
- `source` records kind `public_technical_spec`, document `VADR-TS-002`, issue
  `00.02`, publication date `2026-05-08`, the official document URL, stated
  scope `Virtual Qualifier 1`, and use scope `nominal_geometry_only`. It is not
  provenance for the camera, IMU, stream, build linkage, or any calibration.
- `applicability.status` is exactly
  `nominal_unverified_for_build_3385_training`; configured values are never
  described as build-reported, PAK-derived, or independently measured.
- `geometry` uses metres and explicitly freezes a right-handed target frame,
  outer width/height `2.7/2.7`, inner width/height `1.5/1.5`, depth `0.26`, and
  the clockwise front-view feature order `top_left`, `top_right`,
  `bottom_right`, `bottom_left` for front-inner-aperture boundary intersections.
  These are configurable nominal values, not hidden code constants. Geometry
  uncertainty is `unpublished_unknown`; this is allowed for collection only and
  blocks any calibration fit or acceptance until F0 supplies a reviewed bound.
- `observed_streams` separates measurement, lifecycle, and safety/audit
  streams. Measurement inputs are UDP JPEG and `HIGHRES_IMU`; heartbeat and
  race status provide lifecycle/safety authority; actuator output and collision
  events are mandatory watchdog/audit evidence. The latter four are never
  calibration measurements. The camera contract includes frame, stream,
  generation, publication, decode, consume/work, and host-timing lineage. The IMU contract
  includes raw accelerometer/gyro values, source token, MAVLink generation and
  sequence, and host receipt. Decoded dimensions are observed from the first
  decoded frame, must equal the parent-frozen `640 x 360` before arm, and must
  remain stable. A different first shape stops before arm and requires a revised
  contract/config; it is not resized or silently accepted. This is a runtime
  observation gate, not an intrinsic prior. No focal length, principal point,
  FOV, distortion, exposure timing, or mount is supplied as a default. The
  config explicitly rejects `ATTITUDE`, pose, odometry, and track/gate-map
  geometry as build-3385 truth.
  `ACTUATOR_OUTPUT_STATUS` carries the admitted MAVLink ingress/host receipt.
  `COLLISION` currently carries only runner drain/observation order plus ID,
  threat, and impulse; it has no receiver receipt or source timestamp, and this
  task must not manufacture one.
- `calibration_status` fixes intrinsics, distortion, camera-to-body rotation,
  camera/IMU time model, rank, covariance, and empirical limits to `uncomputed`.
- `data_scope` fixes `private_simulation_capture=true`,
  `physical_or_hil_use=false`, `submitted_run_use=false`,
  `public_release=false`, `external_service_upload=false`,
  `git_storage=false`, and `pak_access=false`.

Every JSON number is an exact JSON integer or finite JSON number as required;
booleans never satisfy numeric fields. Build is the exact integer `3385`.
Geometry width, height, and depth are finite and strictly positive; inner width
and height are strictly smaller than their corresponding outer dimensions. The
feature array contains exactly four finite three-vectors in the frozen order,
and must exactly equal the four `(+/- inner.width/2,
+/- inner.height/2, 0.0)` combinations specified below. A mismatch between
dimensions, coordinates, frame convention, or order fails validation.

The initial tracked value is frozen before implementation as:

```json
{
  "schema": "aigp-vq2-sim-calibration-collection-config/1",
  "config_id": "vq2-build3385-training-gate0-nominal-v1",
  "revision": "1",
  "simulator": {"build": 3385, "mode": "Training"},
  "source": {
    "kind": "public_technical_spec",
    "document_id": "VADR-TS-002",
    "issue": "00.02",
    "publication_date": "2026-05-08",
    "url": "https://www.theaigrandprix.com/wp-content/uploads/2026/05/260508_Technical_Spec_0002.pdf",
    "stated_scope": "Virtual Qualifier 1",
    "use_scope": "nominal_geometry_only"
  },
  "applicability": {
    "status": "nominal_unverified_for_build_3385_training",
    "result_semantics": "conditional_on_nominal_gate_config",
    "replacement_policy": "new_config_id_revision_and_hash"
  },
  "geometry": {
    "units": "m",
    "target_frame": {
      "handedness": "right",
      "origin": "front_inner_aperture_center",
      "x_axis": "front_view_right",
      "y_axis": "front_view_down",
      "z_axis": "front_to_back"
    },
    "outer": {"width": 2.7, "height": 2.7},
    "inner": {"width": 1.5, "height": 1.5},
    "depth": 0.26,
    "feature": {
      "kind": "front_inner_aperture_boundary_intersections",
      "order": ["top_left", "top_right", "bottom_right", "bottom_left"],
      "coordinates": [
        [-0.75, -0.75, 0.0],
        [0.75, -0.75, 0.0],
        [0.75, 0.75, 0.0],
        [-0.75, 0.75, 0.0]
      ]
    },
    "uncertainty_status": "unpublished_unknown"
  },
  "observed_streams": {
    "camera": {
      "transport": "udp_jpeg",
      "stream_id": "vq2-camera-udp-5600",
      "frame_id_field": "frame_id",
      "source_time_field": "sim_time_ns",
      "source_time_semantics": "opaque_ordering_token_not_calibrated_capture_time",
      "identity_schema": "aigp-vq2-frame-identity/1",
      "timing_schema": "aigp-vq2-frame-timing/1",
      "consume_timing_schema": "aigp-vq2-camera-frame-timing-observation/1",
      "expected_decoded_dimensions": {"width": 640, "height": 360},
      "decoded_dimensions_policy": "observe_require_exact_before_arm_and_session_stability",
      "host_receipt_clock": "host-perf-counter"
    },
    "imu": {
      "message": "HIGHRES_IMU",
      "source_time_field": "time_usec",
      "source_time_semantics": "opaque_source_clock_for_ordering_and_integration",
      "accel_fields": ["xacc", "yacc", "zacc"],
      "gyro_fields": ["xgyro", "ygyro", "zgyro"],
      "ingress_schema": "aigp-vq2-mavlink-ingress/1",
      "sample_schema": "aigp-vq2-received-imu/1",
      "host_receipt_clock": "host-perf-counter"
    },
    "race_status": {
      "required_fields": [
        "active_gate_index",
        "last_gate_race_time",
        "race_finish_time_ns",
        "race_start_boot_time_ms",
        "sim_boot_time_ms"
      ],
      "source_time_field": "sim_boot_time_ms",
      "ingress_schema": "aigp-vq2-mavlink-ingress/1",
      "host_receipt_clock": "host-perf-counter"
    },
    "heartbeat": {
      "required_fields": ["base_mode", "custom_mode"],
      "source_time_semantics": "no_admitted_source_timestamp",
      "ingress_schema": "aigp-vq2-mavlink-ingress/1",
      "host_receipt_clock": "host-perf-counter"
    },
    "safety_audit": {
      "actuator": {
        "message": "ACTUATOR_OUTPUT_STATUS",
        "ingress_schema": "aigp-vq2-mavlink-ingress/1",
        "host_receipt_clock": "host-perf-counter"
      },
      "collision": {
        "message": "COLLISION",
        "lineage_semantics": "runner_drain_observation_order_only_no_receiver_receipt_timestamp"
      },
      "semantics": "watchdog_and_evidence_only_not_calibration_inputs"
    },
    "unavailable_as_truth": [
      "ATTITUDE",
      "LOCAL_POSITION_NED",
      "ODOMETRY",
      "track_gate_map"
    ]
  },
  "calibration_status": {
    "intrinsics": "uncomputed",
    "distortion": "uncomputed",
    "camera_to_body_rotation": "uncomputed",
    "camera_imu_time_model": "uncomputed",
    "rank": "uncomputed",
    "covariance": "uncomputed",
    "empirical_limits": "uncomputed"
  },
  "data_scope": {
    "private_simulation_capture": true,
    "physical_or_hil_use": false,
    "submitted_run_use": false,
    "public_release": false,
    "external_service_upload": false,
    "git_storage": false,
    "pak_access": false
  }
}
```

The private `aigp-vq2-simulation-capture-authorization/1` record has exact keys
`schema`, `authority`, `task_id`, `domain`, `simulator`, `session_ids`,
`allowed_purposes`, `allowed_classes`, `storage`, `retention`, `transfer`,
`organizer_media_credential`, and `publication_permitted`. It records the
2026-07-20 user/operator decision, this exact task, domain `simulator_only`,
build 3385 Training, and only session `F00`. Its allowed purposes are collection
discovery, offline replay/analysis, integrity audit, and independent review.
Allowed classes cover UDP/JPEG/decoded frames, exact stream and host timing,
HIGHRES_IMU/race/heartbeat/actuator/collision/command records, process/lease/
cleanup evidence, annotations/crops/features, and content-bound derivatives.

Storage is restricted to the current-user-only private task root, never Git,
public release, network export, or external-service upload. Evidence is retained
through this simulator audit, then sealed and quarantined pending an explicit
disposition; it is not automatically deleted. No successor task, new session,
build/mode, submitted run, physical/HIL phase, or publication inherits the
authority. This is user/operator authority, not an organizer media credential.
Every session and derivative binds the authorization byte SHA-256, target-config
byte SHA-256, candidate commit, and parent hashes.

The private record's semantic value is frozen as follows; P1 records its exact
stable bytes and byte SHA-256 without adding identity-bearing commentary:

```json
{
  "schema": "aigp-vq2-simulation-capture-authorization/1",
  "authority": {
    "kind": "user_operator",
    "authority_id": "conversation-2026-07-20-package2-sim-capture",
    "authorized_on": "2026-07-20",
    "source": "direct_user_instruction"
  },
  "task_id": "vq2-package2-powered-calibration-pilot",
  "domain": "simulator_only",
  "simulator": {"build": 3385, "mode": "Training"},
  "session_ids": ["F00"],
  "allowed_purposes": [
    "calibration_discovery",
    "independent_review",
    "integrity_audit",
    "offline_replay_and_analysis"
  ],
  "allowed_classes": [
    "annotations_crops_features",
    "commands",
    "decoded_frames",
    "derived_replay_and_analysis",
    "highres_imu",
    "process_lease_cleanup",
    "race_heartbeat_actuator_collision",
    "reconstructed_jpegs",
    "source_and_host_timestamps",
    "udp_camera_datagrams"
  ],
  "storage": {
    "private_root": "C:\\Users\\John\\aigp-evidence\\2026-07-20-package2-powered-calibration-pilot",
    "git": false,
    "public_release": false,
    "network_export": false,
    "external_service_upload": false
  },
  "retention": {
    "through": "package2_simulator_audit_closeout",
    "after": "sealed_quarantine_pending_explicit_disposition",
    "automatic_deletion": false
  },
  "transfer": {
    "successor_task": false,
    "new_session": false,
    "new_build_or_mode": false,
    "submitted_run": false,
    "physical_or_hil": false
  },
  "organizer_media_credential": false,
  "publication_permitted": false
}
```

Both JSON inputs reject duplicate keys, non-UTF-8, BOM, nonfinite numbers,
booleans where numbers are required, unknown/missing keys, unsorted/duplicate
set-like arrays, and noncanonical hash text. Their identities are SHA-256 over
the exact stable bytes after strict semantic validation. There is no default
path and no field-level CLI override. The live wrapper requires both absolute
paths and exact reviewed hashes before simulator contact.

P1 admission additionally requires independent review to verify that:

1. no PAK path, PAK identity, asset package, mesh, LOD, material, or map linkage
   is active or enters any collection identity;
2. no public/VQ1 camera value is a default, prior, or acceptance oracle;
3. the frozen collector schema requires every captured frame to carry actual
   decoded dimensions, content hash, frame/stream/generation/publication
   identity, opaque camera token, and complete `FrameTimingV1` host lineage;
4. the frozen collector schema requires every IMU occurrence to carry actual
   raw `HIGHRES_IMU` accelerometer/gyro values, opaque source token, ingress
   generation/sequence, and host receipt;
5. nominal geometry feature semantics and corner order are frozen for a future
   F0 design only. P1 and L1 neither label nor admit/reject feature
   correspondences; F0 must freeze visibility, ambiguity, correspondence,
   rejection, annotation, and uncertainty policies before opening F00;
6. configuration cannot alter excitation, thrust, rate, duration, safety,
   watchdog, lease, deadline, cleanup, or attempt bounds; and
7. review accepts collection readiness only, not calibration readiness.

P1 proves only these schemas, literals, policies, and offline identities. I0
implements their binding, T0 proves the mechanism with non-live tests, L0
freezes the exact implementation, and only L1/E0 can prove actual occurrences.
No future-phase evidence is back-claimed at P1.

The published dimensions condition the discovery protocol but do not prove the
active rendered geometry. The simulator wire cannot verify metric gate size,
mesh/LOD, lens model, camera mount, or exposure delay. Every result therefore
says `conditional_on_nominal_gate_config`, never `verified_build_geometry`.

## Historical inactive R1/R2/R3 PAK route

Everything in this historical subsection through the discovery identities is
retained for audit only and is superseded by P0/P1 above. It creates no active
entry gate, and none of its production commands may be invoked in this task.

R1 may implement only `scripts/aigp_vq2_build_reference.py`, its strict schemas,
and its direct test. It uses only the Python standard library and synthetic PAK
fixtures during R1; no dependency or lock surface is added. The production tool
is a read-only candidate extractor and validator. It cannot self-attest semantic
linkage, uncertainty, rules permission, independence, or admission. Missing,
opaque, inferred, defaulted, or self-certified fields fail closed.

R2 requires an exact private
`aigp-vq2-build-reference-rules-clearance/1` JSON record before any production
tool call reads the real PAK. Its keys are exact: schema literal; nonempty
`record_id`, `reviewer`, `reviewed_at_utc`, and `authority_basis` strings;
`build_sha256` containing exact lowercase SHA-256 strings for `launcher`,
`payload`, and `pak`; a sorted unique nonempty `asset_scope` string array;
booleans `local_read_only_derivation_permitted` and
`competition_use_permitted`, both exact `true`; and a `publication_limits`
string array. The file's byte SHA-256 is the clearance identity. Tool validation
does not substitute for the human authority asserted by the record.

The production CLI has exactly two subcommands and no abbreviation, source,
selector, force, or overwrite option:

```powershell
& 'C:\Users\John\killallhumans\.venv\Scripts\python.exe' `
  -E -s -B -m scripts.aigp_vq2_build_reference inspect-build `
  --rules-clearance <private-absolute-clearance-json> `
  --output <private-absolute-new-discovery-json>

& 'C:\Users\John\killallhumans\.venv\Scripts\python.exe' `
  -E -s -B -m scripts.aigp_vq2_build_reference validate-candidate `
  --rules-clearance <private-absolute-clearance-json> `
  --candidate <private-absolute-candidate-json> `
  --output <private-absolute-new-validation-json>
```

The source paths and expected hashes are code constants: the launcher, payload,
and PAK named in this record. Inputs must be absolute, existing, regular,
non-reparse files. Output must be an absolute, nonexistent, non-reparse path
below the task's private `reference` root; its parent must already exist and be
current-user-only. Creation uses exclusive mode, flush plus `fsync`, and no
replacement. The tool has no network, subprocess, simulator, port, or write-back
path. A failed parse or validation creates no success artifact and exits nonzero.

On Windows, `current-user-only` requires the current user as owner and permits
nonzero allow ACEs only for that user, LocalSystem, or built-in Administrators;
compound, object, and callback allow ACEs fail closed. Component and artifact
handles remain held and are revalidated for disk type, non-reparse status,
identity, and final path through exclusive creation, complete write, flush, and
handle-based ACL verification; failed owned creation cleans up by exact handle.

Every JSON input rejects duplicate keys, non-finite numbers, non-UTF-8, BOM,
unknown keys, missing keys, booleans where integers are required, and noncanonical
hash text. Canonical payload bytes are UTF-8 without BOM, ASCII-escaped, sorted
keys, separators `(',', ':')`, `allow_nan=False`, plus one LF. Each output is an
exact envelope with keys `schema`, `payload`, and `payload_sha256`, where the hash
covers canonical payload bytes. The CLI prints the final canonical envelope
SHA-256 and path; the printed hash is not embedded recursively.

`inspect-build` strictly parses UE PAK v11 footer and index bounds, requires
magic `0x5A6F12E1`, version `11`, an unencrypted index, and matching SHA-1 for
the primary, path-hash, and full-directory indices. It emits only exact source
identities, footer/index facts, compression names, mount point, entry count, and
exact directory-index paths matching the frozen candidate packages. It always
sets `admitted` to `false` and lists every R3 claim as missing. String scanning,
plausible floats, and decoded asset semantics are never emitted as proof.

`aigp-vq2-build-reference-candidate/1` has exact top-level object keys
`schema`, `build`, `parser`, `geometry`, `transform_chain`, `training_linkage`,
`visibility`, `uncertainty`, `independent_checks`, `rules`, and
`annotation_contract`. All nested objects reject unknown keys. A file identity
is exactly absolute `path: str`, `size_bytes: int >= 0`, and lowercase
`sha256: str`. An evidence identity is exactly nonempty `id`, `producer_id`,
and `method_id` strings plus lowercase `artifact_sha256`.

The remaining group types are frozen as follows:

- `build`: literal build `3385`, mode `Training`, launcher/payload/PAK file
  identities, exact typed footer/index fields, and sorted unique package paths;
- `parser`: implementation literal `aigp-vq2-stdlib-pak-parser/1`, parser-source
  file identity, exact interpreter hash, empty dependency array, and config hash;
- `geometry`: mesh/package strings, nonnegative active LOD, exact
  `render_not_collision_only: true`, convention/unit strings, nonempty uniquely
  identified vertex/edge/surface records with finite 3D coordinates, and finite
  nonnegative planarity/aspect/thickness/bevel uncertainty bounds;
- `transform_chain`: nonempty ordered links with IDs and frames, sixteen finite
  row-major matrix values, finite nonzero determinant, three finite positive
  singular values, literal right handedness and uniform scale, plus nonempty
  evidence identities; active actor overrides use the same typed link form;
- `training_linkage`: nonempty master-map, track-map, gate-blueprint, component,
  mesh, material, LOD, UDP-camera strings, exact `proved: true`, and nonempty
  evidence identities;
- `visibility`: nonempty model/surface/feature IDs, explicit front/back/bevel,
  clipping and occlusion policies, and finite nonnegative systematic bounds;
- `uncertainty`: nonempty conditional-pixel-model and shared-nuisance-ledger IDs,
  literal pixel-center convention, finite nonnegative render, material,
  antialias, JPEG, annotation, geometry, and transform bounds, and evidence;
- `independent_checks`: at least two passed check objects with different
  implementation and producer IDs plus exact input/output SHA-256 values;
- `rules`: the R2 clearance file identity and matching record ID; and
- `annotation_contract`: the observation schema literal, immutable producer,
  preprocessing, correspondence, rejection, covariance and shared-ledger hashes,
  plus an independent checker identity/hash.

Finite numbers exclude booleans. Arrays have exact dimensionality where stated;
IDs are nonempty, unique within scope, and references must resolve. Numeric
bounds are evidence inputs, never tool-invented defaults. Mechanism-only
floating-point validation uses frozen relative-uniformity and normalized-dot
roundoff tolerances of `1e-6`; both values are included in the parser config
hash, create no measurement or uncertainty evidence, and cannot replace an R3
bound. The validation report records per-group structural checks, exact input
hashes, `structurally_valid`, `admitted: false`, and
`independent_review_required: true`. R3 alone can admit a structurally valid
candidate through an independently reviewed task correction.

The only admissible reference branches remain those in the parent Package 2
contract:

- an organizer/build-specific render-side pixel-to-ray, camera-mount, and
  sensor-time oracle with verified semantics; or
- an independently specified static target with nondegenerate known geometry
  across the image, camera relative rotations independently constructed from
  admitted visual target poses, body relative rotations independently
  integrated from raw gyroscope samples, and separately bounded uncertainty.

The current local discovery of an Anduril square-gate package is a candidate
for the second branch, not an admitted reference. R3 admission requires a strict
private `aigp-vq2-build-reference-candidate/1` manifest binding all of:

1. exact launcher, shipping payload, PAK, PAK footer/index, parser source,
   parser dependency, configuration, and output SHA-256 identities;
2. the exact render mesh and active LOD, excluding collision-only geometry;
3. immutable visible rim/edge/vertex IDs, coordinate convention, units,
   coordinates, planarity, aspect, thickness/bevel, and uncertainty;
4. the complete `BP_gate` component/construction transform plus active map
   actor overrides, with determinant, singular values, handedness, and proof
   that every unknown scale is a uniform Euclidean similarity;
5. proof that build-3385 Training instantiates that exact map, blueprint,
   component, mesh, material, and LOD in the UDP JPEG view;
6. a uniquely visible fixed surface or a full 3D visibility model; a
   view-dependent front/back/beveled silhouette is not silently labeled as a
   planar inner corner;
7. bounded render-LOD, material/alpha, antialiasing, JPEG, pixel-center, and
   independent annotation systematics;
8. two independent extraction/transform checks that do not share the same
   implementation; and
9. the exact R2 record hash and independent proof that its asset and use scope
   covers every derived input in the candidate.

R3 also freezes a strict private
`aigp-vq2-target-reference-observation/1` label schema. Every observation
contains the admitted reference ID/hash; session, reset epoch, vision
generation, frame ID, camera source token, host receipt time, decoded-image
SHA-256 and dimensions; annotation-producer ID, code/model hash and version;
fixed rim/edge/vertex IDs with explicit correspondence; pixel-center
coordinates; visibility, clipping and occlusion state; and an enumerated
accept/reject reason. Each observation carries dense conditional pixel
covariance for its within-observation labeling noise, including within-feature
correlations, conditional on the separately represented shared nuisances.

A distinct shared-nuisance ledger assigns every annotation, reference, render,
compression, pixel-center, and transform systematic a stable parameter/set ID;
scope across frame, reset epoch, session, build, reference, and producer;
units; and either a joint covariance with its correlations or a conservative
non-Gaussian bounded set. It also freezes the mapping or Jacobian from each
nuisance into every affected observation. Cross-observation and cross-session
blocks are derived from that ledger and never default to zero. A shared term is
never copied into per-observation noise or averaged down as independent noise.

The annotation producer, preprocessing, correspondence convention, rejection
policy, and numeric uncertainty construction are immutable before collection.
The producer is independent of the fitter, runtime detector, IMU, and command
stream, and is blinded to those inputs. A second implementation checks the
labels and transform chain. Producer output and check output are separately
hashed. Candidate-generated labels, a labeler's self-review, or an uncertainty
value selected after inspecting fit residuals fails R3.

Unknown map world pose, target rigid pose, and uniform target scale may remain
per-view nuisance. Unknown nonuniform scale, shear, active linkage, visible
feature correspondence, or reference uncertainty fails R3.

The current exact discovery identities are leads only:

- PAK SHA-256
  `dae7ed0f4d51f7755814bf069cc9299b439ff874a2f77912a0c5678afaff299f`;
- shipping payload SHA-256
  `9064dd1547a30afea1e3fb87652cc8194c3f5af556be40629dc491bb4f681362`;
- PAK magic `0x5A6F12E1`, version `11`, unencrypted index, index offset
  `4573445197`, and index size `185458`;
- candidate render mesh
  `FlightSim/Content/Anduril-TrackEditor/Gates/`
  `SM_Gates_Anduril_Square_Combined`;
- candidate track `FlightSim/Content/levels/MAP_arsenal_track01`; and
- candidate master `FlightSim/Content/levelsMaster/MAP_arsenal_master`.

Names, PAK presence, visual squareness, plausible raw floats, or a map
dependency string do not satisfy R3. Encrypted config is not bypassed, guessed,
or assigned a default. Reverse-engineered native camera defaults are excluded
from the target-reference branch unless a later exact correction independently
proves their semantics and permission.

These discovery identities record work completed before this freeze. R1 does
not reread, extend, decode, or derive from the real PAK. They remain inert leads
until R2 independently permits the exact production parser invocation.

## Exact discovery excitation

The sole new runner literal is `calibration-excite`. It has no amplitude,
duration, thrust, stage, or waveform CLI override. The control period remains
`20 ms`; missed ticks are dropped and never replayed. Yaw is exact zero. Thrust
is fixed at the previously bounded sign-ID value `0.235`, never above the
already proved `0.32` envelope.

The immutable command segments are:

| Segment | Tick indices, inclusive | Count | Duration | Roll / pitch rate (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| dwell-0 | `0-29` | `30` | `0.60 s` | `(0.00, 0.00)` |
| roll-positive | `30-44` | `15` | `0.30 s` | `(+0.08, 0.00)` |
| dwell-1 | `45-53` | `9` | `0.18 s` | `(0.00, 0.00)` |
| roll-negative | `54-73` | `20` | `0.40 s` | `(-0.06, 0.00)` |
| dwell-2 | `74-85` | `12` | `0.24 s` | `(0.00, 0.00)` |
| pitch-positive | `86-105` | `20` | `0.40 s` | `(0.00, +0.07)` |
| dwell-3 | `106-115` | `10` | `0.20 s` | `(0.00, 0.00)` |
| pitch-negative | `116-133` | `18` | `0.36 s` | `(0.00, -0.08)` |
| dwell-4 | `134-149` | `16` | `0.32 s` | `(0.00, 0.00)` |
| coupled-1 | `150-164` | `15` | `0.30 s` | `(+0.06, +0.04)` |
| coupled-2 | `165-179` | `15` | `0.30 s` | `(-0.06, +0.04)` |
| coupled-3 | `180-194` | `15` | `0.30 s` | `(-0.06, -0.04)` |
| coupled-4 | `195-209` | `15` | `0.30 s` | `(+0.06, -0.04)` |
| dwell-final | `210-244` | `35` | `0.70 s` | `(0.00, 0.00)` |

After initial pre-send checks, the runner captures integer monotonic
`t0_ns = monotonic_ns()` and uses exact `T_ns = 20_000_000`. Tick `k` is
eligible for at most one send in the half-open slot
`[t0_ns + k*T_ns, t0_ns + (k+1)*T_ns)`. Equality with the lower bound is
eligible; equality with the upper bound is not. Before a release the scheduler
waits only until that release. On every wake and again after pre-send checks it
recomputes `k = floor((now_ns - t0_ns)/T_ns)`. All lower unsent indices are
recorded as skipped. If checks cross a slot boundary, that tick is skipped and
checks restart for the newly current tick; no stale check authorizes a send.

The table partitions the 245 indices without gaps or overlaps. Segment choice
uses the absolute planned index, never the count of successful sends. Elapsed
indices are never caught up, replayed, or sent back-to-back. Tick 244 has
release `t0_ns + 4_880_000_000`; the nominal plan is the half-open interval
`[t0_ns, t0_ns + 4_900_000_000)`. The stage completes after tick 244 or when
the current index exceeds 244. The powered hard expiry is
`t0_ns + 5_000_000_000` and is checked immediately before every send.

Actual raw gyro and independently referenced camera rotations, not commanded
rates or estimator orientation, are the eventual fit inputs. This acquisition
pilot has no rank, singular-value, conditioning, identifiability, or empirical
limit acceptance threshold. Those fields are emitted as `uncomputed`; only
descriptive motion and image-support summaries may be reported.

The existing command validation remains authoritative: roll/pitch command
rates at most `0.25 rad/s`, yaw zero, finite thrust, attitude/body-rate bounds,
fresh advancing streams, healthy estimator, fresh target, and gate index zero.
For this new stage every collision event aborts; there is no launch-pad contact
exception. Absolute roll and pitch excursion from the proved start attitude
may not exceed `0.05 rad` on either axis.

The first three-frame-confirmed target immediately before arming freezes
initial bbox area `A0`. Before every send, the current fresh target center must
remain inside the closed safety corridor `[0.10 W, 0.90 W]` by
`[0.10 H, 0.90 H]`; bbox width and height must each be at most `160 px`; and
bbox area must be at most `2.0 * A0`. These are safety-only abort limits, not
calibration or data-quality thresholds. Any target loss, limit violation,
gate-index change, reference-lineage failure, or deadline overrun aborts before
another send. This stage never enters crossing confirmation.

## Authoritative I0 executable-interface correction

This section is authoritative wherever the later historical live-command,
attempt, deadline, command-record, invalidation, report, split, or cleanup text
conflicts with it. It supersedes the former `60.0 s` child, `15.0 s` fallback,
`180.0 s` wrapper, ambiguous PID/start token, pre-launch Training assertion,
raw lease-owner token, generic `udpin:` bind, and underspecified generated/sent
record clauses. It does not change the single-attempt limit, the 245-tick plan,
the `5.0 s` powered hard expiry, command/watchdog/target bounds, simulation-only
data authority, or required disarm/reset/clean-epoch cleanup. Nothing before
I0/T0/L0 is back-claimed as implementation or live evidence.

Every new JSON schema below rejects duplicate, missing, or unknown keys; BOM or
non-UTF-8 input; nonfinite numbers; booleans where integers are required;
noncanonical hashes; and relative, reparse-traversing, cross-root, or aliased
paths. Generated bytes are UTF-8
`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)` followed by one LF. A file SHA-256 covers those complete
bytes; an object SHA-256 covers the same canonical encoding without the LF.
UTC is exactly `YYYY-MM-DDTHH:MM:SS.ffffffZ`; `*_monotonic_ns` is an exact
nonnegative integer on `host-perf-counter`; a Git commit is 40 lowercase hex;
and a SHA-256 is 64 lowercase hex. Exact integers reject `bool`.
In every hash formula, `00` between concatenation operators denotes one binary
octet `0x00`, never the two ASCII characters `"00"`.

Named nested shapes are also exact. `IdentityRefV1` is `{path,sha256}`.
`ProcessIdentityV1` is `{pid,creation_filetime_100ns,windows_session_id,image_path,image_sha256,argv_sha256}`.
`ArtifactRefV1` is `{name,path,size_bytes,sha256}`. `PhaseDeadlineV1` is
`{phase,started_monotonic_ns,duration_ns,parent_deadline_monotonic_ns,deadline_monotonic_ns}`.
`ArtifactTimingV1` is that exact five-key shape plus
`prepared_monotonic_ns`; preparation is at or after start and strictly before
deadline, and create/write/flush/readback still must complete before the same
deadline. A wrapper-ledger phase-end row after readback provides the external
completion occurrence for preterminal artifacts. `TerminalPublicationTimingV1`
has the same exact six keys; for complete/invalid terminal files there is no
later ledger row, so the valid create-new terminal bytes and the writer's
required pre-return flush/readback are themselves the terminal proof.
`TerminalCleanupV1` is
`{child_certificate_sha256,fallback_used,fallback_certificate_sha256,child_exit,
fallback,processes,transport,ports,lease,simulator_topology,
simulator_responsive,scheduled_task}`. On a complete terminal, the child hash is
nonnull; the fallback boolean and nullable hash agree; and the exact enum values
are respectively `proved`, `not_required|proved`, `exited`, `closed`, `free`,
`released`, `unchanged`, `yes`, and `absent`. Outside a complete terminal, the
same summary shape may appear only inside a completed capture seal and use the wider
invalidation vocabulary frozen below. `InvalidCleanupStateV1` is the distinct
exact `{child_exit,fallback,ports,lease,processes,transport,scheduled_task,
simulator_topology,simulator_responsive}` shape used by attempt-invalid.
`OutboundAuditV1` is exact
`{timesync,gcs_heartbeat,sim_reset,arm,disarm,attitude_target,
position_target,other_command,receipt_count,receipt_returned,receipt_raised,
receipt_dropped,receipt_buffered}` with nonnegative exact integers.
Arrays described as sorted use ordinal UTF-8 ordering of their declared key,
are unique, and reject a different order. `argv_sha256` hashes canonical JSON of
the exact argument string array. Host boot identity is
`SHA256(UTF8(uppercase MachineGuid registry string)||00||uint64-little-endian NtQuerySystemInformation
boot FILETIME)`; the raw MachineGuid is never stored.

The implementation inventory is
`aigp-vq2-powered-implementation-inventory/1` with exact keys
`schema,commit,tree,entries`; `tree` is the integrated Git tree ID and `entries`
is every regular blob from exact `git ls-tree -r --full-tree`, sorted by path,
as exact `{path,size_bytes,sha256}`. Candidate `code_sha256` hashes the canonical
`{commit,tree,entries}` object. The environment inventory is
`aigp-vq2-powered-environment-inventory/1` with exact
`schema,created_at_utc,variables`; variables is every Windows environment entry
as exact `{name,defined,value_sha256}`, with uppercase name, literal
`defined=true`, and SHA-256 of its UTF-8 value, sorted case-insensitively with
duplicate-name rejection. The import inventory is
`aigp-vq2-powered-import-inventory/1` with exact
`schema,python_sha256,seeds,entries`; `seeds` is the exact sorted array
`scripts.aigp_vq2_powered_attempt,scripts.aigp_vq2_powered_calibration_analysis,
scripts.aigp_vq2_powered_calibration_probe,scripts.aigp_vq2_powered_cleanup,
scripts.aigp_vq2_powered_runtime,scripts.aigp_vq2_run`. In a fresh exact
`-E -s -B` audit interpreter at the exact frozen cwd, those seeds and the
code-owned sorted `POWERED_EAGER_IMPORT_MODULES` list are imported in array
order. `entries` is every nonnull resulting `sys.modules` entry after imports
return;
each entry is exact `{module,origin,size_bytes,sha256,root_class,
namespace_roots}` and is sorted uniquely by module. Origin and byte SHA/size are
nonnull for file-backed modules and null only for built-in, frozen, or namespace
packages; root class is `candidate|venv|stdlib|builtin|frozen|namespace`.
Namespace packages alone carry a nonempty sorted exact absolute
`namespace_roots` array; every root must classify wholly under one frozen
candidate/venv/stdlib root, with mixed or unclassified roots rejected. Every
other entry carries the empty array. L0 runs the isolated probe twice and
requires byte-semantic equality of `{python_sha256,seeds,entries}` before
freezing it.

The wrapper never regenerates an inventory file or its timestamp. Before
attempt and again before child release it re-derives and compares only the
deterministic semantic payloads: implementation `{commit,tree,entries}`,
environment `variables`, and imports `{python_sha256,seeds,entries}`. The original
environment `created_at_utc` remains file provenance and is excluded from that
semantic comparison. Import comparison also re-resolves every inventoried
origin/hash and rejects any loaded candidate- or venv-root module absent from
entries; additional standard-library bootstrap modules are allowed only from
the frozen stdlib root. Any payload or frozen file-byte hash drift invalidates.

### Frozen excitation-plan value

The runner and evidence validator use this exact value. It is code-owned, not
configurable and not CLI-overridable. Its canonical compact/sorted JSON SHA-256
is `6ca6900c1977ba789920fad87aee67dca36f0911de255aa4cf29c30bc8809bce`:

```json
{
  "schema": "aigp-vq2-calibration-excitation-plan/1",
  "plan_id": "vq2-build3385-training-f00-excite-v1",
  "stage": "calibration-excite",
  "control_period_ns": 20000000,
  "tick_count": 245,
  "nominal_end_offset_ns": 4900000000,
  "powered_hard_expiry_offset_ns": 5000000000,
  "command": {"thrust": 0.235, "yaw_rate_rad_s": 0.0},
  "segments": [
    {"segment_id": "dwell-0", "first_tick": 0, "last_tick": 29, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
    {"segment_id": "roll-positive", "first_tick": 30, "last_tick": 44, "roll_rate_rad_s": 0.08, "pitch_rate_rad_s": 0.0},
    {"segment_id": "dwell-1", "first_tick": 45, "last_tick": 53, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
    {"segment_id": "roll-negative", "first_tick": 54, "last_tick": 73, "roll_rate_rad_s": -0.06, "pitch_rate_rad_s": 0.0},
    {"segment_id": "dwell-2", "first_tick": 74, "last_tick": 85, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
    {"segment_id": "pitch-positive", "first_tick": 86, "last_tick": 105, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.07},
    {"segment_id": "dwell-3", "first_tick": 106, "last_tick": 115, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
    {"segment_id": "pitch-negative", "first_tick": 116, "last_tick": 133, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": -0.08},
    {"segment_id": "dwell-4", "first_tick": 134, "last_tick": 149, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
    {"segment_id": "coupled-1", "first_tick": 150, "last_tick": 164, "roll_rate_rad_s": 0.06, "pitch_rate_rad_s": 0.04},
    {"segment_id": "coupled-2", "first_tick": 165, "last_tick": 179, "roll_rate_rad_s": -0.06, "pitch_rate_rad_s": 0.04},
    {"segment_id": "coupled-3", "first_tick": 180, "last_tick": 194, "roll_rate_rad_s": -0.06, "pitch_rate_rad_s": -0.04},
    {"segment_id": "coupled-4", "first_tick": 195, "last_tick": 209, "roll_rate_rad_s": 0.06, "pitch_rate_rad_s": -0.04},
    {"segment_id": "dwell-final", "first_tick": 210, "last_tick": 244, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0}
  ]
}
```

The validator derives counts and durations and rejects a gap, overlap,
reordering, changed literal, alternate hash, or any mismatch with the table
above. Every excitation command/tick record binds both plan ID and plan hash;
cleanup scope intentionally carries null plan identity.

### Live freeze, sole operator command, and paths

L0 creates, independently reviews, and byte-hash pins exactly one private
`aigp-vq2-powered-calibration-live-freeze/1`. Its exact top-level keys are
`schema`, `task_id`, `freeze_id`, `candidate`, `session`, `inputs`, `runtime`,
`simulator`, `transport`, `execution`, `paths`, and `deadline_durations_ns`:

- `task_id` is `vq2-package2-powered-calibration-pilot`; `freeze_id` is
  `vq2-package2-powered-calibration-f00-a01-live-freeze`.
- `candidate` has exact keys `commit`, `code_sha256`, `live_worktree`,
  `detached_head_required`, `clean_tracked_untracked_ignored_required`, and
  `implementation_inventory`. The two booleans are true; the inventory has
  exact absolute `path` and `sha256`; the worktree is the frozen detached live
  worktree; and `code_sha256` is the canonical inventory-object hash.
- `session` is exactly `session_id=F00`, `attempt_id=F00-A01`,
  `attempt_limit=1`, and `split=discovery_fit`.
- `inputs` contains exact identity objects for `target_config`,
  `capture_authorization`, and `excitation_plan`. The first two have exact
  `schema`, absolute `path`, and byte `sha256`; the plan additionally has exact
  `plan_id`. All match the admitted P1 identities and frozen plan.
- `runtime` has exact keys `python,powershell,development_test_lock,
  environment_inventory,import_inventory`. Python is exact
  `{path,implementation,version,sha256}` with `implementation=CPython` and
  `version=3.12.2`; PowerShell is exact `{path,product_version,sha256}` with a
  nonempty version; and the other three are exact `IdentityRefV1`. No user site,
  alternate import root, or mutable candidate-local executable is accepted.
- `simulator` contains exact `build=3385`, `mode=Training`, identity objects for
  `launcher_script`, `launcher`, and `payload`; each is exact `IdentityRefV1`.
  It also contains topology literal
  `one_launcher_parent_retained_one_payload_child`, and `mode_evidence` literal
  `post_topology_local_interactive_attestation`.
- `transport` has exact `mavlink_bind`, `camera_bind`, `peer_policy`,
  `allowed_outbound_categories`, and `unknown_category_policy`. The binds are
  `{host:"127.0.0.1",port:14550,socket_policy:"ipv4-exclusive-address-use"}`
  and `{host:"0.0.0.0",port:5600,socket_policy:"ipv4-exclusive-address-use"}`;
  peer policy is `freeze_first_valid_build3385_source`; the sorted category
  array is `arm,attitude_target,disarm,gcs_heartbeat,sim_reset,timesync`; and
  unknown policy is `invalidate`. This is not a pymavlink `udpin:` URL.
- `execution` has exact `wrapper_cwd`, `security_environment`, `launcher_cwd`,
  `launcher_argv`, `launcher_environment_sha256`, `child_cwd`, and
  `cleanup_cwd`. Every cwd is the exact detached live worktree.
  `security_environment` is exact `PYTHONNOUSERSITE=1`,
  `PYTHONDONTWRITEBYTECODE=1`, and sorted forbidden-defined array
  `PYTHONHOME,PYTHONPATH,PYTHONSTARTUP`; the complete environment must hash to
  the runtime inventory. Launcher argv is the exact string array
  `<absolute-powershell>,-NoLogo,-NoProfile,-NonInteractive,-ExecutionPolicy,Bypass,-File,<absolute-launch_sim.ps1>,-SimulatorPath,C:\Users\John\AIGP\AIGP_3385\FlightSim.exe,-TaskName,AIGP-P2-F00-A01-Launch,-StartupTimeoutSeconds,25`.
  It contains no `RunAsUser`, environment-derived simulator path, or extra
  option. The wrapper proves the exact task name absent before launch, after
  launcher return, before child creation, and in the final process proof.
  The script may spend its 25-second parameter once on its launch mutex and
  once on startup polling; the independent 60-second outer launcher deadline
  reserves the remaining time for its `finally` task deletion and exit.
- `deadline_durations_ns` is the exact object frozen below.

`paths` has exactly these absolute-path keys: `evidence_root`, `live_freeze`,
`attempt_dir`, `attempt_envelope`, `training_attestation`, `process_proof`,
`process_final_proof`, `child_authority`, `cleanup_authority`,
`child_cleanup_certificate`, `fallback_cleanup_certificate`, `lease_directory`,
`lease_final`, `wrapper_ledger_directory`, `wrapper_lifecycle`, `runner_stdout`,
`runner_stderr`, `legacy_record`,
`replay_bundle`, `bundle_verification`, `capture_seal`, `analysis_report`, `split_claim`,
`split_registry`, `attempt_complete`, `attempt_invalid`, `cleanup_stdout`,
`cleanup_stderr`, and `live_poison`. The root is
`C:\Users\John\aigp-evidence\2026-07-20-package2-powered-calibration-pilot`;
the freeze is `<root>\live-freeze-F00-A01.json`; the attempt directory is
`<root>\F00-A01`; the registry is
`<root>\split-registry\registry-000001.json`; and poison is
`<root>\live-poison.json`. Attempt basenames respectively are `attempt.json`,
`training-attestation.json`, `process-proof.json`, `process-final-proof.json`,
`child-authority.json`, `cleanup-authority.json`,
`child-cleanup-certificate.json`, `fallback-cleanup-certificate.json`, `lease`,
`live-lease.json`, `wrapper-ledger`, `wrapper-lifecycle.json`,
`runner-stdout.json`, `runner-stderr.txt`, `session.jsonl.gz`,
`session.vq2replay`, `bundle-verification.json`, `capture-seal.json`,
`analysis.json`, `split-claim.json`,
`attempt-complete.json`, `attempt-invalid.json`, `cleanup-stdout.json`, and
`cleanup-stderr.txt`.

From `candidate.live_worktree` as the exact current directory, the only
operator argv is:

```text
<python> -E -s -B -m scripts.aigp_vq2_powered_calibration_probe --live-freeze <absolute-live-freeze> --live-freeze-sha256 <reviewed-64hex> --expected-commit <reviewed-40hex>
```

The parser uses `allow_abbrev=False`; all three options are required; the
commit must equal both freeze and detached `HEAD`; and no stage, session,
output, target, capture, rate, thrust, duration, or safety selector exists.
Before attempt creation the loaded module path, cwd/final handles, Git worktree,
environment/import inventories, and every repository import must match the
freeze. The wrapper derives every later path and command from it. Invocation
from any other cwd or import origin is outside the sole command and fails before
attempt, simulator, or port contact.

### Attempt consumption, process identity, capabilities, and Training fact

Offline validation does not consume the attempt. Atomic create-new creation of
the protected `<root>\F00-A01` directory does. Its current-user-only DACL is
present in the `CreateDirectoryW` operation itself, not applied after a
world-readable creation window; every ancestor, final handle, reparse bit,
volume, owner, and DACL is locked and revalidated. Any pre-existing F00-A01
directory/envelope, any root poison, or any prior attempt without exactly one
valid terminal record blocks before simulator or port contact. There is no
retry or replacement attempt.

Before lease acquisition, the wrapper creates the inherited anonymous pipes,
generates three independent 32-byte CSPRNG values for lease owner, child, and
cleanup, and create-new flushes
`aigp-vq2-powered-calibration-attempt/1`. Its exact top keys are `schema`,
`context`, `context_sha256`, and `capabilities`. `context` has exactly:

`task_id,session_id,attempt_id,created_at_utc,host,live_freeze,candidate_commit,target_config,capture_authorization,excitation_plan,wrapper_process,paths,child_argv,cleanup_argv,deadline_durations_ns,wrapper_absolute_deadlines,prepublication_timing`.

`host` is exact `host_clock_id=host-perf-counter`, `host_boot_id_sha256`, and
positive integer `qpc_frequency_hz`. Each identity is exact absolute `path` and
`sha256`, with plan ID added for excitation. `wrapper_process` is exact
`pid`, `creation_filetime_100ns`, `windows_session_id`, `image_path`,
`image_sha256`, and `argv_sha256`; it is re-proved through a retained process
handle. `paths` and duration deadlines equal the freeze;
`wrapper_absolute_deadlines` uses the exact three-key shape defined below.
`prepublication_timing` is exact
`{wrapper_started_monotonic_ns,offline_precheck,attempt_publish}`. Offline
precheck is exact
`{phase,started_monotonic_ns,duration_ns,parent_deadline_monotonic_ns,
deadline_monotonic_ns,completed_monotonic_ns,outcome}` with phase
`offline_precheck` and outcome `completed`; attempt publish is exact
`PhaseDeadlineV1` with phase `attempt_publish`. Publication
end cannot self-reference and is the first wrapper-ledger event after attempt
readback.
Child/cleanup argv are exact
nonempty string arrays containing actual inherited decimal handle values but no
secret. `context_sha256` hashes the canonical context.

`capabilities` is exact `algorithm`, `lease_owner_sha256`, `child_sha256`, and
`cleanup_sha256`, with algorithm
`sha256-domain-separated-context-v1`. For domains
`aigp-vq2-lease-owner/1`, `aigp-vq2-powered-child/1`, and
`aigp-vq2-powered-cleanup/1`, the digest is
`SHA256(UTF8(domain)||00||bytes.fromhex(context_sha256)||00||secret32)`.
Raw capability/owner bytes never enter a file, argv, environment, log, replay,
exception, or lease record. Child/cleanup obtains exactly one framed 32-byte
secret. The frame is exactly 36 bytes:
`uint32-little-endian(32) || secret32`, followed by pipe EOF. These are ordinary
anonymous `CreatePipe` handles and never use overlapped I/O. The normally
started child/fallback blocks only in this poll; the wrapper writes all 36 bytes
and closes its sole write handle as the release gate. The reader polls
`PeekNamedPipe` in at most 50 ms slices while
checking its parent handle and three-second capability-release deadline. It
rejects an available-byte count greater than 36, issues exactly one synchronous
36-byte `ReadFile` only after all 36 bytes are available, then enters a bounded
EOF-confirmation poll. A successful post-read peek with zero bytes means writer
closure is not yet observed, so it rechecks parent/deadline and retries within
50 ms; `ERROR_BROKEN_PIPE` proves EOF. Any post-read available byte, other
error, EOF before 36 bytes, short/extra data, parent death, or deadline exits
before live imports/socket contact. It validates the bound
hash, closes the pipe, and consumes the secret once. The Windows explicit handle list permits only
the role capability read pipe, read-only parent-process handle, create-new
stdout and stderr file handles, and read-only `NUL` stdin handle. All five handles
are independently validated; every other inheritable handle fails. The wrapper
retains the sole capability-pipe write end and no output handle can receive raw
capability bytes.

Wrapper phase evidence is an append-only protected `wrapper-ledger` using the
same pending/flush/readback/atomic-no-replace publication rule. Each contiguous
`event-%06d.json` is `aigp-vq2-powered-wrapper-event/1` with exact keys
`schema,task_id,session_id,attempt_id,event_sequence,predecessor_sha256,event,phase,observed_monotonic_ns,duration_ns,parent_deadline_monotonic_ns,deadline_monotonic_ns,outcome,reason_code,artifacts`.
Event sequences are contiguous from zero; predecessor is null only at zero and
otherwise equals the complete-file SHA-256 of the prior row.
Event is `phase_start|phase_end`; start has null outcome/reason, end is
`completed|failed`. Start and completed end have null reason; failed end has
exactly one reason from the invalidation vocabulary below. `artifacts` is a
sorted unique `ArtifactRefV1` array; it is empty for starts and failures and
contains only regular files that this completed phase create-new published and
readback-verified. Every phase has exactly one
ordered pair except the first `attempt_publish` end, whose start/deadline is in
the attempt. Phase is drawn only from
`attempt_publish,lease_acquire,launcher_return,
topology_and_training_attestation,prechild_identity_and_ports,child_spawn,
child_supervision,child_exit_proof,fallback_spawn,fallback_supervision,
postcheck_identity_process_ports,lease_release_and_verify,bundle_verify,
capture_seal,analysis,split_publish,terminal_ready,poison_publish,
invalid_ready`. `child_supervision` and `fallback_supervision` use the frozen
child/fallback total durations; `terminal_ready` and `invalid_ready` use
`terminal_publish`; all other names map to their same-named frozen duration.
The normal order is the list order with the fallback pair omitted unless used;
failure may mark a phase failed but safety-resolution phases retain that order.

Ledger scope runs lease acquisition, every live phase, lease
release, and the postrelease `bundle_verify`, `capture_seal`, `analysis`, and
`split_publish` phases. On the successful branch it ends with a
`terminal_ready` pair under the `terminal_publish` deadline after every
preterminal artifact byte/hash is prepared. On failure it records the failed
phase and, iff the derived poison predicate requires poison and the wrapper
remains capable, runs the bounded `poison_publish` phase. When
`poison.required=false`, that phase is omitted and `invalid_ready` follows
directly. Failed or partial required-poison publication remains fail-closed.
Prepublication timing is explicitly exempt
from the before-wait ledger rule because no attempt directory existed.
Completed postrelease publication rows contain exact artifacts as follows:
`bundle_verify=[bundle_verification]`, `capture_seal=[capture_seal]`, and
`split_publish=[analysis_report,split_claim,split_registry]`, sorted by name.
Completed `lease_release_and_verify` contains `[lease_final]`; completed
`poison_publish` contains `[live_poison]`; `terminal_ready`/`invalid_ready`
contain the empty array because their terminal file is written only after the
lifecycle is finalized. Every wrapper row not explicitly mapped in this
paragraph has an empty artifacts array; process/authority/result/certificate
files are bound later by seal/terminal, not implicitly claimed by this ledger.

After the last ready/failed event and before normal terminal publication,
`wrapper-lifecycle.json` is
`aigp-vq2-powered-wrapper-lifecycle/1` with exact keys
`schema,task_id,session_id,attempt_id,records,final_sequence,final_record_sha256,live_contact_deadline_monotonic_ns,total_deadline_monotonic_ns`;
`records` is the contiguous ordered array of exact
`{event_sequence,path,sha256,event,phase,observed_monotonic_ns,outcome,
reason_code,artifacts}` copied from and byte-hash binding every ledger row. Final sequence
and hash match its last row. The seal, claim, registry, and report deliberately
do not refer to this not-yet-final artifact. The complete terminal binds the
valid lifecycle hash; an invalid terminal records its lifecycle state and hash.
Missing/partial lifecycle evidence invalidates and is never synthesized from
logs. Finalization does not refresh the terminal absolute deadline. If
lifecycle finalization or the complete-terminal create/write/flush/readback
fails after `terminal_ready`, the wrapper does not alter the ledger or lifecycle;
it publishes required poison and a distinct attempt-invalid record under the
frozen `invalid_terminal_publish` reserve deadline. The invalid record binds
the absent/partial/valid lifecycle and absent/partial complete-terminal states.

The exact internal child argv is
`<python> -E -s -B -m scripts.aigp_vq2_run --stage calibration-excite
--powered-attempt-envelope <attempt.json> --wrapper-process <PID:CREATION_FILETIME_100NS>
--powered-process-authority <child-authority.json>
--attempt-capability-handle <decimal-read-handle> --parent-liveness-handle
<decimal-process-handle> --record <session.jsonl.gz> --replay-bundle
<session.vq2replay> --cleanup-certificate <child-cleanup-certificate.json>
--recording-approved`. The exact fallback argv is
`<python> -E -s -B -m scripts.aigp_vq2_powered_cleanup
--powered-attempt-envelope <attempt.json> --wrapper-process
<PID:CREATION_FILETIME_100NS> --powered-process-authority
<cleanup-authority.json> --cleanup-capability-handle <decimal-read-handle>
--parent-liveness-handle <decimal-process-handle> --cleanup-certificate
<fallback-cleanup-certificate.json>`.
These parsers use `allow_abbrev=False`; every listed option is mandatory and
there are no extra options. PID and FILETIME are positive base-10 integers
separated by one colon; handles are positive base-10 integers and must be in
the explicit inherited-handle list. Direct/manual invocation, a second stage,
missing capability EOF, wrong parent, or an option/path mismatch fails before
socket or simulator contact.

Child and fallback start normally with the explicit handle list, but their first
bootstrap operation—before importing/opening a live transport or creating any
subprocess—is that bounded exact capability-frame poll. The wrapper assigns the
blocked process to a non-breakaway job, retains its identity, create-new flushes
and readback-verifies process authority, then writes the framed secret and
closes its sole write end as the release gate. Wrapper death before release
causes short read/EOF and exit without socket/simulator contact, avoiding both
the job-assignment escape window and a suspended orphan. The job has neither
breakaway flag and `kill_on_close=false`; it therefore survives later wrapper
death long enough for bounded child cleanup but cannot spawn an escaped sender.
After a timeout the live wrapper may use `TerminateJobObject` only to remove
residue; termination is never cleanup proof.

Each process authority is exact
`aigp-vq2-powered-process-authority/1` with keys
`schema,task_id,session_id,attempt_id,role,created_at_utc,created_monotonic_ns,attempt_envelope_sha256,attempt_context_sha256,live_freeze_sha256,wrapper_process,process,parent_handle,capability_sha256,lease_record_sha256,training_attestation_sha256,simulator_process_proof_sha256,argv_sha256,job,absolute_deadlines`.
Role is `powered_child` or `cleanup_fallback`. Parent handle is exact numeric
`{value,process,access,inherited}`; value is positive, process is the wrapper
`ProcessIdentityV1`, access is literal
`synchronize_query_limited_information`, and inherited is true. `job` is exact
`{handle_value,assigned_before_capability_release,breakaway_allowed,
silent_breakaway_allowed,kill_on_close,process_in_job}`; handle is positive,
assignment/process-in-job are true, and the three policy booleans are
respectively false, false, and false. Process/argv/capability/lease/Training/simulator identities
must match retained handles, attempt, and current ledger generation.

For powered child, `absolute_deadlines` has exact QPC integers `anchor`, `total`,
`prepower`, `powered`, `cleanup`, `replay_close`, and `exit`, equal anchor +
110/52/57/72/107/110 seconds. For fallback it has exact `anchor`, `total`, and `exit`, both latter
equal anchor + 25 seconds. The attempt context additionally has exact
`wrapper_absolute_deadlines={started_monotonic_ns,live_contact_deadline_monotonic_ns,total_deadline_monotonic_ns}`.
Child phase tokens, in order when reached, are exactly
`connect,preflight,reset_epoch,normalize_disarmed,countdown_go,arm,
powered_stage,cleanup,replay_close,finalize`; they map respectively to the
same-named frozen durations (`child_` prefix in the duration object) and the
absolute caps above. Optional `parent_death_lease_takeover` uses the frozen
one-second duration and occurs either immediately before/within cleanup, or at
any parent death after a proved cleanup certificate while the child remains
alive, including after replay close and before exit. The late form sends
nothing, validates existing cleanup proof, releases/finalizes the abandoned
lease, and continues bounded close/finalize; it never repeats zero/disarm/reset.
If replay is already closed, its frozen phase record appears only in the process
result and no enqueue is attempted. Fallback phase tokens are exactly
`connect,disarm,reset_and_epoch,finalize` plus optional
`parent_death_lease_takeover` at the detection point before its next send; the
takeover preserves completed phases and never repeats disarm or reset. They map
to the frozen `fallback_` durations and total cap. A failed phase may jump only to required cleanup/
replay/finalize phases; no started phase is omitted or reordered.
Every child phase through and including `replay_close` emits nested replay observation
`aigp-vq2-phase-deadline/1` under literal event `calibration_phase_deadline` with
exact keys `schema,attempt_id,producer_role,phase,event_sequence,started_monotonic_ns,duration_ns,parent_deadline_monotonic_ns,deadline_monotonic_ns` before its first wait. Wrapper phase deadlines are recorded in its wrapper ledger;
fallback phase deadlines are frozen in memory before waiting and recorded in
its final or failed cleanup certificate and process result. `child_finalize`
and child exit occur after replay close and are recorded only in the final
process result plus wrapper exit proof; enqueue after replay close is forbidden.
All deadlines equal the frozen min calculation and never refresh. A fallback
crash before certificate publication leaves cleanup unproved and poisoned; no
later artifact reconstructs its phase observations.

A process/topology occurrence is immutable
`aigp-vq2-simulator-process-proof/1` with exact keys
`schema,task_id,session_id,attempt_id,phase,observed_at_utc,observed_monotonic_ns,host_clock_id,wrapper_process,launch,launcher_process,payload_process,window,build,topology,scheduled_task,ports,responsive`.
Phase is `prechild` or `postchild`; every process uses retained-handle identity;
`launch` is exact
`{disposition,observed_before_launch_monotonic_ns,
launcher_return_monotonic_ns,launcher_exit_code,prelaunch_launcher_process,
prelaunch_payload_process}`. Both times are QPC nanoseconds with before-launch
strictly less than launcher-return, return is no later than this proof's
observation, and exit code is exact zero. Disposition is
`absent_before_launcher_current_after|preexisting_exact_topology`. For
`absent_before_launcher_current_after`, both prelaunch values are null after an
exact image/path process enumeration at the recorded before-launch occurrence;
the wrapper acquires both accepted retained process handles only after the
ordered launcher-return occurrence. This proves absent-before/current-after and
does not claim that a QPC value is comparable with a process FILETIME or that
the script, rather than another actor, created the processes. For
`preexisting_exact_topology`, both prelaunch values are nonnull retained-handle
identities byte-equal to the accepted current launcher/payload identities; the
launcher script's documented already-running return grants no authority until
parentage, hashes, window, build, responsiveness, and every other topology fact
passes the same proof. Incomplete, replaced, or mixed preexisting topology
invalidates. Postchild repeats the immutable launch object exactly;
window is exact `{hwnd,owner_pid,visible,unminimized,responsive}` with positive
handle/PID and the three booleans true; build is 3385; topology is the frozen literal;
`scheduled_task` is exact
`{name:"AIGP-P2-F00-A01-Launch",observations:[...]}`;
each ordered observation is exact `phase`, `observed_monotonic_ns`,
`query_exit_code`, and `absent=true`. Prechild contains exactly
`before_launch,after_launcher_return,before_child`; postchild repeats those
byte-identically and adds `after_child_or_fallback`;
`ports` has exact keys `owner_table_observations`, `active_owner_observations`,
`exclusive_probes`, and `status=free`.
An owner observation is exact `observed_monotonic_ns` plus sorted unique PID
arrays `ipv4_14550`, `ipv6_14550`, `ipv4_5600`, and `ipv6_5600`, all empty.
An exclusive probe is exact `host`, `port`, `started_monotonic_ns`,
`ended_monotonic_ns`, and `result=bound_and_closed`. Prechild has two ordered
owner observations, empty active-owner array, followed by one probe per port.
Postchild carries the wrapper's ordered active observations, each exact
`{observed_monotonic_ns,port,role,pid,creation_filetime_100ns}` matching the child/
fallback certificate, then has two free owner
observations, one probe per port, then a third owner observation. Responsive is
true. Prechild is `process-proof.json`; the independently
re-proved postchild occurrence is `process-final-proof.json`. Training binds
the prechild hash; seal/report/terminal bind both and require identical
launcher/payload/window identities and topology.

A pre-launch `--training-mode-attested` switch is forbidden. After exact
topology/window/process proof and before child creation, the wrapper requires an
attached local console and prints `Visually verify the proved FlightSim window
is in Training mode, then enter TRAINING <32-lowerhex challenge>`. It accepts
only that exact response within the topology deadline. It create-new flushes
`aigp-vq2-training-mode-attestation/1` with exact keys
`schema,task_id,session_id,attempt_id,attested_at_utc,attested_monotonic_ns,host_clock_id,mode,method,challenge_sha256,wrapper_process,simulator_process_proof_sha256`.
Mode is `Training`; method is `post_topology_visual_training_check_challenge`;
only the challenge hash is stored. Redirected, missing, wrong, or late input
invalidates before child creation. The response attests that the operator
visually checked the accepted current GUI, whether newly created or preexisting;
it is not a mere challenge echo and not another permission request.

### Exact receiver, collision, camera, and outbound-call lineage

`MavlinkIngressV1` remains a timing token. I0 adds immutable defensive-copy
envelopes in `competition/vq2_capture.py` and constructs each envelope under the
adapter state lock from the one QPC read used for its ingress:

- `aigp-vq2-received-heartbeat/1` has exact keys `schema`, `ingress`, and
  `heartbeat`; heartbeat is exact uint8 `base_mode` and uint32 `custom_mode`.
  Ingress type is `HEARTBEAT` and both source-time fields are null.
- `aigp-vq2-received-race-status/1` has exact keys `schema`, `ingress`, and
  `race_status`; payload is uint64 `sim_boot_time_ms`, int64
  `race_start_boot_time_ms`, int64 `race_finish_time_ns`, uint32
  `active_gate_index`, and int64 `last_gate_race_time`. Ingress type is
  `RACE_STATUS`, source unit is `ms`, and source value equals
  `sim_boot_time_ms`.
- `aigp-vq2-received-actuator-output-status/1` has exact keys `schema`,
  `ingress`, and `actuator_output_status`; payload is uint64 `time_usec`,
  uint32 `active`, and exactly 32 finite JSON numbers in `actuator`. Ingress
  type is `ACTUATOR_OUTPUT_STATUS`, source unit is `us`, and source value equals
  `time_usec`. A nonfinite or wrong-length actuator occurrence invalidates.

`ReceivedIMUSampleV1` remains unchanged and authoritative. Source clocks remain
opaque: camera ns, IMU/actuator us, race ms, and QPC are never directly
subtracted. Every nested ingress retains exact stream ID, reset generation,
global-within-generation receive sequence, `host-perf-counter`, and receipt ns.
The adapter stores envelopes rather than mutable MAVLink objects, provides
immutable latest heartbeat/race envelope accessors, and adds
`drain_received_observations()` returning all four types in ingress order.
Legacy drain APIs project those same single queues to `MavlinkIngressV1`; they
do not duplicate occurrences. `MavlinkIngressStats` remains schema-compatible,
and accepted evidence requires all ingress drop counts zero and queues empty.
The runner emits both the legacy ingress row and the strict payload event for
each occurrence. Heartbeat freshness and race tokens use one exact envelope,
not a side-state snapshot paired later.

Calibration reset uses one additive
`reset_calibration_with_boundary(persist_boundary)` adapter API; legacy reset
behavior is unchanged for every other stage. Under `_state_lock` it snapshots and removes
the complete old-generation received-envelope and collision queues plus both
diagnostic structs, advances/clears generation state, and returns immutable
`aigp-vq2-calibration-reset-boundary/1` before the guarded reset send. Exact
keys are `schema,old_generation,new_generation,boundary_monotonic_ns,observations,collisions,ingress_stats,collision_stats`; observations preserve ingress order,
collisions preserve drain order, and generations differ by one. Outside the
lock but before any reset packet, the API synchronously calls the supplied
persist callback. In prepower mode callback failure sends nothing and aborts.
Once cleanup is mandatory, callback failure latches evidence invalid and keeps
the batch in the cleanup certificate's in-memory construction, but cannot
suppress guarded reset/epoch proof or re-enable production. It then calls the guarded
low-level reset send directly, without invoking legacy `reset()` or advancing a
second time, and returns/preserves the boundary even when send raises. Runner
records the entire batch under literal event `calibration_reset_boundary`, marking each
collision `reset_boundary_discard`, before any later occurrence is consumed.
The same locked transition exactly once clears telemetry-ready, `_have_imu`,
last IMU/race/actuator freshness times, latest telemetry, race status, and
actuator output; heartbeat state intentionally remains for request/newer-
heartbeat comparison. No stale side state survives the new generation.
Post-boundary arrivals enter only the new queues; pre-reset-clock samples that
race after the boundary remain recorded and cannot satisfy rollback proof.
There is no drain/clear race and no silent IMU clear in calibration mode.
The calibration-only reset-baseline and reset-proof loops continuously drain
these immutable envelopes through the sole runner recorder while they wait.
Those loops are record-only: they may derive baseline, rollback, and advance
facts from the drained envelopes but must not update the estimator or recreate
facts from mutable adapter side state. The acceptance transition performs no
additional unrecorded IMU, observation, or collision drain. This replaces the
current `_fresh_reset_baseline`, `_observe_reset_proof`, and
`_accept_reset_proof` side-state/drain behavior only for `calibration-excite`,
prevents the 512-entry ingress queue from starving during a long proof, and
leaves every legacy stage unchanged.

Collision is intentionally not receiver/source-time lineage. Every runner
drain, including reset discard, countdown, watchdog, abort, and cleanup, passes
through one recorder and emits `aigp-vq2-runner-collision-observation/1` with
exact keys `schema`, `reset_generation`, `observation_sequence`,
`host_clock_id`, `observed_monotonic_ns`, `phase`, `disposition`, `boundary`,
and `collision`. Boundary is literal `runner_drain_not_receiver_receipt`;
collision is uint32 `id`, uint8 `threat_level`, and finite `impulse`.
Calibration-only collision diagnostics have exact `generation`, `handled`,
`dropped`, `high_watermark`, `capacity`, and `buffered`; they reset only at the
admitted reset-generation boundary. Accepted evidence requires zero dropped,
zero buffered, and every handled occurrence represented. Every post-boundary
collision still aborts; no `COLLISION` `MavlinkIngressV1` is invented.

Camera wire format and replay `/1` remain unchanged. The owned runner installs
a thread-safe calibration-only callback before forwarding each decoded snapshot
to `AsyncReplayRecorder`. It copies primitive timing/dimension facts only,
requires HxWx3 `uint8`, cross-checks image shape with `CameraFrame.width/height`
and the same `FrameTimingV1` identity, freezes the first observed width/height,
requires configured `640x360` before arm, and latches later drift. Callback
failure calls `replay.fail` and is polled before arm and immediately before each
send because the vision thread contains callback exceptions by design. One
admission event `aigp-vq2-decoded-dimensions-admission/1` has exact keys
`schema`, `config_sha256`, `expected`, `observed`, `first_frame_timing`,
`admitted_monotonic_ns`, and `status=admitted`; expected/observed are exact
width/height. A mismatch never emits admission and stops before arm. The
existing decoded-frame plus linked camera-timing replay rows remain the source
of actual shape, pixels/hash, stream/generation/frame/sim token, and receiver/
decode/publication QPC lineage. Capture FIFO is enabled for this stage; analysis
requires one exact timing link and stable shape for every decoded frame.

The adapter also emits one immutable
`aigp-vq2-attitude-target-outbound/1` receipt inside its send lock for every
attitude-target audit increment, including raised calls. Exact keys are
`schema`, `stream_id`, `reset_generation`, `outbound_sequence`,
`host_clock_id`, `call_start_monotonic_ns`, `call_end_monotonic_ns`, `api`,
`outcome`, `error_type`, and `wire`. `api` is one of
`send_attitude_rate`, `send_attitude_rate_from_attitude`, or
`send_attitude_quaternion`; outcome is `returned` or `raised`, with matching
null/non-null error type. `wire` is exact uint32 `time_boot_ms`, uint8 target
system/component/type mask, four finite `q_wxyz`, three finite
`body_rates_rad_s`, and finite `thrust`, after actual sign/mask conversion.
Its literal key set is
`{time_boot_ms,target_system,target_component,type_mask,q_wxyz,
body_rates_rad_s,thrust}`.
Its bounded queue has exact `generation`, `next_sequence`, `dropped`,
`high_watermark`, `capacity`, and `buffered` diagnostics. Acceptance requires
no drop/buffer, every outcome returned, direct-rate API, mask 128, identity
quaternion, exact expected signed rates/thrust, and trace count equal the
adapter attitude-target audit delta. This proves the payload passed to the API
and normal return, not physical wire delivery.

All other allowlisted sends likewise produce
`aigp-vq2-nonattitude-outbound/1` under the same lock with exact keys
`schema,stream_id,reset_generation,outbound_sequence,host_clock_id,call_start_monotonic_ns,call_end_monotonic_ns,category,api,outcome,error_type,wire`.
Category/API are exact `arm|disarm|sim_reset`/`command_long_send`,
`timesync`/`timesync_send`, or `gcs_heartbeat`/`heartbeat_send`; outcome/error
uses the same returned/raised rule. Command-long wire is exact target system,
target component, uint16 command, uint8 confirmation, and seven finite params;
TIMESYNC wire is exact int64 `tc1,ts1`; heartbeat wire is exact uint8
type/autopilot/base-mode/system-status and uint32 custom-mode. The literal alternatives are
`{target_system,target_component,command,confirmation,params}` for command
long, `{tc1,ts1}` for TIMESYNC, and
`{type,autopilot,base_mode,custom_mode,system_status}` for heartbeat. A single bounded
outbound trace queue contains both receipt schemas in sequence. Every outbound
audit category reconciles returned plus failed/uncertain receipts; acceptance
requires no failed/uncertain or dropped receipt and exact frozen payloads.

### Exclusive transport, process proof, and powered lease

Installed pymavlink `udpin:` is forbidden for this pilot because it enables
address reuse. A repo-owned opt-in endpoint creates IPv4 UDP, sets
`SO_EXCLUSIVEADDRUSE=1` before bind, never sets `SO_REUSEADDR`, binds exact
loopback `127.0.0.1:14550`, verifies option and `getsockname`, and closes on
every partial failure. It is supplied to the adapter without patching the
installed dependency. Before source freeze, each datagram is tested only by a
fresh scratch parser from the inventory-pinned pymavlink dialect used by the
adapter, with robust parsing disabled and no shared parser/state object. A valid
datagram is one unsigned MAVLink v1 (`0xfe`) or v2 (`0xfd`) frame whose declared
payload/frame length equals the datagram length, message ID exists in that
dialect, checksum including the dialect CRC-extra validates, the scratch parser
returns exactly one non-`BAD_DATA` message, its original message buffer equals
the complete datagram, and no leading/trailing byte, second frame, signature,
or incompatible flag remains. Empty, concatenated, partially consumed,
bad-checksum, unknown-dialect, signed, or trailing-byte datagrams are rejected
without production-parser mutation. The endpoint tentatively freezes the first
IPv4-loopback sender satisfying that rule before its bytes reach the production
parser or state, rejects/latches every other endpoint, and promotes that same
peer only after its own fresh heartbeat/race/IMU trio. Before promotion, the
only permitted outbound categories are the two announcements `timesync` and
`gcs_heartbeat`; reset, zero, disarm, arm, attitude target, and every other send
are forbidden. The peer persists across reset
generations; every write targets only it and raises on send error.
The child supplies the same exclusive factory for `0.0.0.0:5600`; fallback has
vision disabled and never binds 5600. Powered vision accepts only IPv4-loopback
datagrams, freezes the first syntactically valid frame-chunk sender before
calling `feed_datagram`, and latches any second source before receiver mutation,
publication, or use; its peer and rejected-source count enter the
cleanup certificate. Peer/count lifetime is the powered vision object and whole
attempt; stop/reset/start never clears or replaces it. Actual exclusive binds, not a port-table snapshot, are
the authoritative no-TOCTOU proof.

The passive `LiveSimulatorLease` behavior is unchanged: ordinary acquisition
still releases/rejects `WAIT_ABANDONED`. I0 adds a powered-only
append-only ledger mode unavailable to passive callers. Protected
`<attempt>\lease` publishes each generation through create-new
`pending-generation-%06d-<owner-role>.json`, flush/readback, then atomic
write-through no-replace rename to `generation-%06d.json`. Complete generations
are contiguous 0..4095 with no gap, duplicate, reparse, or unknown entry. Each
is exact
`aigp-vq2-live-lease-evidence/2` with keys
`schema,mutex_name,attempt_id,attempt_envelope_sha256,attempt_context_sha256,generation,predecessor_sha256,event,abandoned,owner_role,owner_token_sha256,wrapper_process,owner_process,child_process,cleanup_process,host_clock_id,qpc_frequency_hz,observed_monotonic_ns,phase,orphaned_pending,release_proved`.
Mutex name is literal `Global\AIGP-FlightSim-LiveLease-v1`. Predecessor is null only at generation zero and otherwise the prior complete
file hash. Event is `acquired|heartbeat|phase|takeover|release_intent|released`;
abandoned is true only for takeover and its successor records; release proved
is true only for `released`. Process values use exact PID, creation FILETIME,
session, image path/hash, and argv hash or are null only before that process
exists. Owner role is exactly
`wrapper|powered-child-parent-death|cleanup-fallback-parent-death`. Phase is
exactly one of `lease_acquire,launcher_return,
topology_and_training_attestation,prechild_identity_and_ports,child_spawn,
child_supervision,child_cleanup,child_exit_proof,fallback_spawn,
fallback_supervision,fallback_cleanup,postcheck_identity_process_ports,
lease_release_and_verify`. `orphaned_pending` is null except on takeover, where
it is null or exact `{path,size_bytes,sha256,owner_role}` for the sole preserved
pending file and predecessor owner role. Generation zero follows initial
mutex acquisition and precedes simulator contact. Heartbeat period is exact
1,000,000,000 ns, maximum gap is 1,500,000,000 ns, and phase records may occur
between heartbeats; publication failure or excessive gap latches production and
invalidates, while the 4096-row cap exceeds the complete 390-second wrapper
wall plus all phase events.

Initial wrapper owner hash equals the attempt's context-bound
`lease_owner_sha256`. A takeover owner hash is
`SHA256(UTF8("aigp-vq2-takeover-owner/1")||00||bytes.fromhex(context_sha256)||00||UTF8(owner_role)||00||role_secret32)`
using the child's or cleanup fallback's already-bound one-time secret. `/2`
never contains a raw owner/token/capability byte. Initial wrapper acquisition
accepts only `WAIT_OBJECT_0`; abandoned, inaccessible, busy, or unverifiable
initial state consumes/invalidates A01 and stops before live contact.

An orphan may take over only for cleanup. Before waiting it byte-validates the
immutable attempt and predecessor lease record, hash-matches its one-time role
capability, proves the inherited wrapper handle names the exact envelope
PID/creation time and is signaled, and proves its own retained-handle identity
and pre-bound role. It waits no more than the frozen `1.0 s` clipped deadline
and accepts only `WAIT_ABANDONED`; `WAIT_OBJECT_0`, timeout, failure, or any
other result grants no authority. The acquiring thread retains Windows mutex
ownership, re-reads the unchanged predecessor, atomically publishes/readback-
verifies the last complete generation. After abandoned ownership proves the
parent cannot still write, it permits at most one matching next-generation
pending file, preserves/hashes it without treating it as a record, and
publishes generation N+1 from the last complete predecessor with
`abandoned=true`, its `orphaned_pending` evidence, a new owner
token hash, and role `powered-child-parent-death` or
`cleanup-fallback-parent-death` before any cleanup send. The same thread alone
heartbeats/releases. Wrong/missing proof or publication failure sends nothing,
best-effort releases/closes, and cannot claim cleanup or release.

Release publishes `release_intent` while owned, releases the mutex, then
create-new publishes `released` with the API result and readback proof encoded
by `release_proved=true`; failure after intent leaves release unproved. After a
proved released event, create-new `live-lease.json` is
`aigp-vq2-live-lease-ledger/1` with exact keys
`schema,task_id,session_id,attempt_id,attempt_envelope_sha256,records,orphaned_pending_files,final_generation,final_record_sha256,release_proved`.
`records` is the exact ordered array of `{generation,path,sha256,event}` for every
generation file; orphaned files are the sorted exact
`{path,size_bytes,sha256,owner_role}` array and make acceptance impossible;
final values match the last released row and release proved is
true. This immutable final index, not a rewritten occurrence file, is the lease
artifact hashed by seal/report/terminal.

Every adapter outbound path, including the 0.1-second TIMESYNC/GCS-heartbeat
announcer, passes one shared under-`_send_lock` dispatcher with two disjoint
guards. The production guard requires exact role, live wrapper parent, wrapper
lease lineage, allowlisted category, and production QPC deadline. Parent death,
abort, or self-deadline permanently latches production off; it can never be
re-enabled in that process. “Latch zero” disables production and sends no
packet. The cleanup-only guard starts disabled and may enter one new cleanup
epoch only from either (a) valid live-parent delegation while the wrapper still
owns the lease or (b) a verified abandoned takeover with the parent proved
signaled and the survivor now owning the lease. Parent death first latches both
guards; successful takeover may enable only cleanup, never production.
Cleanup permits only exact-zero rate/thrust, disarm, reset, and frozen cleanup
announcements under its own lineage/deadline. Child and fallback connect both
await a fresh same-peer heartbeat, race-status, and IMU trio concurrently under
one absolute deadline; no non-announcement send is enabled before promotion.
Disconnect latches/stops, closes the socket to
unblock receive, then joins workers against recomputed remaining time; a live
worker/socket is unproved cleanup.

The wrapper proves IPv4 and IPv6 UDP ownership with owner-PID tables plus
exclusive probe-bind/close checks. It records two stable free snapshots before
child creation; maps actual child binds to its retained process identity; and,
after child/fallback exit, requires two stable free snapshots, successful
exclusive probes, and a final free snapshot. Process identity is always PID,
creation FILETIME, image path/hash, session, and argv from retained handles.
The exact non-breakaway child tree must be signaled/exited and no owned Python,
socket, temporary launch task, or port may remain. FlightSim may remain running
only with the exact accepted unchanged responsive topology.

Fallback runs exactly once iff a child was created, its entire process tree has
exited, and its bound cleanup certificate is absent/invalid or fails to prove
zero when applicable, newer-heartbeat disarm, reset, fresh advancing race/IMU
epoch, and final disarmed state. Artifact/seal failure with valid cleanup does
not cause another reset. Live/stuck child, occupied port, changed simulator,
consumed/mismatched capability, or unproved tree exit forbids fallback. While
wrapper lives, fallback acts only through its delegated one-time capability and
rechecks the parent before every send. On parent death it latches first and may
resume only after its own valid abandoned takeover. Any fallback use makes F00
invalid even if simulator state is resolved. Forceful termination may free a
process/port but is never cleanup proof.

On the parent-death child branch, the same thread that owns the abandoned mutex
continues the one-second lease heartbeat while it performs guarded cleanup and
publishes its cleanup certificate. It then publishes release intent, releases
and proves the mutex release, and finalizes the lease index before entering the
single bounded replay close and post-close child finalization. It never holds
the takeover mutex across `replay.close()`. On the normal branch the living
wrapper remains the lease owner and heartbeat producer while the child closes
replay. Direct injected tests kill the wrapper both during cleanup and after
cleanup/before replay close and prove this ordering, permanent production latch,
bounded exit, and absence of an escaped sender.

Any signaled wrapper handle permanently invalidates F00 and makes
`attempt-complete` impossible even if survivor cleanup succeeds. A living child
is the sole takeover cleanup authority. If the child already exited and the
wrapper dies before a fallback passed its capability release gate, no new
fallback may be created and no process may send; missing cleanup is unresolved
poison-equivalent. A fallback released before death may latch and perform its
own takeover. A capable survivor writes `wrapper_death` invalidation/poison
where possible; otherwise the absent terminal/root state remains fail-closed.

### Exact nested deadlines

The live freeze and attempt contain this exact integer nanosecond object:

```text
wrapper_total=390000000000; wrapper_live_contact_absolute_offset=300000000000; postrelease_total=90000000000; offline_precheck=10000000000; attempt_publish=2000000000; lease_acquire=5000000000; launcher_return=60000000000; topology_and_training_attestation=30000000000; prechild_identity_and_ports=5000000000; child_spawn=3000000000; child_total=110000000000; child_connect=15000000000; child_preflight=10000000000; child_reset_epoch=20000000000; child_normalize_disarmed=2000000000; child_countdown_go=8000000000; child_arm=2000000000; child_prepower_absolute_offset=52000000000; powered_stage=5000000000; child_powered_absolute_offset=57000000000; child_cleanup=15000000000; child_cleanup_absolute_offset=72000000000; child_replay_close=35000000000; child_replay_close_absolute_offset=107000000000; child_finalize=3000000000; child_exit_absolute_offset=110000000000; child_exit_proof=3000000000; parent_death_lease_takeover=1000000000; fallback_spawn=2000000000; fallback_total=25000000000; fallback_connect=5000000000; fallback_disarm=2000000000; fallback_reset_and_epoch=15000000000; fallback_finalize=2000000000; postcheck_identity_process_ports=5000000000; lease_release_and_verify=2000000000; bundle_verify=20000000000; capture_seal=10000000000; analysis=20000000000; split_publish=5000000000; terminal_publish=5000000000; poison_publish=5000000000; invalid_terminal_publish=5000000000; outbound_call=250000000; lease_heartbeat_period=1000000000; lease_heartbeat_max_gap=1500000000; poll_interval_max=50000000
```

Wrapper absolute boundaries are stored in the attempt before lease acquisition;
child/fallback boundaries are stored in process authority before capability
release. At every phase start, compute and freeze the exact phase observation
using `min(phase_start+duration,relevant_parent_deadline)` before waiting. The
wrapper persists it in the append-only ledger before its wait; the child
persists phases through replay close in replay before its wait; and fallback
retains the frozen record in memory for its final/failed certificate and process
result. Child
phases are also clipped to prepower `child_start+52 s`, production
`child_start+57 s`, cleanup `child_start+72 s`, replay close
`child_start+107 s`, and exit `child_start+110 s`.
Nonzero authority ends at `min(t0+5 s,child_start+57 s)`. Fallback phases are
clipped to its 25-second parent. Wrapper start is its first QPC read;
live-contact deadline is exactly start+300 seconds and total deadline is
start+390 seconds. Launcher, child/fallback, replay close, final process/port
proof, and lease release all end before live-contact cutoff. After proved lease
release, offline seal/analysis/split/terminal work gets
`min(release+90 s,total)` and no live operation is permitted. The nominal
success chain consumes at most 60 seconds of that window; the remaining 30
seconds is fail-closed publication/poison reserve and never extends live contact.
Define `terminal_parent_deadline` as
`min(wrapper_total_deadline,lease_release_monotonic_ns+postrelease_total)` iff
the lease was acquired and its release is proved; otherwise it is exactly
`wrapper_total_deadline`. Poison, `invalid_ready`, lifecycle finalization, and
invalid-terminal publication all clip to that parent, including failures before
a lease exists or before release can be proved. The no-live-after-release rule
still begins at the actual proved release; this parent grants no simulator,
socket, or cleanup authority.
Every invalid terminal gets one nonrefreshing
`min(start+invalid_terminal_publish,terminal_parent_deadline)` deadline. A failure
after lifecycle finalization additionally gets at most one nonrefreshing
`poison_publish` deadline followed by that invalid-terminal deadline; neither
is appended to or represented as a successful wrapper-lifecycle event.
Before `terminal_ready`, poison publication uses a `poison_publish`
`PhaseDeadlineV1`; `invalid_ready` plus lifecycle finalization use
`terminal_publish`; and create/write/flush/readback of `attempt-invalid` uses
`invalid_terminal_publish`. After `terminal_ready`, lifecycle or complete-
terminal failure uses a separate unledgered `poison_publish` deadline followed
by the unledgered `invalid_terminal_publish` deadline; neither modifies the
frozen ledger/lifecycle. Every deadline is independently clipped to
`terminal_parent_deadline` and never refreshed.

No retry refreshes a deadline and `now>=deadline` fails. The 50 ms poll maximum
applies to wrapper/child/fallback liveness, process, pipe, socket, and parent-
death supervision loops. The unchanged hash-pinned launcher may use its
internal bounded 25-second mutex/startup waits and 500 ms poll while the wrapper
polls its outer process at <=50 ms. The unchanged replay `/1` writer may perform
its one bounded close join only after production and cleanup guards are latched
and both UDP transports/workers are proved closed; it is clipped to
`replay_close` and live-contact deadline. Every other blocking operation is
clipped to its recorded parent. Larger process/finalization walls reserve
cleanup and proof time only and do not extend the five-second powered envelope.

### Powered lifecycle and authoritative command accounting

The child preserves the build-3385 sequence: connect/health; stop vision;
capture reset baselines; send reset; prove race and IMU rollback plus multiple
advancing samples; restart vision in that epoch; normalize and confirm disarmed;
witness countdown and GO plus 150 ms; require admitted stable decoded
dimensions and three-frame target; arm and confirm only on a newer exact
heartbeat envelope; recheck epoch/gate/watchdogs/target/collision/lineage/
deadline immediately before every paced send; latch on completion/abort; then
exact-zero, disarm confirmed on a newer heartbeat, reset, clean advancing race/
IMU epoch, final disarmed state, vision/transport termination, port/process
proof, and certificate. Cleanup failure fails the stage. Track/pose/odometry
never supplies truth or authority.

After successful capability release/validation, child and fallback each
create exactly one `aigp-vq2-powered-cleanup-certificate/1`; a failure before
that gate creates no certificate and no live contact. Its exact keys are
`schema,task_id,session_id,attempt_id,producer_role,cleanup_epoch,authority,
trigger,started_monotonic_ns,deadline_monotonic_ns,completed_monotonic_ns,
parent_state,lease,phase_deadlines,endpoints,outbound_receipts,zero_command,
disarm,reset,collisions,final_state,transport,outcome,failure_codes,
collection_invalidating_codes`.
Producer is `powered_child|cleanup_fallback`; epoch is respectively
`child-cleanup-0|fallback-cleanup-0`; trigger is
`normal_completion|stage_abort|parent_death|wrapper_fallback`; and outcome is
`proved|failed`. Failure codes are sorted unique from
`authority_invalid,deadline_expired,parent_dead,lease_invalid,connect_failed,
zero_failed,disarm_failed,reset_failed,final_state_unproved,
transport_unclosed,receipt_incomplete,internal_error`; they are empty iff
outcome is proved and otherwise nonempty. `collection_invalidating_codes` is a
sorted unique subset of `camera_missing,collision_observed,source_rejected,
unexpected_outbound`; it may be nonempty with proved cleanup, always invalidates
F00 externally, and never by itself requests fallback or a second reset.

- `authority` is exact
  `{process_authority,attempt_context_sha256,attempt_envelope_sha256,producer}`;
  process authority is `IdentityRefV1` and producer is `ProcessIdentityV1`.
  `parent_state` is exact
  `{mode,wrapper_process,observed_monotonic_ns,
  takeover_completed_monotonic_ns,takeover_lease_record_sha256}`;
  mode is `live_delegation|signaled_takeover`, wrapper is
  `ProcessIdentityV1`, and both takeover fields are nonnull iff signaled
  takeover. For takeover, receipts with call end at or before the parent-
  observation time use the validated live delegation; no call may overlap that
  time or start in the latch/acquire gap; every later call starts strictly after
  takeover completion and is authorized by the named takeover record. Completed
  disarm/reset phases are never repeated across the partition.
  `lease` is exact
  `{owner_role,generation,record_sha256,authority_valid}`; role is
  `wrapper|powered-child-parent-death|cleanup-fallback-parent-death`, generation
  is nonnegative, and authority-valid is true only when that record grants the
  producer's cleanup sends.
- `phase_deadlines` is the ordered exact `PhaseDeadlineV1` array frozen before
  waits. It may end early on failure but cannot omit a phase that started.
- `endpoints` is exact `{mavlink,camera}`. Camera is null only for fallback;
  each endpoint object is exact `{state,bind,frozen_peer,
  rejected_source_count}`. State is
  `not_opened|bound|peer_frozen|closed_without_peer|closed_with_peer`; bind is
  null only for `not_opened`, otherwise exact
  `{role,family,requested,actual,socket_policy,owner_process}`. Family is
  `AF_INET`; requested/actual are exact `{host,port}`; socket policy is
  `ipv4-exclusive-address-use`; owner is `ProcessIdentityV1`; and frozen peer is
  null until frozen and otherwise exact loopback `{host,port}`. Proved MAVLink
  is `closed_with_peer`; proved child camera may be `not_opened`,
  `closed_without_peer`, or `closed_with_peer` so a pre-arm no-frame abort can
  still prove MAVLink cleanup; fallback camera is null. Missing/rejected camera
  is collection-invalidating, not a reason for fallback. Any second source latches production/collection invalid before use but does
  not stop mandatory cleanup through the already frozen peer.
- `outbound_receipts` is the complete outbound-sequence-ordered array of exact
  attitude/nonattitude receipt objects. `zero_command` is exact
  `{state,required,requested,generated,terminal,outbound_receipt}`. State is
  `not_required|not_attempted|failed|returned`; requested, generated, terminal,
  and receipt nullability is state-derived and never discretionary. For
  `not_required`, required is false and all four are null. For `not_attempted`,
  required is true, requested is the exact four-rate/thrust object with all
  zeros, and generated/terminal/receipt are null. For `failed`, required is
  true, requested/generated and the rich not-sent terminal are nonnull; receipt
  is null exactly when its terminal records no API call or a started call with
  no observed return/raise, and otherwise is the byte-equal raised attitude
  receipt also present in the complete receipt array. A nonnull failed receipt
  can never have outcome `returned`; the rich terminal's call-boundary and audit
  facts distinguish no-call from started-but-uncertain.
  For `returned`, required is true and all four are nonnull, terminal is the
  rich sent event, and receipt is the byte-equal returned attitude receipt in
  the complete array. Fallback embeds this chain, while child evidence is also
  byte-equal to replay. `required` is derived, never chosen: it is true once a MAVLink peer
  was frozen, any arm/attitude call was attempted or its history is uncertain,
  and always for a capability-released fallback; it is false only when no live
  endpoint or outbound authority ever existed. Proved cleanup satisfies that
  derived value.
- `disarm` is exact
  `{state,request_monotonic_ns,receipt,heartbeat_before,heartbeat_after,
  newer_confirmed}` with state
  `not_required|not_attempted|request_failed|unconfirmed|confirmed`. Receipt is
  an exact nonattitude outbound object. `not_required` and `not_attempted` both
  require all four time/evidence fields null and `newer_confirmed=false` (the
  state records whether disarm was derived unnecessary or required but never
  started). `request_failed` requires nonnull request time and heartbeat-before,
  null heartbeat-after, and false. Its receipt is null iff the started API call
  produced no observed return or raise; otherwise it is the exact raised
  receipt in `outbound_receipts` and can never have outcome `returned`.
  `unconfirmed` requires
  nonnull request time, returned receipt, and heartbeat-before; heartbeat-after
  is null iff no post-request heartbeat was observed and otherwise is the last
  complete observed envelope; it is false. `confirmed` requires all four facts
  nonnull, a returned receipt, a strictly newer complete heartbeat-after with
  armed bit clear, and literal true. No other null mask is valid.
- `reset` is exact
  `{state,request_monotonic_ns,receipt,boundary,baseline,clean_epoch,
  advancing_race,advancing_imu,rollback_and_advance_confirmed}` with the same
  five states. Boundary is the complete reset-boundary object; baseline is
  exact `{race,imu}` with complete received envelopes; clean epoch is exact
  `{ingress_generation,race_anchor_boot_ms,imu_anchor_usec}`; advancing arrays
  contain every complete post-request envelope observed in occurrence order.
  `not_required` and `not_attempted` require all scalar/object facts null, both
  arrays empty, and false. `request_failed` requires nonnull request time,
  boundary, and baseline; its receipt is null iff the started API call produced
  no observed return or raise and otherwise is the exact raised receipt in
  `outbound_receipts`, never a returned receipt; clean epoch is null, both
  arrays are empty, and false. `unconfirmed` requires nonnull request time, returned
  receipt, boundary, and baseline; clean epoch is null exactly until both clock
  rollbacks have been observed and otherwise is the derived epoch; both arrays
  contain all and only observations accumulated for the attempted proof and may
  be empty; the boolean is false. `confirmed` requires every scalar/object fact
  nonnull, a returned reset receipt, at least two strictly advancing race and
  two strictly advancing IMU envelopes after the derived anchors, and literal
  true. No other null mask is valid.
- `collisions` is exact `{observations,invalidating_occurrence_count}` with the
  complete ordered runner-observation array and matching nonnegative count. A
  cleanup collision invalidates collection but never stops mandatory cleanup.
  `final_state` is exact
  `{state,heartbeat,disarmed,reset_epoch,last_race,last_imu}` with state
  `unobserved|partial|confirmed`. Unobserved requires all five facts null.
  Partial and confirmed require all five facts nonnull; partial means at least
  one of disarmed, reset-epoch consistency, or last-race/last-IMU advancement
  fails, while confirmed requires `disarmed=true` and every consistency/
  advancement check true. No other null mask is valid.
- `transport` is exact
  `{production_guard_latched,cleanup_guard_closed,vision_closed,
  mavlink_socket_closed,receiver_joined,announcer_joined,
  owned_handles_closed}` with booleans. A not-opened resource counts as closed;
  every field is true for proved outcome. Proved additionally requires confirmed
  disarm/reset/final state, returned required zero, valid authority/lease, and
  complete receipts.

The wrapper accepts cleanup proof only if outcome is proved, every
identity/deadline/receipt/heartbeat/reset/transport check passes, and its
create-new byte hash matches the producing process result. Otherwise the child
certificate is absent/invalid for the fallback truth table. Nonempty collection-
invalidating codes invalidate F00 but do not change that cleanup truth. A fallback has no
replay writer: all of its command and outbound facts live in its certificate;
it never appends to or repairs child replay bytes.

Captured stdout from child/fallback is exactly one canonical
`aigp-vq2-powered-process-result/1` object with keys
`schema,task_id,session_id,attempt_id,producer_role,process_authority_sha256,started_monotonic_ns,completed_monotonic_ns,outcome,reason_codes,phase_deadlines,cleanup_certificate,outbound_audit,artifacts`.
Outcome is `completed|failed`; reason codes are sorted unique from the attempt-
invalidation vocabulary, empty iff completed and otherwise nonempty. Process-
result `phase_deadlines` is the complete ordered `PhaseDeadlineV1` array
consistent with process authority. The cleanup certificate's array is the exact
prefix ending at certificate publication. The result appends every subsequently
started phase in order, including `replay_close`, `finalize`, and, when
applicable, exactly one `parent_death_lease_takeover` at its detection point.
The arrays are byte-equal iff no phase starts after certificate readback. A
takeover after replay close appears only in the process result and never causes
replay enqueue. `cleanup_certificate` is exact `{path,state,sha256}`, state
`absent|published|invalid`, with null hash only when absent and otherwise the
complete-file hash. `outbound_audit` is exact `OutboundAuditV1`.
`artifacts` is exact `{legacy_record,replay_bundle}`. For child, legacy is exact
`{path,state,sha256}` with state `absent|partial|closed` and nonnull hash iff
closed; replay is exact
`{path,state,dataset_hash,manifest_sha256,records_sha256}` with state
`absent|partial|closed` and all hashes nonnull iff closed. For fallback both
values are null. No extra stdout is allowed; diagnostics are sanitized,
capability-free UTF-8 on stderr with a hard one-MiB file ceiling.

Frozen replay-record/bundle `/1` core schemas are not changed. Existing core
command rows remain for compatibility; rich evidence uses the existing replay
event extension seam. Each schema below is nested as the sole `observation`
value in `record_event(<literal-name>, observation=<object>)`. Literal mappings
are received heartbeat/race/actuator/IMU ->
`received_heartbeat|received_race_status|received_actuator_output_status|received_imu`;
collision -> `runner_collision_observation`; dimensions ->
`decoded_dimensions_admission`; attitude/nonattitude outbound ->
`attitude_target_outbound|nonattitude_outbound`; phase deadline ->
`calibration_phase_deadline`; generated/sent/not-sent ->
`calibration_command_generated|calibration_command_sent|calibration_command_not_sent`;
tick disposition -> `calibration_tick_disposition`; and reset boundary ->
`calibration_reset_boundary`. These names are exact;
its `schema`/`session_id` and all other fields are never flattened into the
reserved replay envelope. The runner assigns `event_sequence` before enqueue,
so no asynchronous replay-writer acknowledgement or predicted replay record
sequence is required. It is one strictly increasing sequence across phase/
command/tick observations for `(attempt_id,producer_role=powered_child)`;
receiver/collision/outbound schemas retain their own frozen sequences.
`generation_sha256` covers only the complete canonical nested generated
observation, never the replay envelope. A false/raised enqueue immediately
latches capture invalid. I0 adds these exact observation schemas:

For these calibration rows, the unchanged replay `/1` outer envelope remains
exact `schema,session_id,sequence,type,capture_wall_time_ns` plus its event
extension fields. `schema=aigp-vq2-replay-record/1`, `type=event`, and every
mapped event above except `received_imu` has exactly the extension keys
`event,observation`. Existing `received_imu` retains exactly
`event,observation,linked_imu_record_sequence`; the link is a nonnegative prior
replay sequence naming the byte-equivalent core `imu` row emitted by the same
writer operation. Existing `camera_frame_timing` retains exactly
`event,observation,linked_decoded_frame_record_sequence`; that link is a
nonnegative prior replay sequence naming the exact decoded-frame row emitted by
the same writer operation. The existing `mavlink_ingress` and
`camera_frame_timing_observation` events retain exact `event,observation`
extensions. No other outer extra field is accepted for these events, and each
linked field is required only on its named event.

Frozen core `command` rows also reconcile exactly rather than merely coexist.
In ordinal order, every core `kind=generated` row maps one-to-one to a rich
generated observation and every core `kind=sent` row maps one-to-one to a rich
sent observation, with no extra core row. Its command primitive maps exactly
`roll_rate=roll_rate_rad_s`, `pitch_rate=pitch_rate_rad_s`,
`yaw_rate=yaw_rate_rad_s`, and byte-equal `thrust` from the rich command;
its frame token is `[generation,frame_id,sim_time_ns]` from rich source frame or
null for cleanup; and its `monotonic_s` is exactly the corresponding rich
generated/sent monotonic ns divided by `1_000_000_000`. A rich not-sent event
has no sent core row. This is the `command_pairs_exact` report check.

- `aigp-vq2-calibration-command-generated/1` keys:
  `schema,attempt_id,session_id,candidate_commit,attempt_context_sha256,event_sequence,host_clock_id,generated_monotonic_ns,reset_epoch,plan,scope,command_id,absolute_tick,segment_id,slot,command,source,watchdogs`.
- `aigp-vq2-calibration-command-sent/1` keys:
  `schema,attempt_id,session_id,candidate_commit,attempt_context_sha256,event_sequence,host_clock_id,sent_monotonic_ns,generated_event_sequence,generation_sha256,reset_epoch,plan,scope,command_id,absolute_tick,segment_id,slot,command,source,watchdogs,transport`.
- `aigp-vq2-calibration-command-not-sent/1` keys:
  `schema,attempt_id,session_id,candidate_commit,attempt_context_sha256,event_sequence,host_clock_id,recorded_monotonic_ns,generated_event_sequence,generation_sha256,reset_epoch,plan,scope,command_id,absolute_tick,segment_id,slot,command,source,watchdogs,outcome`.
- `aigp-vq2-calibration-tick-disposition/1` keys:
  `schema,attempt_id,session_id,attempt_context_sha256,plan_id,plan_sha256,event_sequence,host_clock_id,recorded_monotonic_ns,absolute_tick,segment_id,slot,disposition,generated_event_sequence,terminal_event_sequence,reason_code`.

Shared exact shapes are:

- `reset_epoch={ingress_generation,race_anchor_boot_ms,imu_anchor_usec}`, all
  nonnegative exact integers, is required for excitation and exactly null for
  cleanup-zero.
- `plan={plan_id,sha256}`; it is null only for cleanup-zero scope.
- `scope` is `excitation` or `cleanup_zero`. Excitation command ID is
  `excitation/%03d`, tick is 0..244, and plan/segment/slot are non-null. Cleanup
  ID is `cleanup/zero/0` and plan, absolute tick, segment, and slot are null.
- `slot={release_monotonic_ns,end_monotonic_ns,powered_expiry_monotonic_ns}`
  with release < end <= expiry.
- `command={roll_rate_rad_s,pitch_rate_rad_s,yaw_rate_rad_s,thrust}`, all finite.
- `source` has exact keys `frame`, `imu`, `race`, `heartbeat`, and `actuator`.
  `frame` is exact camera `stream_id,generation,frame_id,sim_time_ns` plus the
  complete `FrameTimingV1` primitive and admitted decoded width/height. The
  other four are the complete immutable received-envelope primitives above.
  Every generated excitation requires all five nonnull, including an actual
  fresh actuator occurrence; null is allowed only by the separately frozen
  cleanup scope, where all five are exactly null, never as fabricated values.
  Analysis joins these stable source/receive tokens to replay
  envelope record sequences after seal; the async recorder is not required to
  predict a future record sequence.
- `watchdogs` has exact keys `checked_monotonic_ns`, `heartbeat_age_ns`,
  `imu_age_ns`, `imu_advance_age_ns`, `race_age_ns`, `race_advance_age_ns`,
  `actuator_age_ns`, `vision_age_ns`, `estimator_healthy`,
  `target_consecutive`, `target_center_px`, `target_bbox_px`,
  `target_bbox_area_px`, `initial_target_bbox_area_px`,
  `roll_excursion_rad`, `pitch_excursion_rad`, `collision_count`, `gate_index`,
  `result`, and `failure_codes`. For excitation, every age is a nonnegative
  exact integer; target center is exactly two finite numbers; bbox is exactly
  four finite numbers with positive width/height; areas are positive finite;
  excursions are finite; counts/gate are nonnegative exact integers with gate
  zero; estimator health is boolean; result is `pass`; and failure codes is the
  empty array. For cleanup, only checked time is nonnull, every age/health/
  target/bbox/area/excursion/count/gate field is null, result is
  `cleanup_authorized`, and failure codes is empty.
  Failed checks are recorded only in not-sent/tick evidence, never used to
  authorize a generated excitation command.
- `transport` is exact `{receipt,audit_count_before,audit_count_after}`.
  `receipt` is the complete exact `aigp-vq2-attitude-target-outbound/1` object,
  byte-equal to the sole matching entry in `outbound_receipts`; both counts are
  nonnegative exact integers, the receipt outcome is `returned`, and
  `audit_count_after=audit_count_before+1`.
- `outcome` is exact `kind`, `reason_code`, `detail`, `audit_count_before`,
  `audit_count_after`, `call_started_monotonic_ns`, and
  `call_ended_monotonic_ns`. Kind is `skipped_after_generation` or
  `send_failed_or_uncertain`; a skip has unchanged audit and null call times;
  any failed/uncertain call invalidates. Reason code is drawn from
  `slot_missed,deadline_expired,stream_stale,imu_not_advancing,
  race_not_advancing,estimator_unhealthy,target_missing,target_unstable,
  target_out_of_corridor,target_too_large,attitude_excursion,
  collision_observed,gate_changed,capture_failed,parent_dead,lease_invalid,
  send_raised,internal_error`; detail is sanitized to at most 512 UTF-8
  characters. A no-call skip requires both call times null and a
  reason other than `send_raised`; failed/uncertain permits
  `send_raised|deadline_expired|parent_dead|lease_invalid|internal_error` and
  records every observed call boundary, with a null end only when no return or
  raise was observed.

`generation_sha256` hashes the complete canonical generated event. Sent/not-
sent duplicates reset/plan/scope/ID/tick/segment/slot/command/source/watchdogs
byte-equivalently and points to exactly one earlier runner-assigned generated
event sequence.
Tick disposition is `sent`, `skipped_before_generation`, or
`skipped_after_generation`. Sent has nonnull generated and terminal sequence,
where terminal names the matching rich sent event, and null reason. Pre-
generation skip has both sequences null and a nonnull listed non-send reason.
Post-generation skip has nonnull generated and terminal sequence, where
terminal names the matching rich not-sent event, and the same nonnull reason as
its outcome. Accepted evidence
contains each absolute tick 0..244 exactly once. A sent tick owns exactly one
generated+sent pair; post-generation skip owns generated+not-sent; pre-
generation skip owns neither. Pairing by rounded time or frame token alone is
forbidden. Missed indices are recorded once and never caught up.

The adapter audit counts attempted calls before pymavlink returns. Therefore
acceptance requires
`attitude_target_audit_delta = sent_count + send_failed_or_uncertain_count`,
failed/uncertain count zero, and one matching actual outbound receipt for every
sent event. It never relabels an audit increment as proved wire delivery.
Cleanup zero, if attempted, uses the same generated/terminal receipt chain and
is separately scoped, has no tick-disposition row, and for fallback uses a
certificate-local increasing event sequence beginning at zero rather than a
replay sequence. Position target, quaternion-attitude API, non-frozen
command-long, unknown category, orphan receipt, missing trace, changed rate
sign/mask, or nonzero cleanup immediately invalidates.

### Immutable invalidation, poison, seal, report, split, and terminal records

The child alone closes its replay writer within the child replay-close deadline
after cleanup/transport guard closure. While the lease is still held, wrapper
first proves the child/tree exit, validates the exact child result/certificate
and closed-file/no-handle state, resolves any fallback, then proves transport
closure, stable free ports, unchanged responsive simulator topology, and
publishes the final pre-release lease phase. It does not open or fully verify
bundle contents while a child could mutate them. It then
releases/readback-verifies the lease as the final simulator/transport operation.
No later code may inspect/contact FlightSim or bind/send a socket. Final lease
bytes are followed by full bundle verification, offline seal, analysis, split
records, and exactly one terminal record under the postrelease deadlines.
Artifact failure never triggers another cleanup. This ordering avoids both mutating a hash-sealed lease
artifact and releasing while live cleanup is unresolved. A post-release local
publication failure still invalidates/poisons the attempt but never triggers a
second reset or simulator contact.

Every failure after attempt-directory creation writes create-new
`aigp-vq2-powered-calibration-attempt-invalid/1` at `attempt-invalid.json` and
never edits, deletes, truncates, or replaces captured bytes. Its exact keys are
`schema,task_id,session_id,attempt_id,invalidated_at_utc,
invalidated_monotonic_ns,publication_timing,phase,reason_codes,reason_detail,
identity,artifact_state,cleanup_state,poison`. `publication_timing` is exact
`TerminalPublicationTimingV1` with phase `invalid_terminal_publish` and prepared
time equal to `invalidated_monotonic_ns`; its start/deadline is frozen before
serialization and never refreshed.
Reason codes are sorted unique, nonempty, and drawn only from:

```text
lease_busy,lease_abandoned,lease_unverifiable,launch_failed,topology_failed,training_unattested,build_or_candidate_changed,ports_busy,child_spawn_failed,child_failed,child_timeout,wrapper_death,stream_stale,watchdog_failed,capture_incomplete,unexpected_outbound,command_reconciliation_failed,deadline_expired,cleanup_unconfirmed,process_residue,port_residue,lease_release_unconfirmed,artifact_mismatch,terminal_write_failed,internal_error
```

Sanitized detail is at most 4096 UTF-8 characters and contains no capability.
`identity` is exact `attempt_envelope_state`, `live_freeze_sha256`,
`attempt_context_sha256`, `attempt_envelope_sha256`, `candidate_commit`,
`target_config_sha256`, `capture_authorization_sha256`, and
`excitation_plan_sha256`. Envelope state is `absent|partial|valid`; context and
envelope hashes are nullable when publication did not produce those complete
bytes and otherwise must match them. Freeze/candidate/config/authorization/plan
identities are nonnull because attempt-directory creation follows their proof.
`artifact_state` has exact keys
`legacy_record,legacy_record_sha256,replay_bundle,replay_dataset_hash,
replay_manifest_sha256,replay_records_sha256,bundle_verification,
bundle_verification_sha256,capture_seal,capture_seal_sha256,split_claim,
split_claim_sha256,split_registry,split_registry_sha256,analysis_report,
analysis_report_sha256,wrapper_lifecycle,wrapper_lifecycle_sha256,
attempt_complete,attempt_complete_partial_sha256,terminal_publication,
forensic_bytes_preserved`. Legacy state is `absent|partial|closed`; its hash is
nonnull iff closed. Replay state is `absent|partial|sealed`; all three replay
hashes are nonnull iff sealed. Verification, seal, claim, registry, report, and
lifecycle states are each `absent|partial|valid`; their matching file hash is
nonnull iff valid. Attempt-complete is `absent|partial`; its observed-byte hash
is nonnull iff partial. Terminal publication is literal `invalid_record`, and
forensic preservation is true.
`cleanup_state` is exact `InvalidCleanupStateV1` with
`child_exit=not_created|proved|unproved`,
`fallback=not_eligible|not_required|proved|failed|unproved`,
`ports=not_opened|free|owned|unproved`,
`lease=not_acquired|retained|released|unproved`, and
`processes=not_created|exited|residue|unproved`,
`transport=not_opened|closed|open|unproved`,
`scheduled_task=not_created|absent|present|unproved`,
`simulator_topology=not_launched|unchanged|changed|unproved`, and
`simulator_responsive=not_launched|yes|no|unproved`. `poison` is exact
`required`, absolute `path`, and nullable `sha256`. Its hash is null when
`required=false`. When required is true, the hash is the complete-file hash iff
a valid poison file was create-new flush/readback-verified; it is null only when
poison publication is absent, partial, or failed. Those states remain
fail-closed and cannot satisfy a poison-free predicate.

Poison is false only for one of these fully safe terminal predicates: (a) no
child was created, fallback is not eligible, ports are not opened or proved
free, lease was not acquired or is proved released, and simulator topology is
not launched or proved unchanged, processes are not-created/exited, transport
is not-opened/closed, the scheduled task is not-created/absent, and responsiveness is
not-launched/yes; or (b) child exit is proved, fallback is
not-required or proved, ports are free, lease is released, and topology is
unchanged, processes exited, transport closed, scheduled task absent, and
simulator responsiveness yes. In either case publication is safe only when
bundle verification and capture seal are each absent or valid,
claim/registry/report are all absent or all valid, wrapper lifecycle is valid,
attempt-complete is absent, and no reason is `wrapper_death`. Lifecycle may be
absent only in predicate (a) when attempt-envelope publication itself is
absent/partial and no ledger, lease, child, port, transport, task, or simulator
contact ever existed. Any partial state, other absent lifecycle, mixed
claim/registry/report tuple, or partial complete-terminal publication is
poisoned.
Every other tuple—including fallback failed/unproved, owned or
unproved ports, retained/unproved lease, changed/unproved topology, or unproved
child exit, process/task residue, open transport, unresponsive simulator, or
partial publication—requires poison even though the unsafe state is known
rather than unknown.

Required poison is create-new
`aigp-vq2-powered-calibration-live-poison/1` at the root before invalidation
where possible. Exact keys are
`schema,task_id,session_id,attempt_id,created_at_utc,created_monotonic_ns,
publication_timing,phase,reason_codes,attempt_context_sha256,
attempt_envelope_sha256,wrapper_process,child_process,cleanup_process,
lease_state,port_state,process_state,transport_state,scheduled_task_state,
publication_state,simulator_state,required_action`. `publication_timing` is
exact `ArtifactTimingV1` with phase `poison_publish` and prepared time equal to
`created_monotonic_ns`. Before terminal-ready its ledger phase end follows
readback; on the explicitly unledgered post-lifecycle failure branch, valid
poison bytes and mandatory readback are their own completion proof.
Process values use the retained-handle identity shape or null. Attempt context/
envelope hashes are nullable only when publication failed before complete
bytes. Lease state is exact `phase`, nullable `owner_token_sha256`, and
`release_proved`; port state is exact
`mavlink_14550` and `camera_5600`, each `not_opened|free|owned|unproved`;
process, transport, and task states use the cleanup enums above; publication
state is exact `bundle_verification`, `capture_seal`, `claim`, `registry`,
`report`, and `wrapper_lifecycle`, each `absent|partial|valid`, plus
`attempt_complete=absent|partial` and terminal
`missing|partial_complete`; simulator state is exact
`topology=not_launched|unchanged|changed|unproved` and
`responsive=not_launched|yes|no|unproved`; required action is literal
`new_reviewed_recovery_task_no_automatic_clear`. Poison is never overwritten,
removed, or automatically cleared. A terminal-write failure or an A01
directory with no valid sole terminal is equivalent fail-closed poison.

After lease release, the wrapper performs a pure offline verification and
create-new writes `aigp-vq2-replay-bundle-verification/1` at the frozen
`bundle_verification` path. Its exact keys are
`schema,task_id,session_id,attempt_id,verified_at_utc,
verified_monotonic_ns,timing,identity,bundle,checks,valid`. `timing` is exact
`ArtifactTimingV1` with phase `bundle_verify`, and its prepared time equals
`verified_monotonic_ns`:

- `identity` is exact `candidate_commit,live_freeze_sha256,
  attempt_context_sha256,attempt_envelope_sha256,child_authority_sha256,
  child_process_result_sha256,child_cleanup_certificate_sha256,
  lease_final_sha256`.
- `bundle` is exact `path,dataset_hash,manifest,records,frames`. Path is the
  frozen replay directory; dataset hash is the replay `/1` dataset hash;
  manifest and records are `ArtifactRefV1` named `replay_manifest` and
  `replay_records`; and frames is the sorted unique `ArtifactRefV1` array named
  `replay_frame/<decoded-sha256>` with exactly the blobs named by the manifest.
- `checks` has exactly boolean `manifest_schema_valid`,
  `records_schema_valid`, `dataset_hash_valid`, `records_complete`,
  `frame_blob_set_exact`, `frame_blob_hashes_valid`,
  `decoded_frame_shape_valid`, `camera_timing_links_exact`,
  `observation_schemas_valid`, `event_sequences_contiguous`,
  `resource_stats_zero`, and `writer_closed`.
- Every check and `valid` are literal true in a successfully published
  verification. Any failed/unavailable check writes no complete verification,
  seal, claim, registry, or report and enters immutable invalidation; a crash
  during its create-new write is represented as a partial verification.

After valid bundle verification, the wrapper create-new writes
`aigp-vq2-powered-calibration-capture-seal/1` with exact keys
`schema,task_id,session_id,attempt_id,sealed_at_utc,timing,identity,artifacts,
capture_stats,outbound_audit,cleanup`. `timing` is exact `ArtifactTimingV1`
with phase `capture_seal`.
`identity` has exact keys `candidate_commit,code_sha256,live_freeze_sha256,
attempt_context_sha256,attempt_envelope_sha256,target_config_sha256,
capture_authorization_sha256,excitation_plan_id,excitation_plan_sha256,
training_attestation_sha256,simulator_process_proof_sha256,
simulator_final_process_proof_sha256,child_authority_sha256,
cleanup_authority_sha256,lease_final_sha256,bundle_verification_sha256`;
cleanup authority is null iff fallback was never capability-released.

`artifacts` is a sorted unique `ArtifactRefV1` array. Required names are
`live_freeze,implementation_inventory,environment_inventory,import_inventory,
attempt_envelope,training_attestation,process_prechild,process_postchild,
child_authority,child_cleanup_certificate,lease_final,bundle_verification,
runner_stdout,runner_stderr,legacy_record,replay_manifest,replay_records` plus
one `replay_frame/<decoded-sha256>` for every manifest frame blob. Exact
conditional names `cleanup_authority,fallback_cleanup_certificate,
cleanup_stdout,cleanup_stderr` are all present iff fallback crossed capability
release and all absent otherwise. No seal/report/claim/registry/terminal artifact
is in this array.

`capture_stats` has exactly nonnegative integers
`record_count,decoded_frames,frame_blobs,camera_timing_records,imu_records,
mavlink_ingress_records,race_records,heartbeat_records,actuator_records,
collision_records,generated_commands,sent_commands,not_sent_commands,
tick_dispositions,capture_drops,decoded_frame_drops,writer_queue_drops,
writer_errors,ingress_drops,observation_queue_drops,collision_queue_drops,
outbound_trace_drops,queue_overflows`; all loss/error fields are zero.
`outbound_audit` is exact `OutboundAuditV1`; raised/drop/
buffer plus position/other are zero and category arithmetic reconciles.
`cleanup` is exact `TerminalCleanupV1`; its nullable fallback hash agrees with
the boolean and every enum uses the invalidation vocabulary.
The seal establishes integrity only, not acceptance.

Pure offline analysis emits
`aigp-vq2-powered-calibration-acquisition-report/1` with exact top keys
`schema,task_id,session_id,attempt_id,generated_at_utc,timing,collection_valid,
invalid_reasons,reference_scope,identity,input_artifacts,checks,counts,
command_accounting,excitation_accounting,descriptive_support,
calibration_status,unmeasured,split`. `timing` is exact `ArtifactTimingV1` with
phase `split_publish`; the separate wrapper `analysis` pair covers computation.
Its exact contract is:

- `invalid_reasons` is sorted unique from the invalidation vocabulary.
  `reference_scope` is exact
  `conditional_on_nominal_gate_config=true`,
  `geometry_status=nominal_unverified_for_build_3385_training`, and target-
  config key `target_config_sha256`. Collection validity never upgrades nominal geometry to truth.
- `identity` is `candidate_commit`, `live_freeze_sha256`,
  `attempt_context_sha256`, `attempt_envelope_sha256`,
  `target_config_sha256`, `capture_authorization_sha256`,
  `excitation_plan_id`, `excitation_plan_sha256`,
  `training_attestation_sha256`, `simulator_process_proof_sha256`,
  `simulator_final_process_proof_sha256`, `child_authority_sha256`, nullable
  `cleanup_authority_sha256`, `lease_final_sha256`, and
  `bundle_verification_sha256`.
- `input_artifacts` is `capture_seal_sha256`, `bundle_dataset_hash`,
  `bundle_verification_sha256`, `bundle_manifest_sha256`,
  `bundle_records_sha256`, `legacy_record_sha256`,
  `lease_final_sha256`, `runner_stdout_sha256`,
  `runner_stderr_sha256`, `child_cleanup_certificate_sha256`, and nullable
  `fallback_cleanup_certificate_sha256`.
- `checks` has exactly boolean `identity_bound`,
  `build3385_training_attested`, `bundle_complete`, `frame_hashes_valid`,
  `decoded_dimensions_640x360_stable`, `camera_lineage_complete`,
  `imu_lineage_complete`, `race_heartbeat_actuator_collision_lineage_complete`,
  `capture_loss_zero`, `ingress_loss_zero`, `outbound_allowlist_exact`,
  `command_pairs_exact`, `ticks_0_through_244_accounted`, `plan_exact`,
  `watchdogs_passed`, `cleanup_confirmed`, `fallback_not_used`,
  `child_process_tree_exited`,
  `ports_released`, `lease_released`, `simulator_topology_unchanged`,
  `simulator_responsive`, `scheduled_task_absent`,
  `exclusive_binds_and_peers_exact`, `collection_invalidating_codes_empty`,
  `conditional_on_nominal_gate_config`, and `no_fit_or_rank_inspection`.
  `fallback_not_used` is true iff the capture-seal cleanup summary has
  `fallback_used=false` and `fallback=not_required`; cleanup authority, fallback
  certificate, cleanup stdout, and cleanup stderr are absent; and every
  corresponding nullable report identity/input hash is null. Successfully
  published analysis requires this check literal true.
- `counts` has exact nonnegative `decoded_frames`, `unique_decoded_hashes`,
  `camera_timing_records`, `imu_records`, `mavlink_ingress_records`,
  `race_records`, `heartbeat_records`, `actuator_records`, `collision_records`,
  `generated_commands`, `sent_commands`, `not_sent_commands`, `ticks_sent`,
  `ticks_skipped_before_generation`, `ticks_skipped_after_generation`,
  `capture_drops`, `decoded_frame_drops`, `writer_errors`, `ingress_drops`,
  `queue_overflows`, and `send_failed_or_uncertain`.
- `command_accounting` is exact `attitude_target_audit_delta`,
  `generated_count`, `sent_count`, `not_sent_count`,
  `unmatched_generation_count`, `unmatched_sent_count`,
  `failed_or_uncertain_count`, `envelope_violation_count`,
  `payload_mismatch_count`, and `all_reconciled`.
- `excitation_accounting` has exact keys
  `plan_id,plan_sha256,tick_count,segments,first_release_monotonic_ns,
  last_slot_end_monotonic_ns,powered_expiry_monotonic_ns`; tick count is 245 and
  ordered segment entries are exact `segment_id,planned_ticks,generated,sent,
  skipped`.
- `descriptive_support` has exact keys
  `target_observation_count,target_center_x_px_min,target_center_x_px_max,
  target_center_y_px_min,target_center_y_px_max,target_bbox_area_px_min,
  target_bbox_area_px_max,gyro_x_rad_s_min,gyro_x_rad_s_max,gyro_y_rad_s_min,
  gyro_y_rad_s_max,gyro_z_rad_s_min,gyro_z_rad_s_max,roll_reversal_count,
  pitch_reversal_count,semantics`; published-valid evidence has positive target
  and IMU counts and therefore every listed extrema is finite; counts are
  nonnegative, and semantics is
  `descriptive_only_no_acceptance_threshold`.
- `calibration_status` has exactly `intrinsics`, `distortion`,
  `camera_to_body_rotation`, `camera_imu_time_model`, `rank`, `covariance`, and
  `empirical_limits`, all literal `uncomputed`.
- `unmeasured` is exact sorted
  `absolute_host_phase,accepted_calibration_coefficients,command_to_actuator_response,empirical_limits,encode_queue_component_delays,package2_acceptance,render_exposure_delay`.
- `split` has exact keys `assigned_split,claim_path,claim_sha256,registry_path,
  registry_sha256,activation`; split is `discovery_fit` and activation is
  `requires_matching_attempt_complete` until terminal validation.

`collection_valid` is true iff every exact check is true and invalid reasons are
empty; every successfully published report therefore has literal true and an
empty reason array. It means acquisition integrity only. The analyzer does not fit, compute
singular values/rank/Jacobians/covariance, label features, inspect another
session, or say calibration passed/accepted. The wrapper first runs every check
and constructs the prospective claim, registry, and report bytes in memory. A
failed check writes none of those three and goes directly to invalidation. On a
fully valid result, their hashes are non-circular (registry binds claim; report
binds claim and registry; claim binds neither), then create-new publication proceeds claim, registry,
report. Any partial publication is invalid/poisoned and never reused.

The create-new `aigp-vq2-package2-run-split-claim/1` has exact keys
`schema,task_id,session_id,attempt_id,claimed_at_utc,claimed_monotonic_ns,timing,
run_id,assigned_split,identity,reset_epochs,run_artifacts,
decoded_content_sha256,derivative_sha256,collision_policy`. `timing` is exact
`ArtifactTimingV1` with phase `split_publish` and prepared time equal to
`claimed_monotonic_ns`.
Run ID is `F00-A01/reset-epoch-1/excitation-1`; split is `discovery_fit`;
`identity` is exact `attempt_context_sha256,attempt_envelope_sha256,
capture_seal_sha256,excitation_plan_id,excitation_plan_sha256`; `reset_epochs`
is exactly one `{ingress_generation,race_anchor_boot_ms,imu_anchor_usec}`;
`run_artifacts` is a sorted unique `ArtifactRefV1` array with exact required
names `bundle_verification,child_cleanup_certificate,legacy_record,
replay_manifest,replay_records,runner_stdout,runner_stderr` plus exactly one
`replay_frame/<decoded-sha256>` for each manifest frame blob. Receiver, command,
collision, and phase evidence is nested in `replay_records`, not represented as
a fictitious separate file. Because valid collection forbids fallback, no
fallback artifact is permitted. The array explicitly excludes capture seal,
claim, registry, report, lifecycle index, and terminal records.
`decoded_content_sha256` is the sorted unique array of replay `/1` `frame_hash`
values. Each is exactly
`SHA256(canonical_json({"shape":[H,W,3],"dtype":"|u1"}) || 00 ||
contiguous_uint8_bgr_bytes_in_C_order)`, matching the frozen replay writer and
not a raw-pixel-only digest. `derivative_sha256` is the exact empty
array because this pilot creates no label/crop/fit derivative; and policy is
`f00_fixed_future_whole_run_discovery_fit_or_global_exclusion`.

Immediately after the claim, create the initial immutable
`aigp-vq2-package2-split-registry/1`; it has exact keys
`schema,task_id,session_id,attempt_id,published_at_utc,
published_monotonic_ns,timing,registry_id,revision,previous_registry_sha256,
claims,content_groups`. `timing` is exact `ArtifactTimingV1` with phase
`split_publish` and prepared time equal to `published_monotonic_ns`.
It is registry `vq2-package2-calibration`, revision 1, previous null; `claims`
is one exact `{claim_path,claim_sha256,session_id,attempt_id,run_id,
assigned_split,activation}` with F00/A01/run/discovery-fit and activation
`requires_matching_attempt_complete`. Each content group is exact
`decoded_sha256`, sorted unique `run_ids`, `assigned_split=discovery_fit`,
`disposition=assigned`, and the same activation. Content groups are sorted
uniquely by `decoded_sha256`. A row is active only if exactly
one valid `attempt-complete` hash-binds its seal/report/claim/this registry;
without that terminal it is inert and globally ineligible. `globally_excluded`
is the only other disposition allowed in later revisions and changes evaluation
eligibility, never the immutable claim label. Successors append
N+1 revisions with prior byte hash and all claims. Before any later split opens,
the full chain is hash-pinned and globally reconciled. F00 is never relabeled.
A future run sharing an F00 decoded hash wholly joins discovery-fit or every
collided item and derivative is globally excluded; per-frame moves, retained
derivatives, or held-out repair by resplitting are forbidden.

Only after valid final lease bytes, seal, report, split claim/registry, and
process/port/cleanup proof does wrapper create-new
`aigp-vq2-powered-calibration-attempt-complete/1`. Exact keys are
`schema,task_id,session_id,attempt_id,completed_at_utc,
completed_monotonic_ns,deadline_monotonic_ns,publication_timing,identity,
artifact_hashes,cleanup`. `publication_timing` is exact
`TerminalPublicationTimingV1` with
phase `terminal_publish`, prepared time equal to `completed_monotonic_ns`, and
deadline equal to `deadline_monotonic_ns`.
`identity` has exactly `candidate_commit,code_sha256,live_freeze_sha256,
attempt_context_sha256,attempt_envelope_sha256,target_config_sha256,
capture_authorization_sha256,excitation_plan_id,excitation_plan_sha256,
wrapper_lifecycle_sha256`. The lifecycle is finalized and readback-verified
after the ledger's `terminal_ready` end row; `completed_monotonic_ns` is the
last QPC read after that verification and before create-new terminal
serialization, is less than the frozen terminal deadline, and terminal
write/flush/readback must also finish before that unchanged deadline.

`artifact_hashes` has exactly `bundle_dataset_hash,
bundle_verification_sha256,capture_seal_sha256,analysis_report_sha256,
split_claim_sha256,split_registry_sha256,bundle_manifest_sha256,
bundle_records_sha256,legacy_record_sha256,runner_stdout_sha256,
runner_stderr_sha256,lease_final_sha256,training_attestation_sha256,
simulator_process_proof_sha256,simulator_final_process_proof_sha256,
implementation_inventory_sha256,environment_inventory_sha256,
import_inventory_sha256,child_authority_sha256,cleanup_authority_sha256,
child_cleanup_certificate_sha256,fallback_cleanup_certificate_sha256,
cleanup_stdout_sha256,cleanup_stderr_sha256,wrapper_lifecycle_sha256`.
`bundle_dataset_hash` is the replay `/1` canonical dataset hash; every
`*_sha256` is a complete-file byte hash. All values are nonnull lowercase
64-hex except cleanup authority, fallback certificate, and cleanup stdout/
stderr, which are null on the only successful branch because fallback use
invalidates F00. `cleanup` is exact `TerminalCleanupV1` with `fallback_used`
false, `fallback=not_required`, the same nonnull child-certificate hash, null
fallback hash, and every other terminal enum proved/free/released/unchanged as
defined above. Exactly one valid complete or invalid terminal is allowed.
Complete+invalid, neither, hash mismatch, partial lifecycle, or any poison makes
the attempt unusable.

## Historical pre-I0 executable draft (superseded)

The following four subsections are retained only to show what the I0 audit
corrected. They are not executable authority. The authoritative I0 correction
immediately above controls every CLI, path, schema, lease, deadline, transport,
cleanup, command-accounting, and evidence decision.

### Historical live command and private paths

The only operator command is a no-selector wrapper invocation from the fresh,
detached, exact candidate live worktree. Before invocation, the wrapper must
reject defined `PYTHONPATH`, `PYTHONHOME`, or `PYTHONSTARTUP`; a user-site
directory on `sys.path`; any untracked, ignored, or source-adjacent Python
module/bytecode; and any import-origin mismatch. The exact environment sets
`PYTHONNOUSERSITE=1` and `PYTHONDONTWRITEBYTECODE=1`:

```powershell
Set-Location -LiteralPath `
  C:\Users\John\aigp-worktrees\wt-package2-powered-calibration-live
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\John\killallhumans\.venv\Scripts\python.exe' `
  -E -s -B -m scripts.aigp_vq2_powered_calibration_probe `
  --target-config `
  <live-worktree-absolute-config-json> `
  --capture-authorization <private-absolute-authorization-json> `
  --session-id F00 `
  --output-root `
  C:\Users\John\aigp-evidence\2026-07-20-package2-powered-calibration-pilot
```

The wrapper alone constructs and records the exact child command:

```powershell
& 'C:\Users\John\killallhumans\.venv\Scripts\python.exe' `
  -E -s -B -m scripts.aigp_vq2_run --stage calibration-excite `
  --powered-attempt-envelope <private-absolute-attempt-json> `
  --wrapper-process <pid-and-creation-time> `
  --attempt-capability-handle <inherited-read-handle> `
  --parent-liveness-handle <inherited-process-handle> `
  --record <private-session-jsonl-gz> `
  --replay-bundle <private-session-vq2replay> `
  --recording-approved
```

For this wrapper-only child invocation, `--recording-approved` means that the
wrapper validated and hash-bound the exact task-scoped user/operator capture
authorization above. It does not assert organizer credentials and cannot be
used by another stage, session, build/mode, or physical/publication consumer.

Those four internal options are wrapper-only and mandatory for this stage.
Their values are generated and validated by the wrapper; the capability secret
travels through the inherited pipe, never as argument text. The runner rejects
direct `calibration-excite` invocation, a non-inherited/extra handle, a parent
PID without the exact creation time, or an envelope/capability mismatch.

The interpreter is CPython `3.12.2` with SHA-256
`9b0bffb7a259cd2722df454fdfff41ee13665820cff1f578b1d97d31f9ef93d5`.
The development-test lock SHA-256 is
`5484a062c5c71343bbbf4f2bed3aa93a040f83311ef95627d33059888aa65f34`.
L0 freezes the complete installed-package and environment inventory. Module
spec origins for every repository import must resolve below the detached live
worktree; third-party imports must resolve below the exact venv; and standard
library imports must resolve below the exact interpreter installation. The
wrapper repeats this audit before simulator launch and before child creation.

It rejects relative, reused, pre-existing, symlink/reparse, cross-root,
world-writable, dirty-candidate, wrong-commit, wrong-build, wrong-mode,
unreviewed configuration/authorization, or selector-bearing input. Session
`F00` is permanently assigned to discovery/fit and every duplicate or derived
artifact remains in that split. It can never become limit, held-out, or repeat
evidence.

That split label does not transfer access. After this task's E0 audit, no
successor may open, replay, annotate, derive from, or fit F00 until a new
simulation-only data-use authority names the exact admitted artifact hashes and
permitted purpose. A pre-fit design alone is not access authority.

Content-hash grouping is global across every Package 2 session, not merely
local to F00. If any decoded frame SHA-256 occurs in runs assigned to different
splits, every complete containing reset-epoch/excitation run is reassigned as
one unit to a single split, including every derived frame, crop, label, and
observation from those runs. If whole-run reassignment is not allowed, the
collided content and every derivative are excluded from every evaluation and
the exclusion is recorded. Moving only matching frames is forbidden. The same
run-level rule is applied before any future split is opened.

### Historical lease, process, port, and attempt gate

Before simulator contact, the wrapper completes its offline identity checks and
durably creates the new attempt envelope. It then follows this exact order:

1. Prove the exact clean committed candidate and admitted target-config plus
   capture-authorization hashes.
2. Create, without overwrite, a private ACL-restricted session directory and
   durable attempt/invalidation envelope.
3. Acquire `Global\AIGP-FlightSim-LiveLease-v1` within `5.0 s` and publish its
   cryptographically random owner token, wrapper PID, and process-creation time.
   Abandoned, inaccessible, or unverifiable lease state fails closed.
4. While retaining that lease, invoke the hash-pinned canonical
   `scripts/launch_sim.ps1` with absolute simulator path
   `C:\Users\John\AIGP\AIGP_3385\FlightSim.exe` and
   `-StartupTimeoutSeconds 60`. The script retains its independent
   `Global\AIGP-FlightSim-Launch` mutex and double-launch refusal. Its current
   review lead hash is
   `d133043d691175b150787218025e5d811da2329495c48cebb316ff92c7d6852e`;
   L0 re-hashes it and freezes the absolute PowerShell executable and complete
   launch command.
5. Within `15.0 s` after launcher return, prove exactly one documented
   launcher/payload topology, including launcher SHA-256
   `0d3217fa72e9fee847b2c154432476a687f21b79f0ab6b910728a6254b4dce32`,
   payload SHA-256
   `9064dd1547a30afea1e3fb87652cc8194c3f5af556be40629dc491bb4f681362`,
   parentage where the launcher remains alive, process creation times, session,
   command lines, responsive visible unminimized window, and operator-attested
   Training mode. A pre-existing process is accepted only if it passes the same
   proof; it is never relaunched.
6. Prove UDP ports `14550` and `5600` unowned, repeat the build/process proof,
   and only then create the powered child.
7. Bind QPC frequency, host boot, process starts, graphics/focus context,
   target-config and capture-authorization identities, candidate commit, exact
   commands, and private paths into provenance before the first child packet.

The launcher phase has a `60.0 s` limit, the powered child has an independent
`60.0 s` creation-to-exit wall limit, fallback cleanup has a `15.0 s` limit,
and the wrapper has a `180.0 s` creation-to-final-release wall limit. Every
connect, bind, reset, epoch proof, countdown, arm, stage, disarm, cleanup, seal,
and release await receives a smaller explicit monotonic deadline frozen at L0;
no nested await is unbounded or allowed to extend its parent deadline.

The child is not killed merely because its parent handle closes. It receives a
read-only attempt capability and wrapper-liveness handle and owns its independent
`60.0 s` deadline. On wrapper death it latches zero and performs no further send
until it acquires the now-abandoned `Global\AIGP-FlightSim-LiveLease-v1` within
`1.0 s`, validates the same attempt/owner-token lineage, and atomically records
phase `child-parent-death-cleanup` with its PID and creation time. Only while it
owns that lease may it execute bounded disarm/reset/epoch cleanup; it releases
and verifies the lease last. Failure to acquire or validate authority permits no
cleanup send, publishes unresolved poison evidence, and exits. Wrapper death
invalidates all evidence even if child cleanup later succeeds. Direct tests must
prove successful takeover, unavailable/wrong-token refusal, injected parent
death, and a stalled child operation.

The task permits one live attempt total. A busy/abandoned/inaccessible lease,
bind conflict, process/build change, stale stream, missing or changed
configuration/authorization, child timeout, unexpected outbound category,
capture drop/overflow, incomplete
bundle, cleanup uncertainty, powered-child/transport/port residue, or artifact
mismatch
permanently invalidates the attempt and ends live work.

FlightSim itself may intentionally remain running after the attempt. It is not
child residue. The final proof requires the exact accepted build/process
topology to remain unchanged and responsive while every Python child,
transport, owned socket, temporary launch task, and port binding is gone.

### Historical powered lifecycle and cleanup

The runner retains the authoritative build-3385 lifecycle without weakening:

1. connect and perform the existing internal health checks;
2. stop vision during reset;
3. prove both race and IMU clock rollback plus multiple advancing samples;
4. restart vision only in the proved epoch;
5. normalize and freshly confirm disarmed;
6. witness countdown and wait through GO plus `150 ms`;
7. send the arm request, then confirm arm only on a newer heartbeat;
8. recheck epoch, gate zero, all watchdogs, and the absolute deadline
   immediately before every paced send;
9. latch further production on abort or completion; and
10. send exact zero rate/thrust, send the disarm request, confirm disarm only on
    a newer heartbeat, reset, prove the clean race/IMU epoch and disarmed state,
    then terminate vision and transport. Unconfirmed cleanup makes the result
    failed.

The live wrapper retains the lease through child exit, bundle sealing or
invalidation, process/build recheck, port release, and cleanup proof. The lease
is released and verified last.

Before L0, `scripts/aigp_vq2_powered_cleanup.py` must provide a separate,
hash-pinned cleanup-only fallback. It has no manually usable mode, accepts no
flight stage or nonzero target, and requires a one-time 256-bit cleanup
capability delivered through an inherited anonymous pipe. The wrapper creates
separate child and cleanup capabilities in a create-new, current-user-only ACL
attempt envelope; neither secret is logged or placed on a command line.

Fallback order is exact: while retaining the live lease, the wrapper proves the
powered child and its process tree exited, both ports are unowned, and its own
PID/start time and lease owner token match the attempt envelope. It then starts
the hash-pinned fallback, which atomically consumes the cleanup capability,
repeats the parent/attempt proof, exclusively binds UDP `14550`, leaves `5600`
unbound, and establishes a fresh heartbeat/race/IMU receive epoch. Only then may
it send exact zero, send disarm and require a newer heartbeat, send reset and
prove clean advancing race/IMU epochs plus disarmed state. It closes transport,
proves its `14550` binding released, and exits; the wrapper re-proves both ports
free before releasing the lease last.

If the wrapper dies before fallback creation, fallback cannot consume the
capability and the child's parent-death cleanup remains the only authority. If
the wrapper dies after fallback creation, fallback latches zero, acquires the
now-abandoned live lease under the same attempt token before any further send,
finishes bounded cleanup, releases it, poisons the evidence, and exits. Failure
of child exit, capability consumption, bind, fallback cleanup, port proof, or
lease release publishes a poison marker and reports unresolved emergency
state. Forceful process termination is never labeled simulator cleanup.

### Historical outbound audit and evidence

Powered acceptance validates each outbound category, not the passive
`disallowed_count == 0` aggregate. Only GCS heartbeat, TIMESYNC, bounded reset,
arm, disarm, and rate-mode attitude-target sends are permitted. Position
target, quaternion-attitude mode, command-long other than the frozen
reset/arm/disarm paths, and every unknown category must be zero.

Every successful attitude-target audit increment must reconcile one-to-one
with an exact sent-command record containing candidate/session/epoch, absolute
tick, segment ID, command, source frame token, current race token, and
pre-send watchdog/deadline result. Generated but skipped commands are separate
and never counted as sent.

One accepted artifact must include:

- complete replay verification and every decoded frame identity/hash;
- exact camera source tokens, camera host receipt/publication/consume points,
  IMU source times, IMU host receipts, generation and receive sequence;
- zero capture, writer, ingress, queue, and receiver loss/overflow;
- exact excitation-plan ID/hash and complete segment/tick accounting;
- raw gyro, accelerometer, race, heartbeat, actuator, collision, generated and
  sent command lineage;
- target-config and capture-authorization identities, actual decoded dimensions,
  and descriptive target visibility/support; feature annotations, correspondence
  uncertainty, and fit eligibility remain `uncomputed` and unaccepted until F0;
- descriptive observed rotation-axis diversity, reversals, rate changes,
  angular acceleration, and image-region/scale/tilt support, with rank,
  singular-value, conditioning, identifiability, model, parameter-order,
  scaling, Jacobian, and nuisance-order fields fixed to `uncomputed`;
- outbound reconciliation, process/load context, ports, lease, cleanup, and
  artifact hashes; and
- explicit `unmeasured` fields for accepted calibration coefficients,
  empirical limits, absolute host phase, render/exposure delay, encode/queue
  component delays, command-to-actuator response, and Package 2 acceptance.

## Offline implementation and test gate

Direct tests must cover at least:

- strict target-config and simulation-capture-authorization schemas, exact
  stable-byte hashes, change/replacement behavior, no default or field override,
  and rejection of incomplete, inferred, PAK-bearing, camera-default-bearing,
  physical/publication-enabled, or wrong-build/mode/session inputs;
- actual decoded-dimension observation and within-session stability, raw
  camera/IMU/race/heartbeat source and host lineage, absence of pose/map truth,
  and exact config/authorization binding through attempt, capture, and report;
- exact round trips and missing/unknown/type/range/bool/nonfinite rejection for
  every new envelope/event/attempt/terminal schema; same-QPC construction,
  defensive copies, cross-stream sequence, source-time joins, no actuator
  overwrite, queue/drop accounting, and legacy ingress-drain compatibility;
- the exact 245 tick/index partition, every segment boundary, absolute deadline
  selection, multi-tick skip behavior, duration, rate/yaw/thrust bounds, no
  unsafe override, fixed gate zero, and no crossing path;
- reset/countdown/GO/new-heartbeat ordering and no arm after failed proof;
- watchdog, gate-index, target, lineage, and deadline rechecks immediately
  before sends, including injected stalls, missed ticks, `0.05 rad` excursions,
  center/bbox/area limits, and every collision aborting without an exception;
- every normal, exception, cancellation, timeout, partial-construction, and
  cleanup path;
- powered outbound allowlist, actual post-sign adapter payload receipts, raised-
  call evidence, generated/sent/not-sent pairing, all 245 tick dispositions,
  and attempted-audit reconciliation without a wire-delivery claim;
- capture completeness, source/host lineage, schema strictness, split binding,
  global cross-session hash collisions with whole containing-run reassignment
  or exclusion, `uncomputed` structural fields, and no calibration-acceptance
  claim or pre-model singular-value inspection;
- absolute interpreter/hash, environment rejection, import-origin/bytecode,
  detached-clean-worktree, canonical-launch, exclusive lease, process/build,
  port ordering, deadlines, and atomic invalidation;
- `SO_EXCLUSIVEADDRUSE` before bind with no reuse, bind-conflict and partial-
  construction close, one-peer freeze, second-source rejection, actual child
  PID/port mapping, fallback never binding 5600, and stuck-worker disconnect;
- child self-deadline, injected wrapper death, liveness/capability checks, exact
  process-tree exit proof, cleanup-only fallback bind/consume order, poison
  behavior, initial passive/initial-powered abandoned refusal, authorized
  same-thread abandoned cleanup takeover, wrong-token/live-parent/wrong-role/
  changed-predecessor refusal, and failure to claim clean release;
- exact wrapper/child/fallback QPC deadline arithmetic, cleanup reserves, no
  relative-deadline refresh after injected stalls, and final port/transport
  proof before lease release;
- unchanged historical replay `/1` verification, new rich event acceptance,
  and continued rejection of rich fields inserted into frozen core rows; and
- passive probe, preflight, and existing powered-stage non-regression.

Every automated/non-live case uses injected kernels/transports or OS-assigned
ephemeral loopback ports and a unique per-test non-production mutex name. No
test may bind or send fixed port `14550`/`5600`, acquire/open
`Global\AIGP-FlightSim-LiveLease-v1`, launch/query FlightSim as a side effect,
or read/write the private live root. Production literals become reachable only
after wrapper/capability admission and are exercised solely by the one L1
command, never pytest.

After each edit, run directly affected tests. Before L0, run from the required
clean candidates:

```powershell
.\scripts\dev.cmd test-target tests/test_aigp_vq2_runner.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_calibration_target.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_powered_attempt.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_powered_runtime.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_powered_calibration_probe.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_powered_calibration_analysis.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_powered_cleanup.py
.\scripts\dev.cmd test-target tests/test_aigp_live_lease.py
.\scripts\dev.cmd test-target competition/tests/test_vq2_capture.py
.\scripts\dev.cmd test-target competition/tests/test_aigp_mavlink.py
.\scripts\dev.cmd test-target competition/tests/test_vq2_vision.py
.\scripts\dev.cmd test-target tests/test_aigp_loop_replay.py
.\scripts\dev.cmd test-vq2
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
```

At the promotion boundary run `test-full-non-live` from a fresh exact
promotion worktree, then the exact hash-pinned VQ2 suite from a separate fresh
worktree. Integrate the unchanged candidate and run post-merge VQ2. No test
command may launch or contact FlightSim.

## P1 admission evidence

P1 is admitted on the exact offline candidate. Its tracked identities before
this record-only update are:

- nominal target-config SHA-256
  `e16e2a70e6be8d6d083e5739773473090c62d244a1b69f120ce027f51b84f82b`;
- validator source SHA-256
  `768969dfb1b77f9e99de1da18aa9248c94ca7536fd889c039e2cc0dc0a948ae6`;
- direct-test source SHA-256
  `26bef1c5ac81b1b71cd4721ca4946a728c9aa144861b3260cacc6ed743da83e3`;
- private authorization path
  `C:\Users\John\aigp-evidence\2026-07-20-package2-powered-calibration-pilot\capture-authorization.json`;
- private authorization SHA-256
  `5fb440b48ae7d1d60d8c59399eadb5c01f74ffa049bc7245da01c9cea3d04b9a`;
- direct suite: `85 passed` in `0.22s`; and
- canonical `test-vq2`: `1648 passed, 1 skipped` in `40.82s`.

The protected private root is owned by `DESKTOP-1RCQP2L\John`; its ACL has
inheritance disabled and grants only that user full control. The authorization
file inherits only that effective user access. The hash-pinned CLI validated the
tracked config and real authorization together and emitted only their two exact
hashes. Independent schema/security review returned `CLEAR`, including strict
JSON/types, geometry consistency and replacement identity, stable file/hash
reads, simulation-only non-transfer scope, no defaults/field overrides, and no
PAK or camera-prior path.

P1 admits only this collection protocol and these exact identities. It does not
admit target/build linkage, geometry uncertainty, labels, calibration values,
powered implementation, or simulator execution. No PAK, FlightSim process,
port, frame capture, reset, arm/disarm, target, or powered command was touched.

## Historical inactive R1 completion evidence

R1 is complete on the synthetic-only candidate. The exact reviewed identities
before this record-only update were:

- parser source SHA-256
  `3573390cf7377eac306cf10085b2621dcd0ec017d9fb0bfcf656460ceedf18cf`;
- direct-test source SHA-256
  `82991ae67e91871fd126f016d3a117a1b305ccd5c82c1e956ca9f017cb401cc7`;
- direct suite: `83 passed, 1 skipped` in `0.74s`;
- `test-vq2`: `1563 passed, 1 skipped` in `42.43s`;
- `test-fast`: `2661 passed, 21 skipped, 42 deselected` in `115.10s`;
  and
- isolated `test-unit` rerun: `2661 passed, 21 skipped, 42 deselected` in
  `105.89s`.

The single direct skip is the real Windows symlink-creation case because this
account lacks `SeCreateSymbolicLinkPrivilege`; deterministic device, namespace,
fixed-drive, disk-handle, reparse-attribute, final-path, identity, collision,
cleanup, and ACL cases still execute. Independent calibration/schema and
filesystem/security reviews both returned `CLEAR` on the exact hashes. A native
Win32 private-ACL create, flush, identity, ACL, readback, and cleanup smoke also
passed. No test or review read a real build asset or contacted FlightSim.

## I0 correction review evidence

The authoritative I0 executable-interface correction was independently cleared
on exact pre-evidence-update document SHA-256
`0cbbf582069dfb32211eaf606c2f5c3f3d0d1fd85763d3017a82bc7e4c8727f3`:

- schema/evidence review returned `CLEAR` for cleanup discriminants, fallback
  exclusion, replay compatibility, artifact/hash ordering, terminal and poison
  publication, process authority, and bounded state transitions;
- powered-safety review returned `CLEAR` for source promotion, process/job
  authority, lease/takeover, absolute deadlines, cleanup/fallback, the four user
  decisions, and non-live isolation; and
- current-code compatibility review returned `CLEAR` for anonymous-pipe framing,
  scratch MAVLink validation, created/preexisting topology evidence, phase
  suffix ordering, and failed/uncertain outbound-call representation.

`git diff --check` passed. All review was read-only; no simulator process, fixed
port, PAK, or private evidence was contacted. This state/evidence-only update
does not alter the reviewed executable contract. Its exact resulting hash is
confirmed before the freeze commit.

## Current entry audit and I0 release

P0 records the user's four scope decisions. The cooked-PAK route is inactive
and no organizer PAK clearance is an active prerequisite. The separately
authorized-collection disjunct in the parent Package 2 entry gate is satisfied
only for this one bounded build-3385 Training simulation discovery pilot.

The actual runtime boundary supplies UDP JPEG frame/generation identities and
opaque `sim_time_ns`, decoded image contents and dimensions, `HIGHRES_IMU` raw
gyro plus opaque `time_usec`, host performance-counter receipts, heartbeat, and
race status. It supplies no intrinsic, distortion, FOV, camera mount, exposure
phase, calibrated camera/IMU clock map, pose, odometry, or usable track/gate map.
Those absent values remain `uncomputed`; 640x360 is an observed historical
decoded shape and the parent-frozen compatibility requirement. The collector
must observe it from the actual decoded frame before arm and continuously; it is
not a wire-header field or intrinsic prior. Any different shape stops for a
revised config/contract rather than being resized or silently accepted.

The public
[technical specification issue 00.02](https://www.theaigrandprix.com/wp-content/uploads/2026/05/260508_Technical_Spec_0002.pdf)
publishes the nominal gate dimensions used above, but states a Virtual Qualifier
1 scope. The configuration therefore treats them as editable, hash-pinned,
unverified inputs for build 3385 Training. No public camera constants are used.

P1 now passes with the exact identities and evidence above. The collection
configuration remains conditional on nominal, VQ1-scoped public gate geometry;
all absent camera and timing quantities remain `uncomputed`.

The read-only I0 audit found that the earlier draft exceeded its owned receiver
surface, relied on unsupported abandoned takeover and nonexclusive pymavlink
bind behavior, paired mutable payload side state with ingress, starved cleanup
under its process wall, and left attempt/command/poison/report/split interfaces
underspecified. The authoritative correction above now owns and freezes those
interfaces while leaving excitation and safety bounds unchanged. Independent
authority/lifecycle, evidence-schema, and compatibility review now clears the
correction. The next admitted work is I0 implementation in the dependency order
recorded above, beginning with immutable wire schemas, pure powered contracts,
and OS/runtime primitives. This releases no T0, integration, live-freeze, or
powered gate.

No FlightSim process, port, frame capture, reset, arm/disarm, target, or powered
command has occurred. No PAK is read, and the completed passive timing tranche
is not repeated.
