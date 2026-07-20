# Package 2 powered calibration discovery pilot

- Task ID: `vq2-package2-powered-calibration-pilot`
- Parent: `vq2-package2-production-calibration`
- State: `P0 scope pivot recorded - stopped before P1 admission; no simulator contact`
- Starting main commit:
  `ccbea8ac9fa9b53c3f86324662f616041693277b`.
- R0 contract commit: `49b331f`.
- Branch: `package2-powered-calibration`.
- Worktree:
  `C:\Users\John\aigp-worktrees\wt-package2-powered-calibration`.
- Future detached live worktree:
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
5. `L0 live freeze`: the exact candidate commit, detached live worktree,
   interpreter/environment/import inventory, launch command, child command,
   private paths, configuration and authorization hashes, excitation ID,
   attempt ID, build/process proof, lease, outbound allowlist, phase deadlines,
   cleanup fallback, and invalidation rules pass independent review.
6. `L1 pilot`: at most one powered discovery session is attempted.
7. `E0 review`: evidence is validated offline and independently reviewed
   before any fit or successor collection is proposed.
8. `F0 pre-fit successor`: before opening or using F00, admit a new hash-bound
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
- `scripts/aigp_vq2_run.py` for one additive fixed
  `calibration-excite` stage;
- `scripts/aigp_vq2_powered_calibration_probe.py` for the exact live wrapper;
- `scripts/aigp_vq2_powered_calibration_analysis.py` for pure evidence
  validation and acquisition-support summaries;
- `scripts/aigp_vq2_powered_cleanup.py` for a cleanup-only out-of-process
  fallback after a confirmed child exit;
- direct tests named after those five scripts;
- additive direct cases in `tests/test_aigp_vq2_runner.py`; and
- `scripts/dev.ps1` only if needed to include the new offline tests in the
  explicit VQ2 suite.

The already committed `scripts/aigp_vq2_build_reference.py`, its direct test,
and its VQ2 test registration are inert historical surfaces. This candidate
does not invoke, extend, or use them. Their presence grants no PAK authority and
no PAK identity may enter the active configuration, attempt, capture, or report.

The behavior candidate does not own promotion policy or trusted-manifest
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

## Exact live command and private paths

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

## Lease, process, port, and attempt gate

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

## Powered lifecycle and cleanup

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

## Outbound audit and evidence

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
- the exact 245 tick/index partition, every segment boundary, absolute deadline
  selection, multi-tick skip behavior, duration, rate/yaw/thrust bounds, no
  unsafe override, fixed gate zero, and no crossing path;
- reset/countdown/GO/new-heartbeat ordering and no arm after failed proof;
- watchdog, gate-index, target, lineage, and deadline rechecks immediately
  before sends, including injected stalls, missed ticks, `0.05 rad` excursions,
  center/bbox/area limits, and every collision aborting without an exception;
- every normal, exception, cancellation, timeout, partial-construction, and
  cleanup path;
- powered outbound allowlist and exact sent-command count reconciliation;
- capture completeness, source/host lineage, schema strictness, split binding,
  global cross-session hash collisions with whole containing-run reassignment
  or exclusion, `uncomputed` structural fields, and no calibration-acceptance
  claim or pre-model singular-value inspection;
- absolute interpreter/hash, environment rejection, import-origin/bytecode,
  detached-clean-worktree, canonical-launch, exclusive lease, process/build,
  port ordering, deadlines, and atomic invalidation;
- child self-deadline, injected wrapper death, liveness/capability checks, exact
  process-tree exit proof, cleanup-only fallback bind/consume order, poison
  behavior, and failure to claim clean release; and
- passive probe, preflight, and existing powered-stage non-regression.

After each edit, run directly affected tests. Before L0, run from the required
clean candidates:

```powershell
.\scripts\dev.cmd test-target tests/test_aigp_vq2_runner.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_calibration_target.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_powered_calibration_probe.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_powered_calibration_analysis.py
.\scripts\dev.cmd test-target tests/test_aigp_vq2_powered_cleanup.py
.\scripts\dev.cmd test-vq2
.\scripts\dev.cmd test-fast
.\scripts\dev.cmd test-unit
```

At the promotion boundary run `test-full-non-live` from a fresh exact
promotion worktree, then the exact hash-pinned VQ2 suite from a separate fresh
worktree. Integrate the unchanged candidate and run post-merge VQ2. No test
command may launch or contact FlightSim.

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

## Current entry audit and stop

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

Execution currently stops before P1. The strict config/authorization validator,
tracked nominal configuration, private stable authorization bytes/hash, direct
tests, and independent P1 admission review do not yet exist. No FlightSim
process, port, private capture, reset, arm/disarm, target, or powered command
occurs before those exact artifacts are admitted and I0/T0/L0 subsequently pass.
No PAK is read, and the completed passive timing tranche is not repeated.
