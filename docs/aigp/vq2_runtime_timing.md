# VQ2 runtime timing foundation

**Target:** FlightSim build 3385, Training mode

**Status:** accepted passive receiver/capture-ingress evidence; not full M1,
not a no-capture production-latency result, and not authority for a powered
stage

## Measured boundary

Candidate `a50f4ea0e18b2f5a295fdb1cc94e183734616601` recorded three
accepted, replay-capture-loaded passive preflights. The measured path is:

```text
UDP JPEG packet arrival -> reassembly -> decode -> publication
  -> passive consumption -> detection -> tracking -> replay snapshot enqueue

decoded MAVLink message receive boundary
  -> exact HIGHRES_IMU or other ingress envelope
```

Every accepted timing point is an integer nanosecond occurrence on Windows
`QueryPerformanceCounter`, exposed by `time.perf_counter_ns()` and labeled
`host-perf-counter`. The camera source timestamp remains an opaque ordering
token. It is not converted to host time or subtracted from a host occurrence.
The legacy `received_monotonic_s` freshness clock remains separate.

The additive evidence schemas are:

- `aigp-vq2-mavlink-ingress/1`;
- `aigp-vq2-received-imu/1`; and
- `aigp-vq2-camera-frame-timing-observation/1`.

The existing `aigp-vq2-replay-record/1` core records remain readable. Exact
events bind receiver generation and sequence or camera stream, generation,
frame ID, source token, and publication sequence. The analyzer verifies every
frame blob, exact event, one-to-one decoded/processed/timed frame relationship,
queue diagnostics, and replay manifest before accepting a session.

## Accepted build-3385 sessions

The private evidence root is
`C:\Users\John\aigp-evidence\2026-07-20-package3b-m1-passive-timing`.
Sessions 02 through 04 are the exactly three accepted sessions. Each retained
five seconds of healthy state after readiness and completed in six seconds of
runner observation. Counts are exact, not estimates.

| Session | Dataset hash | Replay records | Camera observations | Exact IMU arrivals | Camera publish / consume Hz | IMU Hz |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 02 | `b92b28209eac340b4c7f9397666ba2fe781fed86648f500f025556c6a0e1cfc0` | 2,806 | 182 | 700 | 30.151 / 30.240 | 115.756 |
| 03 | `591844c0b6679bf3aed205e77d9e520c5d615eead8f9ed0e2fa04e7cf6daede4` | 2,817 | 181 | 705 | 30.063 / 30.101 | 115.767 |
| 04 | `7539af300d1d03861f44c5c55cb5793c88bb60de04892355d79e4ee757409cd6` | 2,816 | 181 | 705 | 30.035 / 30.109 | 116.218 |
| Total | three distinct datasets | 8,439 | 544 | 2,110 | per-session only | per-session only |

All three sessions have zero command records, zero disallowed outbound sends,
zero ingress or capture drops, zero timing gaps, zero frame-count shortfalls,
zero frame-work observations over 20 ms, and complete frame-blob verification.
Only GCS heartbeat and TIMESYNC were sent. SIM_RESET, arm, disarm, attitude
target, position target, and other command counts are all zero.

## Passive timing distributions

Values below are milliseconds in `p50 / p95 / p99 / maximum (count)` order.
They are rounded only for this tracked summary; the hash-bound private analysis
files retain the exact nanosecond values.

| Metric | Session 02 | Session 03 | Session 04 |
| --- | --- | --- | --- |
| Camera packet span | 0.253 / 0.459 / 1.047 / 1.106 (182) | 0.260 / 0.392 / 0.999 / 1.144 (181) | 0.240 / 0.454 / 0.939 / 1.042 (181) |
| Final packet to reassembly | 0.040 / 0.063 / 0.069 / 0.073 (182) | 0.043 / 0.059 / 0.064 / 0.067 (181) | 0.041 / 0.064 / 0.067 / 0.076 (181) |
| Decode | 0.903 / 1.217 / 1.305 / 1.359 (182) | 0.769 / 1.040 / 1.408 / 1.557 (181) | 0.889 / 1.233 / 1.408 / 1.653 (181) |
| Decode to publication | 0.002 / 0.003 / 0.003 / 0.003 (182) | 0.002 / 0.003 / 0.003 / 0.004 (181) | 0.002 / 0.003 / 0.003 / 0.003 (181) |
| First packet to publication | 1.211 / 1.721 / 2.079 / 2.333 (182) | 1.081 / 1.696 / 1.922 / 2.066 (181) | 1.181 / 1.666 / 1.936 / 2.007 (181) |
| Publication to consumption | 8.065 / 14.653 / 16.136 / 32.136 (182) | 8.608 / 14.980 / 15.758 / 17.969 (181) | 8.408 / 14.707 / 15.492 / 16.329 (181) |
| Detection | 1.317 / 1.649 / 1.902 / 2.595 (182) | 1.436 / 1.787 / 2.015 / 2.967 (181) | 1.260 / 1.637 / 1.831 / 2.800 (181) |
| Tracking | 0.011 / 0.014 / 0.015 / 0.016 (182) | 0.011 / 0.014 / 0.017 / 0.018 (181) | 0.011 / 0.014 / 0.016 / 0.021 (181) |
| Total passive frame work | 1.700 / 2.106 / 2.397 / 2.970 (182) | 1.827 / 2.206 / 2.502 / 3.188 (181) | 1.646 / 2.098 / 2.337 / 3.173 (181) |
| Publication interval | 34.625 / 36.878 / 38.016 / 38.315 (181) | 34.546 / 36.148 / 37.695 / 38.203 (180) | 34.657 / 36.183 / 37.731 / 38.119 (180) |
| Consumption interval | 31.507 / 46.935 / 47.896 / 48.100 (181) | 31.426 / 47.147 / 48.270 / 49.515 (180) | 31.533 / 47.209 / 48.007 / 48.195 (180) |

`total passive frame work` ends after the synchronous replay snapshot/copy
enqueue. Asynchronous blob serialization is diagnosed separately by the
bounded replay writer. These measurements therefore characterize the approved
replay-capture-loaded path and must not be presented as no-frame-write runtime
latency.

## Queue, process, graphics, and cleanup context

| Session | Ingress high-water (IMU / other / total, capacity 4,096 each queue) | Replay writer high-water | Vision capture FIFO high-water / capacity | Payload CPU-time delta across 8 samples | Payload working set MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 02 | 4 / 5 / 9 | 9 | 2 / 256 | 8.5625 s | 219.805–219.852 |
| 03 | 10 / 9 / 19 | 20 | 1 / 256 | 8.0156 s | 219.852 |
| 04 | 7 / 7 / 14 | 14 | 1 / 256 | 9.0469 s | 219.805–219.852 |

Every sampled payload window was valid, visible, and unminimized, and was not
the foreground window. The launcher/payload PIDs, parent relationship,
creation times, paths, session ID, and executable hashes remained stable.
QPC frequency was 10,000,000 Hz. These are contextual observations, not a
controlled graphics/load experiment or a causal load correction.

The named mutex `Global\AIGP-FlightSim-LiveLease-v1` was acquired and cleanly
released for each session. UDP ports 14550 and 5600 were free before and after
each probe. No live-probe poison marker was created.

## Artifact hashes

The private `probe-context.json` in each accepted session binds these hashes,
the candidate commit/code hash, process samples, analyzer source hash, lease,
and postcheck Git/port state.

| Artifact | Session 02 | Session 03 | Session 04 |
| --- | --- | --- | --- |
| Bundle dataset | `b92b28209eac340b4c7f9397666ba2fe781fed86648f500f025556c6a0e1cfc0` | `591844c0b6679bf3aed205e77d9e520c5d615eead8f9ed0e2fa04e7cf6daede4` | `7539af300d1d03861f44c5c55cb5793c88bb60de04892355d79e4ee757409cd6` |
| Bundle manifest | `45cc7139603fa3009805d7bc3b1041a8e5c598b3addb9e5729cb5dac3142e2fc` | `2bc65d1fd746a7e6c28e6a9d6ea60f1952d3887debce334a09b92743aa524642` | `f60612e4c12c23875fa74b31aa1f25c978f7f161e5403192e56fd5969df61f3e` |
| Bundle records | `5a8515574597c6fc8f53e3a7df41c1ff102f99f5e07b90ee785048d4caee0cea` | `0eaa7cf41096d4e40bcf1dca48a874352dbfb1be00ce3ccdc24a4cc2d934d309` | `3a87825590d665dea458ac8e3189f4c3936c2d89b47076eb16c38b646e0f3fe8` |
| Timing analysis | `c119331b6e35c6b2c8c6a76de05af2fb39d025e6996f307064043c6bad32b741` | `86e0ae0bf3ac23eaa6a58dfd886e8acb0b0f1669376194ce7a891d5743cd96a1` | `d1827ea2eb8882a526373c38b9fd9aea99bfc1b9cce91fbc1cc2586aab7d9adc` |
| Lease evidence | `64dbc7b359632483964b264b857bd69882c6c7fd257e56f247f073f265f8ef17` | `fc75bad56813d506aadf28b72f37f500c1a98802b9ba9916905be7fde1aee125` | `e573ad9b00c2d49199ca1bbf20c88b38d084a9b2f25c400c42ea9cd51cb4cace` |
| Legacy JSONL | `1fc3e8a1920af9122aea9d9113009101b16da7cf65fe9efacadef068f2181b50` | `ad3150d3e94ed8e76cec407f7c748cc44eaa5c9e5f655ac650f99c956591f8cb` | `203082333a034f29cbb8dbb07a5a72685ffabf4a651f8d1cc1273827d4b84521` |
| Runner stdout | `e857af4bb9a97e521d10ffc3b9891a24af3bf24cd8c810abb5ec1431e8d09eba` | `79a2c34f3b750e91ab20a683934ca556258018f2ea73013cf3059aa53c10587f` | `34c795f16b77471fcd798cf75481296853cd9f1964533b9a9f9ffe6fa8c4a559` |
| Runner stderr | `b703e9fe128c19947e55e0c77a90f0eb913973bebb040ce4e4dd487302f1cad0` | `4f6492a5a338ccad595d6610711208cd85278d4f8ca8160057ee5af7b929829e` | `7abb5c720b7f89839634e61970175ae0fed3ed97f201ab38a224accd17d1330f` |

## Rejected attempt retained as failure provenance

Session 01 at commit `923c24bf4c6f0257f292cd407296347b88bdbf2c`
was rejected because generic async dataclass normalization omitted exact
contract `schema` fields. The runner otherwise completed passive preflight,
but replay capture was incomplete and therefore unusable. The wrapper marked
the session invalid, proved child exit, released its lease and ports, and
created no poison marker. Commit `a50f4ea0e18b2f5a295fdb1cc94e183734616601`
preserves the three exact contract roots through the worker as fresh public
primitive trees and revalidates them on the writer thread. Independent review
accepted the fix before sessions 02–04. The failed evidence remains private
and was not deleted or upgraded.

## Explicitly unmeasured and remaining work

This tranche does not measure or claim:

- control scheduler deadlines or skip behavior;
- command-send timing or command-to-actuator/gyro causal response;
- a camera or IMU measurement-clock model;
- calibrated camera/IMU offset or command-effect delay;
- simulator/wall ratio;
- machine detection of Training mode (mode is operator-attested);
- no-capture production latency; or
- a controlled host-load or graphics-focus effect.

A later M1 task still needs a no-send shadow 50 Hz scheduler and production
handoff/deadline/skip evidence. A successor passive tranche must compare
timing-only/no-frame-write load. Actual send-to-actuator/gyro delay is powered
work and requires a separately named, freshly authorized task. Nothing in this
dossier authorizes reset, arm, hover, Gate 0 motion, or any flight target.
