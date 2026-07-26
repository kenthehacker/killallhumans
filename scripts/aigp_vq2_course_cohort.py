"""Read-only cohort analysis for compact VQ2 visual-course flight evidence.

The analyzer deliberately consumes only the compact fast-flight-cycle
``result.json`` and ``session.jsonl.gz`` artifacts.  It never creates,
rewrites, repairs, or indexes evidence.  The fitted command effects are
diagnostic correlations, not promotion evidence or plant-authority claims.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPORT_SCHEMA = "aigp-vq2-visual-course-cohort-analysis/1"
DEFAULT_YAW_IMAGE_GAIN_NORM_PER_RAD = 0.62
COMMAND_FIELDS = ("roll_rate", "pitch_rate", "yaw_rate", "thrust")
RATE_COMMAND_FIELDS = ("roll_rate", "pitch_rate", "yaw_rate")
ROLL_REVERSAL_DEADBAND_RAD_S = 0.01
MAX_COMMAND_AGE_S = 0.10
MAX_INTERPOLATION_GAP_S = 0.12
MAX_EFFECT_LAG_S = 0.20
EFFECT_LAG_STEP_S = 0.01
MIN_EFFECT_SAMPLES = 8
MIN_EFFECT_CORRELATION = 0.25


@dataclass(frozen=True)
class TickSample:
    time_s: float
    command: dict[str, float]
    body_rates: tuple[float, float, float] | None
    rpy: tuple[float, float, float] | None
    imu_accel: tuple[float, float, float] | None
    gate_index: int | None


@dataclass(frozen=True)
class FeatureSample:
    time_s: float
    gate_index: int
    role: str
    track_id: str
    x: float | None
    y_down: float | None
    log_scale: float | None
    reported_x_rate: float | None
    reported_y_rate_down: float | None
    reported_log_scale_rate: float | None
    visible: bool
    censored: bool
    ambiguous: bool


@dataclass(frozen=True)
class TimedValue:
    time_s: float
    value: float


@dataclass
class TraceRead:
    events: list[dict[str, Any]]
    invalid_json_lines: int = 0
    non_object_lines: int = 0
    read_error: str | None = None

    @property
    def status(self) -> str:
        if self.read_error or self.invalid_json_lines or self.non_object_lines:
            return "partial"
        return "complete"


def default_evidence_root() -> Path:
    """Return the fast-flight-cycle evidence root without creating it."""

    configured = _environment_value("AIGP_EVIDENCE_ROOT")
    base = Path(configured) if configured else Path.home() / "aigp-evidence"
    return base / "fast-flight-cycles"


def _environment_value(name: str) -> str | None:
    # Keep the import surface small and make environment access easy to stub.
    import os

    value = os.environ.get(name)
    return value if value else None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _triple(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    converted = tuple(_finite(item) for item in value)
    if any(item is None for item in converted):
        return None
    return converted  # type: ignore[return-value]


def _event_time_s(row: Mapping[str, Any]) -> float | None:
    wall_time_ns = _finite(row.get("wall_time_ns"))
    if wall_time_ns is not None:
        return wall_time_ns / 1_000_000_000.0
    for key in (
        "received_monotonic_s",
        "monotonic_s",
        "time_s",
        "elapsed_s",
    ):
        value = _finite(row.get(key))
        if value is not None:
            return value
    return None


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip().replace("\r", " ").replace("\n", " ")
    if len(message) > 180:
        message = message[:177] + "..."
    return f"{type(exc).__name__}: {message}"


def _read_result(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _error_text(exc)
    if not isinstance(value, dict):
        return None, "result root is not an object"
    return value, None


def _read_trace(path: Path) -> TraceRead:
    trace = TraceRead(events=[])
    if not path.is_file():
        trace.read_error = "missing"
        return trace
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            while True:
                try:
                    line = stream.readline()
                except (OSError, EOFError, UnicodeError) as exc:
                    trace.read_error = _error_text(exc)
                    break
                if not line:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    trace.invalid_json_lines += 1
                    continue
                if not isinstance(row, dict):
                    trace.non_object_lines += 1
                    continue
                trace.events.append(row)
    except (OSError, EOFError, UnicodeError) as exc:
        trace.read_error = _error_text(exc)
    return trace


def _nested(
    value: Mapping[str, Any] | None,
    *keys: str,
) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _parse_ticks(events: Iterable[Mapping[str, Any]]) -> list[TickSample]:
    ticks: list[TickSample] = []
    for row in events:
        if row.get("event") != "tick":
            continue
        time_s = _event_time_s(row)
        raw_command = row.get("command")
        if time_s is None or not isinstance(raw_command, Mapping):
            continue
        command = {
            key: number
            for key in COMMAND_FIELDS
            if (number := _finite(raw_command.get(key))) is not None
        }
        if not command:
            continue
        ticks.append(
            TickSample(
                time_s=time_s,
                command=command,
                body_rates=_triple(row.get("body_rates")),
                rpy=_triple(row.get("rpy")),
                imu_accel=_triple(row.get("imu_accel")),
                gate_index=_integer(row.get("gate_index")),
            )
        )
    ticks.sort(key=lambda item: item.time_s)
    return ticks


def _track_ids_from_candidates(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        track_id
        for candidate in value
        if isinstance(candidate, Mapping)
        and isinstance((track_id := candidate.get("track_id")), str)
        and track_id
    }


def _parse_features(
    events: Iterable[Mapping[str, Any]],
) -> list[FeatureSample]:
    features: list[FeatureSample] = []
    for row in events:
        if row.get("event") != "visual_gate_graph_frame":
            continue
        time_s = _event_time_s(row)
        graph = row.get("graph")
        tracks = row.get("tracks")
        if (
            time_s is None
            or not isinstance(graph, Mapping)
            or not isinstance(tracks, list)
        ):
            continue
        gate_index = _integer(graph.get("current_gate_index"))
        if gate_index is None:
            gate_index = _integer(row.get("gate_index"))
        if gate_index is None or gate_index < 0:
            continue
        current_track_id = graph.get("current_track_id")
        if not isinstance(current_track_id, str):
            current_track_id = None
        next_track_ids = _track_ids_from_candidates(
            graph.get("next_candidates")
        )
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            track_id = track.get("track_id")
            if not isinstance(track_id, str) or not track_id:
                continue
            role = track.get("role")
            if track_id == current_track_id:
                role = "current"
            elif track_id in next_track_ids:
                role = "successor"
            elif role == "next":
                role = "successor"
            if role not in {"current", "successor"}:
                continue
            center = track.get("center_norm_image_down")
            x = y_down = None
            if isinstance(center, (list, tuple)) and len(center) == 2:
                x = _finite(center[0])
                y_down = _finite(center[1])
            scale = _finite(track.get("apparent_scale"))
            log_scale = (
                math.log(scale) if scale is not None and scale > 0.0 else None
            )
            velocity = track.get("center_velocity_norm_s_image_down")
            x_rate = y_rate = None
            if isinstance(velocity, (list, tuple)) and len(velocity) == 2:
                x_rate = _finite(velocity[0])
                y_rate = _finite(velocity[1])
            clipping_edges = _integer(track.get("clipping_edges")) or 0
            features.append(
                FeatureSample(
                    time_s=time_s,
                    gate_index=gate_index,
                    role=role,
                    track_id=track_id,
                    x=x,
                    y_down=y_down,
                    log_scale=log_scale,
                    reported_x_rate=x_rate,
                    reported_y_rate_down=y_rate,
                    reported_log_scale_rate=_finite(
                        track.get("log_scale_rate_s")
                    ),
                    visible=track.get("visible") is not False,
                    censored=bool(track.get("center_censored"))
                    or clipping_edges != 0,
                    ambiguous=bool(track.get("ambiguous")),
                )
            )
    features.sort(key=lambda item: item.time_s)
    return features


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return None
    if len(finite_values) == 1:
        return finite_values[0]
    position = (len(finite_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite_values[lower]
    weight = position - lower
    return (
        finite_values[lower] * (1.0 - weight)
        + finite_values[upper] * weight
    )


def _linear_fit(
    xs: Sequence[float],
    ys: Sequence[float],
) -> tuple[float, float, float] | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    variance_x = sum((value - mean_x) ** 2 for value in xs)
    variance_y = sum((value - mean_y) ** 2 for value in ys)
    if variance_x <= 1e-18:
        return None
    covariance = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys)
    )
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    correlation = (
        covariance / math.sqrt(variance_x * variance_y)
        if variance_y > 1e-18
        else 0.0
    )
    return slope, intercept, max(-1.0, min(1.0, correlation))


def _scalar_trend(samples: Sequence[TimedValue]) -> dict[str, Any]:
    ordered = sorted(samples, key=lambda item: item.time_s)
    if not ordered:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "delta": None,
            "slope_per_s": None,
        }
    relative_times = [
        sample.time_s - ordered[0].time_s for sample in ordered
    ]
    values = [sample.value for sample in ordered]
    fit = _linear_fit(relative_times, values)
    return {
        "count": len(ordered),
        "first": values[0],
        "last": values[-1],
        "delta": values[-1] - values[0],
        "slope_per_s": None if fit is None else fit[0],
    }


def _dominant_track(
    samples: Sequence[FeatureSample],
) -> list[FeatureSample]:
    counts = Counter(
        sample.track_id
        for sample in samples
        if sample.visible
        and sample.x is not None
        and sample.y_down is not None
        and sample.log_scale is not None
    )
    if not counts:
        return []
    track_id = sorted(
        counts,
        key=lambda item: (-counts[item], item),
    )[0]
    return sorted(
        (sample for sample in samples if sample.track_id == track_id),
        key=lambda item: item.time_s,
    )


def _local_rate_series(
    samples: Sequence[FeatureSample],
    field: str,
) -> list[TimedValue]:
    usable = [
        sample
        for sample in samples
        if sample.visible
        and not sample.ambiguous
        and _finite(getattr(sample, field)) is not None
    ]
    usable.sort(key=lambda item: item.time_s)
    rates: list[TimedValue] = []
    for index, sample in enumerate(usable):
        neighbors = [
            candidate
            for candidate in usable[max(0, index - 2) : index + 3]
            if abs(candidate.time_s - sample.time_s) <= 0.12
        ]
        if len(neighbors) < 2:
            continue
        times = [candidate.time_s - sample.time_s for candidate in neighbors]
        values = [
            float(getattr(candidate, field)) for candidate in neighbors
        ]
        fit = _linear_fit(times, values)
        if fit is not None:
            rates.append(TimedValue(sample.time_s, fit[0]))
    return rates


def _feature_summary(samples: Sequence[FeatureSample]) -> dict[str, Any]:
    dominant = _dominant_track(samples)
    if not dominant:
        return {
            "track_id": None,
            "observation_count": 0,
            "usable_count": 0,
            "duration_s": None,
            "bearing_x": _scalar_trend([]),
            "bearing_y_down": _scalar_trend([]),
            "log_scale": _scalar_trend([]),
            "closure": {
                "expansion_rate_log_s": None,
                "time_to_contact_s": None,
            },
            "censored_fraction": None,
            "ambiguous_fraction": None,
        }
    usable = [
        sample
        for sample in dominant
        if sample.visible and not sample.ambiguous
    ]
    x_values = [
        TimedValue(sample.time_s, sample.x)
        for sample in usable
        if sample.x is not None
    ]
    y_values = [
        TimedValue(sample.time_s, sample.y_down)
        for sample in usable
        if sample.y_down is not None
    ]
    scale_values = [
        TimedValue(sample.time_s, sample.log_scale)
        for sample in usable
        if sample.log_scale is not None
    ]
    scale_rates = _local_rate_series(dominant, "log_scale")
    expansion_rate = _percentile(
        [sample.value for sample in scale_rates],
        0.5,
    )
    time_to_contact = (
        1.0 / expansion_rate
        if expansion_rate is not None and expansion_rate > 0.03
        else None
    )
    duration = (
        dominant[-1].time_s - dominant[0].time_s
        if len(dominant) > 1
        else 0.0
    )
    return {
        "track_id": dominant[0].track_id,
        "observation_count": len(dominant),
        "usable_count": len(usable),
        "duration_s": duration,
        "bearing_x": _scalar_trend(x_values),
        "bearing_y_down": _scalar_trend(y_values),
        "log_scale": _scalar_trend(scale_values),
        "closure": {
            "expansion_rate_log_s": expansion_rate,
            "time_to_contact_s": time_to_contact,
            "peak_expansion_rate_log_s": _percentile(
                [sample.value for sample in scale_rates],
                0.95,
            ),
        },
        "censored_fraction": (
            sum(sample.censored for sample in dominant) / len(dominant)
        ),
        "ambiguous_fraction": (
            sum(sample.ambiguous for sample in dominant) / len(dominant)
        ),
    }


def _command_metrics(ticks: Sequence[TickSample]) -> dict[str, Any]:
    periods = [
        newer.time_s - older.time_s
        for older, newer in zip(ticks, ticks[1:])
        if newer.time_s > older.time_s
    ]
    axes: dict[str, Any] = {}
    for field in COMMAND_FIELDS:
        values = [
            TimedValue(tick.time_s, tick.command[field])
            for tick in ticks
            if field in tick.command
        ]
        steps: list[float] = []
        slews: list[float] = []
        for older, newer in zip(values, values[1:]):
            elapsed = newer.time_s - older.time_s
            if elapsed <= 0.0:
                continue
            step = newer.value - older.value
            steps.append(abs(step))
            slews.append(abs(step) / elapsed)
        axes[field] = {
            "sample_count": len(values),
            "max_abs_command": (
                max((abs(value.value) for value in values), default=None)
            ),
            "max_abs_step": max(steps, default=None),
            "p95_abs_slew_per_s": _percentile(slews, 0.95),
            "max_abs_slew_per_s": max(slews, default=None),
        }
    roll_values = [
        TimedValue(tick.time_s, tick.command["roll_rate"])
        for tick in ticks
        if "roll_rate" in tick.command
    ]
    reversals = 0
    reversal_intervals: list[float] = []
    reversal_steps: list[float] = []
    prior: TimedValue | None = None
    for sample in roll_values:
        if abs(sample.value) < ROLL_REVERSAL_DEADBAND_RAD_S:
            continue
        if prior is not None and sample.value * prior.value < 0.0:
            reversals += 1
            reversal_intervals.append(sample.time_s - prior.time_s)
            reversal_steps.append(abs(sample.value - prior.value))
        prior = sample
    return {
        "tick_count": len(ticks),
        "tick_period_ms": {
            "median": (
                None
                if not periods
                else 1000.0 * float(statistics.median(periods))
            ),
            "p95": (
                None
                if not periods
                else 1000.0 * float(_percentile(periods, 0.95))
            ),
            "maximum": (
                None if not periods else 1000.0 * max(periods)
            ),
        },
        "axes": axes,
        "roll_reversals": {
            "deadband_rad_s": ROLL_REVERSAL_DEADBAND_RAD_S,
            "count": reversals,
            "minimum_interval_ms": (
                None
                if not reversal_intervals
                else 1000.0 * min(reversal_intervals)
            ),
            "maximum_reversal_step_rad_s": max(
                reversal_steps,
                default=None,
            ),
        },
    }


def _command_series(
    ticks: Sequence[TickSample],
    field: str,
) -> list[TimedValue]:
    return [
        TimedValue(tick.time_s, tick.command[field])
        for tick in ticks
        if field in tick.command
    ]


def _tick_response_series(
    ticks: Sequence[TickSample],
    source: str,
    index: int,
) -> list[TimedValue]:
    output: list[TimedValue] = []
    for tick in ticks:
        vector = getattr(tick, source)
        if vector is not None:
            output.append(TimedValue(tick.time_s, vector[index]))
    return output


def _held_value(
    series: Sequence[TimedValue],
    times: Sequence[float],
    query_time_s: float,
) -> TimedValue | None:
    # ``wall_time_ns`` becomes a large floating-point second value.  One
    # nanosecond of tolerance prevents an exact 20/30 ms history boundary from
    # selecting the preceding command due only to float cancellation.
    index = bisect_right(times, query_time_s + 1e-9) - 1
    if index < 0:
        return None
    sample = series[index]
    if query_time_s - sample.time_s > MAX_COMMAND_AGE_S:
        return None
    return sample


def estimate_delayed_effect(
    command: Sequence[TimedValue],
    response: Sequence[TimedValue],
    *,
    response_name: str,
) -> dict[str, Any]:
    """Fit a compact step-held command-to-response lag/gain correlation."""

    ordered_command = sorted(command, key=lambda sample: sample.time_s)
    ordered_response = sorted(response, key=lambda sample: sample.time_s)
    if len(ordered_command) < 2 or len(ordered_response) < MIN_EFFECT_SAMPLES:
        return {
            "status": "unavailable",
            "response": response_name,
            "reason": "insufficient samples",
            "sample_count": 0,
            "lag_ms": None,
            "gain": None,
            "correlation": None,
            "r_squared": None,
        }
    command_times = [sample.time_s for sample in ordered_command]
    candidates: list[dict[str, Any]] = []
    lag_count = round(MAX_EFFECT_LAG_S / EFFECT_LAG_STEP_S)
    for lag_index in range(lag_count + 1):
        lag_s = lag_index * EFFECT_LAG_STEP_S
        xs: list[float] = []
        ys: list[float] = []
        for response_sample in ordered_response:
            query_time = response_sample.time_s - lag_s
            command_sample = _held_value(
                ordered_command,
                command_times,
                query_time,
            )
            if command_sample is None:
                continue
            xs.append(command_sample.value)
            ys.append(response_sample.value)
        if len(xs) < MIN_EFFECT_SAMPLES:
            continue
        command_span = max(xs) - min(xs)
        if command_span <= 1e-6:
            continue
        fit = _linear_fit(xs, ys)
        if fit is None:
            continue
        gain, intercept, correlation = fit
        candidates.append(
            {
                "lag_s": lag_s,
                "gain": gain,
                "intercept": intercept,
                "correlation": correlation,
                "sample_count": len(xs),
                "command_span": command_span,
                "response_span": max(ys) - min(ys),
            }
        )
    if not candidates:
        return {
            "status": "unavailable",
            "response": response_name,
            "reason": "no varying command window",
            "sample_count": 0,
            "lag_ms": None,
            "gain": None,
            "correlation": None,
            "r_squared": None,
        }
    best_abs_correlation = max(
        abs(candidate["correlation"]) for candidate in candidates
    )
    # Autocorrelated commands often produce a broad delay plateau.  Select the
    # earliest fit within 0.02 correlation of the peak instead of claiming a
    # later, falsely precise delay.
    near_peak = [
        candidate
        for candidate in candidates
        if abs(candidate["correlation"]) >= best_abs_correlation - 0.02
    ]
    selected = min(near_peak, key=lambda candidate: candidate["lag_s"])
    ordered_near_peak = sorted(
        near_peak,
        key=lambda candidate: candidate["lag_s"],
    )
    selected_index = ordered_near_peak.index(selected)
    contiguous = [selected]
    for candidate in ordered_near_peak[selected_index + 1 :]:
        if (
            candidate["lag_s"] - contiguous[-1]["lag_s"]
            > EFFECT_LAG_STEP_S * 1.01
        ):
            break
        contiguous.append(candidate)
    correlation = float(selected["correlation"])
    return {
        "status": (
            "identified"
            if abs(correlation) >= MIN_EFFECT_CORRELATION
            else "weak"
        ),
        "response": response_name,
        "reason": (
            None
            if abs(correlation) >= MIN_EFFECT_CORRELATION
            else "absolute correlation below diagnostic threshold"
        ),
        "sample_count": int(selected["sample_count"]),
        "lag_ms": 1000.0 * float(selected["lag_s"]),
        "lag_range_ms": {
            "lower": 1000.0 * float(contiguous[0]["lag_s"]),
            "upper": 1000.0 * float(contiguous[-1]["lag_s"]),
        },
        "gain": float(selected["gain"]),
        "intercept": float(selected["intercept"]),
        "correlation": correlation,
        "r_squared": correlation * correlation,
        "command_span": float(selected["command_span"]),
        "response_span": float(selected["response_span"]),
        "lag_search": {
            "maximum_ms": 1000.0 * MAX_EFFECT_LAG_S,
            "step_ms": 1000.0 * EFFECT_LAG_STEP_S,
            "near_peak_tolerance": 0.02,
        },
    }


def _interpolate(
    series: Sequence[TimedValue],
    time_s: float,
) -> float | None:
    if not series:
        return None
    times = [sample.time_s for sample in series]
    right = bisect_right(times, time_s)
    if right == 0:
        sample = series[0]
        return (
            sample.value
            if abs(sample.time_s - time_s) <= MAX_INTERPOLATION_GAP_S
            else None
        )
    if right >= len(series):
        sample = series[-1]
        return (
            sample.value
            if abs(sample.time_s - time_s) <= MAX_INTERPOLATION_GAP_S
            else None
        )
    older = series[right - 1]
    newer = series[right]
    span = newer.time_s - older.time_s
    if span <= 0.0 or span > MAX_INTERPOLATION_GAP_S:
        return None
    blend = (time_s - older.time_s) / span
    return older.value + blend * (newer.value - older.value)


def _residual_horizontal_rates(
    samples: Sequence[FeatureSample],
    body_yaw_rates: Sequence[TimedValue],
    yaw_image_gain: float,
) -> tuple[list[TimedValue], dict[str, Any]]:
    raw_rates = _local_rate_series(samples, "x")
    residual: list[TimedValue] = []
    predicted_values: list[float] = []
    raw_values: list[float] = []
    outward_values: list[float] = []
    x_by_time = [
        TimedValue(sample.time_s, sample.x)
        for sample in samples
        if sample.visible and sample.x is not None
    ]
    for rate in raw_rates:
        body_yaw_rate = _interpolate(body_yaw_rates, rate.time_s)
        bearing_x = _interpolate(x_by_time, rate.time_s)
        if body_yaw_rate is None:
            continue
        predicted = yaw_image_gain * body_yaw_rate
        value = rate.value - predicted
        residual.append(TimedValue(rate.time_s, value))
        predicted_values.append(predicted)
        raw_values.append(rate.value)
        if bearing_x is not None and abs(bearing_x) > 1e-6:
            outward_values.append(math.copysign(value, bearing_x))
    values = [sample.value for sample in residual]
    return residual, {
        "sample_count": len(values),
        "yaw_image_gain_norm_per_rad": yaw_image_gain,
        "raw_bearing_rate_median_norm_s": _percentile(raw_values, 0.5),
        "predicted_rotation_rate_median_norm_s": _percentile(
            predicted_values,
            0.5,
        ),
        "residual_rate_median_norm_s": _percentile(values, 0.5),
        "residual_rate_p10_norm_s": _percentile(values, 0.10),
        "residual_rate_p90_norm_s": _percentile(values, 0.90),
        "residual_rate_rms_norm_s": (
            None
            if not values
            else math.sqrt(statistics.fmean(value * value for value in values))
        ),
        "outward_residual_rate_median_norm_s": _percentile(
            outward_values,
            0.5,
        ),
        "outward_residual_rate_p90_norm_s": _percentile(
            outward_values,
            0.90,
        ),
    }


def _feature_groups(
    features: Sequence[FeatureSample],
) -> dict[tuple[int, str], list[FeatureSample]]:
    grouped: dict[tuple[int, str], list[FeatureSample]] = defaultdict(list)
    for sample in features:
        grouped[(sample.gate_index, sample.role)].append(sample)
    return {
        key: _dominant_track(samples)
        for key, samples in grouped.items()
        if _dominant_track(samples)
    }


def _authoritative_progress(
    result: Mapping[str, Any] | None,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = _nested(result, "runner_result", "details", "visual_course")
    source = "result"
    if not isinstance(summary, Mapping):
        summary = None
        for row in reversed(events):
            if row.get("event") in {
                "visual_course_post_cleanup",
                "visual_course_complete",
                "visual_course_aborted",
            }:
                summary = row
                source = "trace_summary"
                break
    if not isinstance(summary, Mapping):
        summary = {}
        source = "unavailable"
    initial = _integer(summary.get("initial_gate_index"))
    final = _integer(summary.get("final_gate_index"))
    maximum = _integer(summary.get("maximum_authoritative_gate_index"))
    transitions_value = summary.get("authoritative_transitions")
    transitions = (
        transitions_value if isinstance(transitions_value, list) else []
    )
    sanitized_transitions: list[dict[str, Any]] = []
    for transition in transitions:
        if not isinstance(transition, Mapping):
            continue
        sanitized_transitions.append(
            {
                key: transition.get(key)
                for key in (
                    "from_gate_index",
                    "to_gate_index",
                    "promotion_confirmed",
                    "pre_transition_navigation_command_count",
                    "post_transition_navigation_command_count",
                )
                if key in transition
            }
        )
    if maximum is None:
        numeric = [
            value
            for value in (
                initial,
                final,
                *[
                    _integer(transition.get("to_gate_index"))
                    for transition in transitions
                    if isinstance(transition, Mapping)
                ],
            )
            if value is not None
        ]
        maximum = max(numeric, default=None)
    race_finished = summary.get("race_finished")
    if not isinstance(race_finished, bool):
        race_finished = False
    runner_cleanup = _nested(result, "runner_result", "cleanup_confirmed")
    cleanup_confirmed = (
        runner_cleanup if isinstance(runner_cleanup, bool) else None
    )
    if cleanup_confirmed is None:
        for row in reversed(events):
            if row.get("event") == "cleanup_complete":
                value = row.get("confirmed")
                if isinstance(value, bool):
                    cleanup_confirmed = value
                break
    return {
        "source": source,
        "initial_gate_index": initial,
        "final_gate_index": final,
        "maximum_gate_index": maximum,
        "transition_count": len(sanitized_transitions),
        "transitions": sanitized_transitions,
        "race_finished": race_finished,
        "cleanup_confirmed": cleanup_confirmed,
        "outcome": (
            summary.get("outcome")
            if isinstance(summary.get("outcome"), str)
            else None
        ),
    }


def _run_effects(
    ticks: Sequence[TickSample],
    focus_groups: Mapping[str, Sequence[FeatureSample]],
    yaw_image_gain: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body_roll = _tick_response_series(ticks, "body_rates", 0)
    body_pitch = _tick_response_series(ticks, "body_rates", 1)
    body_yaw = _tick_response_series(ticks, "body_rates", 2)
    accel_z = _tick_response_series(ticks, "imu_accel", 2)
    horizontal_samples = list(
        focus_groups.get("successor") or focus_groups.get("current") or []
    )
    current_samples = list(
        focus_groups.get("current") or horizontal_samples
    )
    x_rates = _local_rate_series(horizontal_samples, "x")
    y_rates = _local_rate_series(current_samples, "y_down")
    log_scale_rates = _local_rate_series(current_samples, "log_scale")
    residual_rates, residual_summary = _residual_horizontal_rates(
        horizontal_samples,
        body_yaw,
        yaw_image_gain,
    )
    response_map = {
        "yaw": {
            "command": _command_series(ticks, "yaw_rate"),
            "imu": (body_yaw, "body_yaw_rate_rad_s"),
            "image": (x_rates, "bearing_x_rate_norm_s"),
        },
        "roll": {
            "command": _command_series(ticks, "roll_rate"),
            "imu": (body_roll, "body_roll_rate_rad_s"),
            "image": (
                residual_rates,
                "yaw_derotated_bearing_x_rate_norm_s",
            ),
        },
        "pitch": {
            "command": _command_series(ticks, "pitch_rate"),
            "imu": (body_pitch, "body_pitch_rate_rad_s"),
            "image": (log_scale_rates, "log_scale_rate_s"),
        },
        "thrust": {
            "command": _command_series(ticks, "thrust"),
            "imu": (accel_z, "body_accel_z_m_s2"),
            "image": (y_rates, "bearing_y_down_rate_norm_s"),
        },
    }
    effects: dict[str, Any] = {}
    for channel, mapping in response_map.items():
        command = mapping["command"]
        imu_response, imu_name = mapping["imu"]
        image_response, image_name = mapping["image"]
        effects[channel] = {
            "imu": estimate_delayed_effect(
                command,
                imu_response,
                response_name=imu_name,
            ),
            "image": estimate_delayed_effect(
                command,
                image_response,
                response_name=image_name,
            ),
        }
    return effects, residual_summary


def analyze_run(
    run_directory: Path,
    *,
    yaw_image_gain: float = DEFAULT_YAW_IMAGE_GAIN_NORM_PER_RAD,
) -> dict[str, Any]:
    """Analyze one run directory without opening any artifact for writing."""

    result_path = run_directory / "result.json"
    trace_path = run_directory / "session.jsonl.gz"
    result, result_error = _read_result(result_path)
    trace = _read_trace(trace_path)
    ticks = _parse_ticks(trace.events)
    features = _parse_features(trace.events)
    groups = _feature_groups(features)
    gate_indices = sorted({gate_index for gate_index, _role in groups})
    progress = _authoritative_progress(result, trace.events)
    focus_gate = progress["maximum_gate_index"]
    if focus_gate not in gate_indices:
        focus_gate = max(gate_indices, default=focus_gate)
    gate_features: list[dict[str, Any]] = []
    for gate_index in gate_indices:
        gate_features.append(
            {
                "gate_index": gate_index,
                "current": _feature_summary(
                    groups.get((gate_index, "current"), [])
                ),
                "successor": _feature_summary(
                    groups.get((gate_index, "successor"), [])
                ),
            }
        )
    focus_groups = {
        role: groups.get((focus_gate, role), [])
        for role in ("current", "successor")
    }
    effects, residual = _run_effects(
        ticks,
        focus_groups,
        yaw_image_gain,
    )
    reason = result.get("reason") if isinstance(result, Mapping) else None
    if not isinstance(reason, str):
        for row in reversed(trace.events):
            if row.get("event") in {"stage_abort", "visual_course_aborted"}:
                candidate = row.get("reason") or row.get(
                    "first_causal_blocker"
                )
                if isinstance(candidate, str):
                    reason = candidate
                    break
    return {
        "run_id": (
            result.get("run_id")
            if isinstance(result, Mapping)
            and isinstance(result.get("run_id"), str)
            else run_directory.name
        ),
        "directory": str(run_directory.resolve()),
        "result": {
            "status": "complete" if result_error is None else "unavailable",
            "error": result_error,
            "stage": (
                result.get("stage")
                if isinstance(result, Mapping)
                and isinstance(result.get("stage"), str)
                else None
            ),
            "success": (
                result.get("success")
                if isinstance(result, Mapping)
                and isinstance(result.get("success"), bool)
                else None
            ),
            "reason": reason,
        },
        "trace": {
            "status": trace.status,
            "event_count": len(trace.events),
            "invalid_json_lines": trace.invalid_json_lines,
            "non_object_lines": trace.non_object_lines,
            "read_error": trace.read_error,
        },
        "authoritative_progress": progress,
        "focus_gate_index": focus_gate,
        "gate_features": gate_features,
        "commands": _command_metrics(ticks),
        "effects": effects,
        "residual_horizontal_motion": residual,
    }


def discover_run_directories(root: Path) -> list[Path]:
    """Return visual-course run directories, newest lexical run id first."""

    if not root.is_dir():
        return []
    try:
        directories = [
            path
            for path in root.glob("*-visual-course-*")
            if path.is_dir()
        ]
    except OSError:
        return []
    return sorted(directories, key=lambda path: path.name, reverse=True)


def _median_or_none(values: Iterable[Any]) -> float | None:
    usable = [
        number
        for value in values
        if (number := _finite(value)) is not None
    ]
    return float(statistics.median(usable)) if usable else None


def _focus_feature(run: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    focus_gate = run.get("focus_gate_index")
    gate_features = run.get("gate_features")
    if not isinstance(gate_features, list):
        return {}
    for gate in gate_features:
        if (
            isinstance(gate, Mapping)
            and gate.get("gate_index") == focus_gate
            and isinstance(gate.get(role), Mapping)
        ):
            return gate[role]
    return {}


def _aggregate_effect(
    runs: Sequence[Mapping[str, Any]],
    channel: str,
    domain: str,
) -> dict[str, Any]:
    estimates: list[Mapping[str, Any]] = []
    for run in runs:
        estimate = _nested(run, "effects", channel, domain)
        if (
            isinstance(estimate, Mapping)
            and estimate.get("status") in {"identified", "weak"}
            and _finite(estimate.get("lag_ms")) is not None
        ):
            estimates.append(estimate)
    identified = [
        estimate
        for estimate in estimates
        if estimate.get("status") == "identified"
    ]
    selected = identified or estimates
    return {
        "run_count": len(estimates),
        "identified_run_count": len(identified),
        "status": (
            "identified"
            if identified
            else ("weak" if estimates else "unavailable")
        ),
        "response": (
            selected[0].get("response") if selected else None
        ),
        "median_lag_ms": _median_or_none(
            estimate.get("lag_ms") for estimate in selected
        ),
        "median_lag_range_ms": {
            "lower": _median_or_none(
                _nested(estimate, "lag_range_ms", "lower")
                for estimate in selected
            ),
            "upper": _median_or_none(
                _nested(estimate, "lag_range_ms", "upper")
                for estimate in selected
            ),
        },
        "median_gain": _median_or_none(
            estimate.get("gain") for estimate in selected
        ),
        "median_correlation": _median_or_none(
            estimate.get("correlation") for estimate in selected
        ),
        "median_r_squared": _median_or_none(
            estimate.get("r_squared") for estimate in selected
        ),
    }


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    highest_gate_values = [
        value
        for run in runs
        if (
            value := _integer(
                _nested(run, "authoritative_progress", "maximum_gate_index")
            )
        )
        is not None
    ]
    effects = {
        channel: {
            domain: _aggregate_effect(runs, channel, domain)
            for domain in ("imu", "image")
        }
        for channel in ("yaw", "roll", "pitch", "thrust")
    }
    return {
        "run_count": len(runs),
        "complete_result_count": sum(
            _nested(run, "result", "status") == "complete" for run in runs
        ),
        "complete_trace_count": sum(
            _nested(run, "trace", "status") == "complete" for run in runs
        ),
        "partial_or_missing_trace_count": sum(
            _nested(run, "trace", "status") != "complete" for run in runs
        ),
        "cleanup_confirmed_count": sum(
            _nested(
                run,
                "authoritative_progress",
                "cleanup_confirmed",
            )
            is True
            for run in runs
        ),
        "highest_authoritative_gate_index": max(
            highest_gate_values,
            default=None,
        ),
        "race_finished_run_count": sum(
            _nested(run, "authoritative_progress", "race_finished") is True
            for run in runs
        ),
        "total_authoritative_transitions": sum(
            _integer(
                _nested(
                    run,
                    "authoritative_progress",
                    "transition_count",
                )
            )
            or 0
            for run in runs
        ),
        "total_ticks": sum(
            _integer(_nested(run, "commands", "tick_count")) or 0
            for run in runs
        ),
        "total_roll_reversals": sum(
            _integer(
                _nested(run, "commands", "roll_reversals", "count")
            )
            or 0
            for run in runs
        ),
        "median_focus_current_bearing_x_slope_norm_s": _median_or_none(
            _nested(
                _focus_feature(run, "current"),
                "bearing_x",
                "slope_per_s",
            )
            for run in runs
        ),
        "median_focus_successor_bearing_x_slope_norm_s": _median_or_none(
            _nested(
                _focus_feature(run, "successor"),
                "bearing_x",
                "slope_per_s",
            )
            for run in runs
        ),
        "median_focus_current_expansion_rate_log_s": _median_or_none(
            _nested(
                _focus_feature(run, "current"),
                "closure",
                "expansion_rate_log_s",
            )
            for run in runs
        ),
        "median_residual_horizontal_rate_norm_s": _median_or_none(
            _nested(
                run,
                "residual_horizontal_motion",
                "residual_rate_median_norm_s",
            )
            for run in runs
        ),
        "median_outward_residual_horizontal_rate_norm_s": _median_or_none(
            _nested(
                run,
                "residual_horizontal_motion",
                "outward_residual_rate_median_norm_s",
            )
            for run in runs
        ),
        "effects": effects,
    }


def analyze_cohort(
    root: Path,
    *,
    limit: int = 0,
    yaw_image_gain: float = DEFAULT_YAW_IMAGE_GAIN_NORM_PER_RAD,
) -> dict[str, Any]:
    """Analyze the discovered cohort and return a JSON-safe report."""

    if not math.isfinite(yaw_image_gain) or not 0.05 <= yaw_image_gain <= 2.0:
        raise ValueError("yaw image gain must be finite and in [0.05, 2.0]")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    resolved_root = root.expanduser().resolve()
    root_status = "available" if resolved_root.is_dir() else "missing"
    directories = discover_run_directories(resolved_root)
    if limit:
        directories = directories[:limit]
    runs = [
        analyze_run(directory, yaw_image_gain=yaw_image_gain)
        for directory in directories
    ]
    return {
        "schema": REPORT_SCHEMA,
        "evidence_root": str(resolved_root),
        "evidence_root_status": root_status,
        "selection": {
            "stage": "visual-course",
            "run_directory_pattern": "*-visual-course-*",
            "limit": limit,
            "selected_run_count": len(runs),
        },
        "assumptions": {
            "read_only": True,
            "authoritative_progress_source": (
                "runner result or compact visual-course terminal event only"
            ),
            "camera_geometry": (
                "normalized image-down centers; no pose or gate-map stream"
            ),
            "yaw_image_gain_norm_per_rad": yaw_image_gain,
            "yaw_image_gain_basis": (
                "approximate build-3385 paired yaw/image evidence; "
                "override explicitly when a newer calibration is accepted"
            ),
            "delay_fit": (
                "step-held command correlation over 0-200 ms; earliest lag "
                "within 0.02 correlation of the peak"
            ),
            "effect_limit": (
                "closed-loop, correlated course inputs are confounded; weak "
                "or identified labels are diagnostic, not causal authority"
            ),
            "thrust_imu_response": (
                "body-frame HIGHRES_IMU z acceleration, including gravity "
                "and attitude coupling"
            ),
            "closure": (
                "positive local log-apparent-scale rate; TTC is its reciprocal"
            ),
        },
        "aggregate": _aggregate_runs(runs),
        "runs": runs,
    }


def _format_number(value: Any, digits: int = 3) -> str:
    number = _finite(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _truncate(value: Any, width: int) -> str:
    text = "-" if value is None else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _format_effect_row(
    channel: str,
    domain: str,
    value: Mapping[str, Any],
) -> str:
    return (
        f"{channel:<7} {domain:<5} "
        f"{value.get('status', '-'):>11} "
        f"{_format_number(value.get('median_lag_ms'), 1):>8} "
        f"{_format_number(value.get('median_gain'), 3):>10} "
        f"{_format_number(value.get('median_correlation'), 3):>7} "
        f"{str(value.get('identified_run_count', 0)):>5}/"
        f"{str(value.get('run_count', 0)):<5} "
        f"{_truncate(value.get('response'), 34)}"
    )


def format_table(report: Mapping[str, Any], *, max_runs: int = 20) -> str:
    """Render a concise human table while retaining JSON as the full format."""

    aggregate = report.get("aggregate")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    runs_value = report.get("runs")
    runs = runs_value if isinstance(runs_value, list) else []
    lines = [
        "VQ2 visual-course compact cohort (read-only)",
        f"root: {report.get('evidence_root')} "
        f"[{report.get('evidence_root_status')}]",
        (
            "runs={run_count} complete_traces={complete_trace_count} "
            "partial/missing={partial_or_missing_trace_count} "
            "cleanup={cleanup_confirmed_count} highest_gate={gate} "
            "race_finished={race_finished_run_count}"
        ).format(
            gate=aggregate.get("highest_authoritative_gate_index", "-"),
            **{
                key: aggregate.get(key, 0)
                for key in (
                    "run_count",
                    "complete_trace_count",
                    "partial_or_missing_trace_count",
                    "cleanup_confirmed_count",
                    "race_finished_run_count",
                )
            },
        ),
        (
            "ticks={ticks} roll_reversals={reversals} "
            "median_residual_x={residual} norm/s "
            "median_outward_residual_x={outward} norm/s"
        ).format(
            ticks=aggregate.get("total_ticks", 0),
            reversals=aggregate.get("total_roll_reversals", 0),
            residual=_format_number(
                aggregate.get("median_residual_horizontal_rate_norm_s")
            ),
            outward=_format_number(
                aggregate.get(
                    "median_outward_residual_horizontal_rate_norm_s"
                )
            ),
        ),
        "",
        (
            "run_id                                     gate fin clean ticks "
            "rev roll_slew cur_x/s next_x/s close/s trace reason"
        ),
    ]
    selected_runs = runs if max_runs == 0 else runs[:max_runs]
    for run in selected_runs:
        if not isinstance(run, Mapping):
            continue
        progress = run.get("authoritative_progress")
        progress = progress if isinstance(progress, Mapping) else {}
        commands = run.get("commands")
        commands = commands if isinstance(commands, Mapping) else {}
        current = _focus_feature(run, "current")
        successor = _focus_feature(run, "successor")
        lines.append(
            f"{_truncate(run.get('run_id'), 42):<42} "
            f"{str(progress.get('maximum_gate_index', '-')):>4} "
            f"{('Y' if progress.get('race_finished') is True else 'N'):>3} "
            f"{('Y' if progress.get('cleanup_confirmed') is True else 'N'):>5} "
            f"{str(commands.get('tick_count', 0)):>5} "
            f"{str(_nested(commands, 'roll_reversals', 'count') or 0):>3} "
            f"{_format_number(_nested(commands, 'axes', 'roll_rate', 'max_abs_slew_per_s'), 2):>9} "
            f"{_format_number(_nested(current, 'bearing_x', 'slope_per_s'), 2):>7} "
            f"{_format_number(_nested(successor, 'bearing_x', 'slope_per_s'), 2):>8} "
            f"{_format_number(_nested(current, 'closure', 'expansion_rate_log_s'), 2):>7} "
            f"{_truncate(_nested(run, 'trace', 'status'), 7):<7} "
            f"{_truncate(_nested(run, 'result', 'reason'), 44)}"
        )
    if len(selected_runs) < len(runs):
        lines.append(
            f"... {len(runs) - len(selected_runs)} additional runs are "
            "included in aggregate/JSON output"
        )
    lines.extend(
        [
            "",
            (
                "channel domain      status lag_ms       gain    corr "
                " identified/used response"
            ),
        ]
    )
    effects = aggregate.get("effects")
    if isinstance(effects, Mapping):
        for channel in ("yaw", "roll", "pitch", "thrust"):
            domains = effects.get(channel)
            if not isinstance(domains, Mapping):
                continue
            for domain in ("imu", "image"):
                value = domains.get(domain)
                if isinstance(value, Mapping):
                    lines.append(_format_effect_row(channel, domain, value))
    lines.extend(
        [
            "",
            (
                "Effect fits are closed-loop diagnostic correlations. "
                "Weak/confounded fits do not establish plant authority."
            ),
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read compact external visual-course evidence and report "
            "authoritative progress plus delay/effect diagnostics."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "fast-flight-cycles evidence root "
            "(default: AIGP_EVIDENCE_ROOT or user profile)"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="analyze only the newest N runs; zero means the full cohort",
    )
    parser.add_argument(
        "--table-runs",
        type=int,
        default=20,
        help="show at most N per-run table rows; zero means all",
    )
    parser.add_argument(
        "--yaw-image-gain",
        type=float,
        default=DEFAULT_YAW_IMAGE_GAIN_NORM_PER_RAD,
        help=(
            "normalized-x rotational rate per measured yaw rad/s "
            f"(default: {DEFAULT_YAW_IMAGE_GAIN_NORM_PER_RAD})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    if parsed.limit < 0 or parsed.table_runs < 0:
        print("limits must be nonnegative", file=sys.stderr)
        return 2
    root = parsed.root if parsed.root is not None else default_evidence_root()
    try:
        report = analyze_cohort(
            root,
            limit=parsed.limit,
            yaw_image_gain=parsed.yaw_image_gain,
        )
    except (OSError, ValueError) as exc:
        print(_error_text(exc), file=sys.stderr)
        return 2
    if parsed.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(format_table(report, max_runs=parsed.table_runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
