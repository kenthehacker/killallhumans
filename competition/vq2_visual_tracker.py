"""Deterministic multi-target image-space tracking for VQ2 build 3385.

The tracker consumes every eligible detector result from one exact camera
publication.  It deliberately has no race, controller, transport, pose, or
metric-map authority.  Geometry is limited to normalized image coordinates,
normalized support boxes, and rates derived from a caller-named monotonic time
basis.

Association is global and bounded.  Pair costs combine predicted center
motion, support overlap, width/height/log-area change, confidence continuity,
edge-fragment continuity, temporal continuity, and optional appearance
features.  Near-tied assignments remain visible but are marked ambiguous so a
navigation layer can fail closed instead of silently treating detector order or
the largest contour as semantic gate identity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Optional, Sequence

from competition.vq2_contracts import FrameEdge


_UINT32_MAX = 2**32 - 1
_NS_PER_S = 1_000_000_000.0


class VisualTrackRole(str, Enum):
    """Camera-derived lifecycle role; only ``CURRENT`` may carry gate credit."""

    CURRENT = "current"
    NEXT = "next"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    RETIRED = "retired"


class FrameProvenanceBasis(str, Enum):
    """Distinguish live receiver timing from older exact capture evidence."""

    RECEIVER_TIMING_V1 = "receiver_timing_v1"
    LEGACY_CAPTURE = "legacy_capture"


class StaleVisualFrameError(ValueError):
    """The supplied camera publication cannot advance tracker state."""


@dataclass(frozen=True, slots=True)
class CameraFrameToken:
    """Exact build-3385 publication identity used by the production runner.

    The underlying camera-frame identity remains ``(generation, frame_id)``.
    ``publication_sequence`` binds that identity to the receiver publication
    ledger and must strictly advance inside one generation.
    """

    generation: int
    frame_id: int
    publication_sequence: Optional[int] = None
    stream_id: Optional[str] = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.generation, "generation")
        _nonnegative_int(self.frame_id, "frame_id", maximum=_UINT32_MAX)
        if self.publication_sequence is not None:
            _positive_int(self.publication_sequence, "publication_sequence")
        if self.stream_id is not None and (
            type(self.stream_id) is not str or not self.stream_id
        ):
            raise TypeError("stream_id must be a non-empty exact string or None")

    @property
    def exact_tuple(self) -> tuple[int, int]:
        return self.generation, self.frame_id

    @property
    def publication_tuple(self) -> Optional[tuple[int, int, int]]:
        if self.publication_sequence is None:
            return None
        return self.generation, self.frame_id, self.publication_sequence

    @property
    def live_identity_tuple(self) -> Optional[tuple[str, int, int, int]]:
        if self.stream_id is None or self.publication_sequence is None:
            return None
        return (
            self.stream_id,
            self.generation,
            self.frame_id,
            self.publication_sequence,
        )

    @classmethod
    def from_vision_snapshot(cls, snapshot: Any) -> "CameraFrameToken":
        """Copy and cross-check exact identity from ``VQ2VisionSnapshot.timing``."""

        timing = getattr(snapshot, "timing", None)
        if timing is None:
            raise ValueError("vision snapshot lacks exact FrameTimingV1 provenance")
        identity = getattr(timing, "identity", None)
        generation = _nonnegative_int(
            getattr(snapshot, "generation", None),
            "snapshot.generation",
        )
        frame_id = _nonnegative_int(
            getattr(snapshot, "frame_id", None),
            "snapshot.frame_id",
            maximum=_UINT32_MAX,
        )
        if (
            getattr(identity, "generation", None) != generation
            or getattr(identity, "frame_id", None) != frame_id
        ):
            raise ValueError("snapshot identity disagrees with timing identity")
        return cls(
            generation=generation,
            frame_id=frame_id,
            publication_sequence=_positive_int(
                getattr(timing, "publication_sequence", None),
                "timing.publication_sequence",
            ),
            stream_id=_nonempty_string(
                getattr(identity, "stream_id", None),
                "timing.identity.stream_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class VisualInnerApertureGeometry:
    """One detector-co-timed inner-aperture fit in image coordinates.

    The tracker carries this value as observation evidence only.  Association
    continues to use the detector's outer support center and bounding box.
    A rejected fit may be represented with all four measurement fields set to
    ``None`` and a non-empty ``health_reason``.
    """

    center_norm: Optional[tuple[float, float]]
    half_size_norm: Optional[tuple[float, float]]
    log_scale: Optional[float]
    measurement_std: Optional[tuple[float, float, float]]
    confidence: float
    clipping: FrameEdge
    visible_edges: FrameEdge
    geometry_model_id: Optional[str]
    covariance_model_id: Optional[str]
    health_reason: Optional[str] = None

    def __post_init__(self) -> None:
        confidence = _finite(
            self.confidence,
            "confidence",
            minimum=0.0,
            maximum=1.0,
        )
        if type(self.clipping) is not FrameEdge:
            raise TypeError("clipping must be an exact FrameEdge")
        if type(self.visible_edges) is not FrameEdge:
            raise TypeError("visible_edges must be an exact FrameEdge")
        for name in ("geometry_model_id", "covariance_model_id", "health_reason"):
            value = getattr(self, name)
            if value is not None:
                _nonempty_string(value, name)

        measurement_fields = (
            self.center_norm,
            self.half_size_norm,
            self.log_scale,
            self.measurement_std,
        )
        fitted = all(value is not None for value in measurement_fields)
        if fitted != any(value is not None for value in measurement_fields):
            raise ValueError(
                "inner-aperture measurement fields must be all present or all absent"
            )
        if fitted:
            assert self.center_norm is not None
            assert self.half_size_norm is not None
            assert self.log_scale is not None
            assert self.measurement_std is not None
            center = _finite_pair(
                self.center_norm,
                "center_norm",
                minimum=-2.5,
                maximum=2.5,
            )
            half_size = _finite_pair(
                self.half_size_norm,
                "half_size_norm",
                minimum=0.0,
                maximum=2.0,
            )
            if any(value <= 0.0 for value in half_size):
                raise ValueError("half_size_norm values must be positive")
            log_scale = _finite(self.log_scale, "log_scale")
            if abs(log_scale) > 12.0:
                raise ValueError("log_scale must remain within +/-12")
            if (
                type(self.measurement_std) is not tuple
                or len(self.measurement_std) != 3
            ):
                raise TypeError("measurement_std must be an exact 3-tuple")
            measurement_std = tuple(
                _finite(
                    value,
                    f"measurement_std[{index}]",
                    strictly_positive=True,
                )
                for index, value in enumerate(self.measurement_std)
            )
            if self.geometry_model_id is None:
                raise ValueError("fitted geometry requires geometry_model_id")
            if self.covariance_model_id is None:
                raise ValueError("fitted geometry requires covariance_model_id")
            object.__setattr__(self, "center_norm", center)
            object.__setattr__(self, "half_size_norm", half_size)
            object.__setattr__(self, "log_scale", log_scale)
            object.__setattr__(self, "measurement_std", measurement_std)
        else:
            if self.geometry_model_id is not None:
                raise ValueError("rejected geometry cannot name geometry_model_id")
            if self.covariance_model_id is not None:
                raise ValueError("rejected geometry cannot name covariance_model_id")
            if self.health_reason is None:
                raise ValueError("rejected geometry requires health_reason")
        object.__setattr__(self, "confidence", confidence)

    @property
    def fitted(self) -> bool:
        return self.center_norm is not None

    @property
    def complete_visibility(self) -> bool:
        return self.visible_edges == (
            FrameEdge.LEFT
            | FrameEdge.TOP
            | FrameEdge.RIGHT
            | FrameEdge.BOTTOM
        )

    @property
    def passage_usable(self) -> bool:
        """Whether the fit meets the existing conservative passage semantics."""

        return bool(
            self.fitted
            and self.clipping == FrameEdge.NONE
            and self.complete_visibility
            and self.confidence >= 0.25
            and self.health_reason is None
        )


@dataclass(frozen=True, slots=True)
class VisualDetection:
    """One eligible detector result represented only in image space.

    ``bbox_norm`` is ``(left, top, right, bottom)`` in the unit image square.
    ``center_norm`` follows the existing VQ2 convention: right and down are
    positive in ``[-1, 1]``.  Navigation-facing elevation properties invert
    the second coordinate so upward elevation is positive.
    """

    source_index: int
    center_norm: tuple[float, float]
    bbox_norm: tuple[float, float, float, float]
    confidence: float
    clipping: FrameEdge = FrameEdge.NONE
    center_censored: bool = False
    detection_method: str = "vq2_red_gate"
    appearance: Optional[tuple[float, ...]] = None
    inner_aperture: Optional[VisualInnerApertureGeometry] = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_index, "source_index")
        center_x, center_y = _finite_pair(
            self.center_norm,
            "center_norm",
            minimum=-1.0,
            maximum=1.0,
        )
        left, top, right, bottom = _bbox(self.bbox_norm, "bbox_norm")
        center_unit_x = 0.5 * (center_x + 1.0)
        center_unit_y = 0.5 * (center_y + 1.0)
        if not left <= center_unit_x <= right or not top <= center_unit_y <= bottom:
            raise ValueError("center_norm must lie inside bbox_norm")
        _finite(self.confidence, "confidence", minimum=0.0, maximum=1.0)
        if type(self.clipping) is not FrameEdge:
            raise TypeError("clipping must be an exact FrameEdge")
        if type(self.center_censored) is not bool:
            raise TypeError("center_censored must be an exact bool")
        if type(self.detection_method) is not str or not self.detection_method:
            raise TypeError("detection_method must be a non-empty exact string")
        if self.appearance is not None:
            if type(self.appearance) is not tuple or not self.appearance:
                raise TypeError("appearance must be a non-empty exact tuple or None")
            for index, value in enumerate(self.appearance):
                _finite(value, f"appearance[{index}]")
        if self.inner_aperture is not None and (
            type(self.inner_aperture) is not VisualInnerApertureGeometry
        ):
            raise TypeError(
                "inner_aperture must be an exact VisualInnerApertureGeometry or None"
            )

    @property
    def apparent_scale(self) -> float:
        left, top, right, bottom = self.bbox_norm
        return math.sqrt((right - left) * (bottom - top))

    @property
    def log_scale(self) -> float:
        return math.log(self.apparent_scale)

    @property
    def bearing_norm(self) -> float:
        return self.center_norm[0]

    @property
    def elevation_norm(self) -> float:
        return -self.center_norm[1]

    @classmethod
    def from_detector_result(
        cls,
        detection: Any,
        *,
        source_index: int,
        image_size_px: tuple[int, int],
        edge_margin_px: int = 2,
        appearance: Optional[tuple[float, ...]] = None,
        inner_aperture: Optional[VisualInnerApertureGeometry] = None,
    ) -> "VisualDetection":
        """Adapt one legacy ``GateDetection`` without using metric placeholders."""

        width_px, height_px = _image_size(image_size_px)
        _nonnegative_int(edge_margin_px, "edge_margin_px")
        bbox = getattr(detection, "bbox", None)
        if type(bbox) is not tuple or len(bbox) != 4:
            raise TypeError("detection.bbox must be an exact four-tuple")
        x, y, width, height = (
            _exact_int(value, f"detection.bbox[{index}]")
            for index, value in enumerate(bbox)
        )
        if (
            x < 0
            or y < 0
            or width < 1
            or height < 1
            or x + width > width_px
            or y + height > height_px
        ):
            raise ValueError("detection bbox must stay inside the image")
        center_x = _exact_int(getattr(detection, "center_x", None), "center_x")
        center_y = _exact_int(getattr(detection, "center_y", None), "center_y")
        if not x <= center_x <= x + width or not y <= center_y <= y + height:
            raise ValueError("detection center must lie inside its bbox")
        if not 0 <= center_x < width_px or not 0 <= center_y < height_px:
            raise ValueError("detection center must stay inside the image")
        clipping = FrameEdge.NONE
        if x <= edge_margin_px:
            clipping |= FrameEdge.LEFT
        if y <= edge_margin_px:
            clipping |= FrameEdge.TOP
        if x + width >= width_px - edge_margin_px:
            clipping |= FrameEdge.RIGHT
        if y + height >= height_px - edge_margin_px:
            clipping |= FrameEdge.BOTTOM
        confidence = _finite(
            getattr(detection, "confidence", None),
            "detection.confidence",
            minimum=0.0,
            maximum=1.0,
        )
        method = getattr(detection, "detection_method", "vq2_red_gate")
        if type(method) is not str or not method:
            method = "vq2_red_gate"
        return cls(
            source_index=source_index,
            center_norm=(
                2.0 * center_x / width_px - 1.0,
                2.0 * center_y / height_px - 1.0,
            ),
            bbox_norm=(
                x / width_px,
                y / height_px,
                (x + width) / width_px,
                (y + height) / height_px,
            ),
            confidence=confidence,
            clipping=clipping,
            # A clipped legacy bbox center is support geometry, not a recovered
            # aperture center.
            center_censored=clipping != FrameEdge.NONE,
            detection_method=method,
            appearance=appearance,
            inner_aperture=inner_aperture,
        )


@dataclass(frozen=True, slots=True)
class VisualDetectionFrame:
    """All eligible detections from one fresh camera publication."""

    token: CameraFrameToken
    provenance_basis: FrameProvenanceBasis
    time_basis_id: str
    image_size_px: tuple[int, int]
    detections: tuple[VisualDetection, ...]
    camera_source_time_ns: Optional[int] = None
    final_unique_packet_monotonic_ns: Optional[int] = None
    publish_monotonic_ns: Optional[int] = None
    legacy_received_monotonic_s: Optional[float] = None

    def __post_init__(self) -> None:
        if type(self.token) is not CameraFrameToken:
            raise TypeError("token must be an exact CameraFrameToken")
        if type(self.provenance_basis) is not FrameProvenanceBasis:
            raise TypeError("provenance_basis must be an exact FrameProvenanceBasis")
        if type(self.time_basis_id) is not str or not self.time_basis_id:
            raise TypeError("time_basis_id must be a non-empty exact string")
        _image_size(self.image_size_px)
        if type(self.detections) is not tuple:
            raise TypeError("detections must be an exact tuple")
        if any(type(item) is not VisualDetection for item in self.detections):
            raise TypeError("detections must contain exact VisualDetection values")
        source_indexes = tuple(item.source_index for item in self.detections)
        if len(source_indexes) != len(set(source_indexes)):
            raise ValueError("one frame cannot repeat a detection source_index")
        if self.camera_source_time_ns is not None:
            _nonnegative_int(self.camera_source_time_ns, "camera_source_time_ns")
        if self.provenance_basis is FrameProvenanceBasis.RECEIVER_TIMING_V1:
            if self.token.publication_sequence is None:
                raise ValueError("receiver timing requires a publication sequence")
            if self.token.stream_id is None:
                raise ValueError("receiver timing requires an exact camera stream_id")
            if (
                self.final_unique_packet_monotonic_ns is None
                or self.publish_monotonic_ns is None
            ):
                raise ValueError("receiver timing requires final-packet and publish times")
            _nonnegative_int(
                self.final_unique_packet_monotonic_ns,
                "final_unique_packet_monotonic_ns",
            )
            _nonnegative_int(self.publish_monotonic_ns, "publish_monotonic_ns")
            if self.publish_monotonic_ns < self.final_unique_packet_monotonic_ns:
                raise ValueError("publish time cannot predate the final unique packet")
            if self.legacy_received_monotonic_s is not None:
                raise ValueError("receiver timing cannot carry legacy receipt time")
        else:
            if self.token.publication_sequence is not None:
                raise ValueError("legacy capture cannot invent a publication sequence")
            if (
                self.final_unique_packet_monotonic_ns is not None
                or self.publish_monotonic_ns is not None
            ):
                raise ValueError("legacy capture cannot invent /1 packet/publish timing")
            if self.camera_source_time_ns is None:
                raise ValueError("legacy capture requires its opaque sim-time token")
            _finite(
                self.legacy_received_monotonic_s,
                "legacy_received_monotonic_s",
                minimum=0.0,
            )

    @property
    def observation_monotonic_ns(self) -> int:
        """Rate time basis: final unique packet proxy, not calibrated capture."""

        if self.final_unique_packet_monotonic_ns is not None:
            return self.final_unique_packet_monotonic_ns
        assert self.legacy_received_monotonic_s is not None
        return round(self.legacy_received_monotonic_s * _NS_PER_S)

    @classmethod
    def from_detector_results(
        cls,
        detections: Iterable[Any],
        *,
        generation: int,
        frame_id: int,
        publication_sequence: int,
        stream_id: str,
        final_unique_packet_monotonic_ns: int,
        publish_monotonic_ns: int,
        time_basis_id: str,
        image_size_px: tuple[int, int] = (640, 360),
        edge_margin_px: int = 2,
        appearances: Optional[Sequence[Optional[tuple[float, ...]]]] = None,
        aperture_geometries: Optional[
            Sequence[Optional[VisualInnerApertureGeometry]]
        ] = None,
        camera_source_time_ns: Optional[int] = None,
    ) -> "VisualDetectionFrame":
        """Adapt every supplied detector result, preserving its source index."""

        raw = tuple(detections)
        if appearances is not None and len(appearances) != len(raw):
            raise ValueError("appearances must match the number of detections")
        if (
            aperture_geometries is not None
            and len(aperture_geometries) != len(raw)
        ):
            raise ValueError(
                "aperture_geometries must match the number of detections"
            )
        adapted = tuple(
            VisualDetection.from_detector_result(
                detection,
                source_index=index,
                image_size_px=image_size_px,
                edge_margin_px=edge_margin_px,
                appearance=(None if appearances is None else appearances[index]),
                inner_aperture=(
                    None
                    if aperture_geometries is None
                    else aperture_geometries[index]
                ),
            )
            for index, detection in enumerate(raw)
        )
        return cls(
            token=CameraFrameToken(
                generation=generation,
                frame_id=frame_id,
                publication_sequence=publication_sequence,
                stream_id=stream_id,
            ),
            provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
            final_unique_packet_monotonic_ns=final_unique_packet_monotonic_ns,
            publish_monotonic_ns=publish_monotonic_ns,
            time_basis_id=time_basis_id,
            image_size_px=image_size_px,
            detections=adapted,
            camera_source_time_ns=camera_source_time_ns,
        )

    @classmethod
    def from_legacy_detector_results(
        cls,
        detections: Iterable[Any],
        *,
        generation: int,
        frame_id: int,
        camera_source_time_ns: int,
        received_monotonic_s: float,
        time_basis_id: str = "legacy-capture-monotonic-s",
        image_size_px: tuple[int, int] = (640, 360),
        edge_margin_px: int = 2,
        appearances: Optional[Sequence[Optional[tuple[float, ...]]]] = None,
        aperture_geometries: Optional[
            Sequence[Optional[VisualInnerApertureGeometry]]
        ] = None,
    ) -> "VisualDetectionFrame":
        """Ingest historical capture fields without relabelling them as /1 timing."""

        raw = tuple(detections)
        if appearances is not None and len(appearances) != len(raw):
            raise ValueError("appearances must match the number of detections")
        if (
            aperture_geometries is not None
            and len(aperture_geometries) != len(raw)
        ):
            raise ValueError(
                "aperture_geometries must match the number of detections"
            )
        adapted = tuple(
            VisualDetection.from_detector_result(
                detection,
                source_index=index,
                image_size_px=image_size_px,
                edge_margin_px=edge_margin_px,
                appearance=(None if appearances is None else appearances[index]),
                inner_aperture=(
                    None
                    if aperture_geometries is None
                    else aperture_geometries[index]
                ),
            )
            for index, detection in enumerate(raw)
        )
        return cls(
            token=CameraFrameToken(generation=generation, frame_id=frame_id),
            provenance_basis=FrameProvenanceBasis.LEGACY_CAPTURE,
            time_basis_id=time_basis_id,
            image_size_px=image_size_px,
            detections=adapted,
            camera_source_time_ns=camera_source_time_ns,
            legacy_received_monotonic_s=received_monotonic_s,
        )

    @classmethod
    def from_vision_snapshot(
        cls,
        snapshot: Any,
        detections: Iterable[Any],
        *,
        edge_margin_px: int = 2,
        appearances: Optional[Sequence[Optional[tuple[float, ...]]]] = None,
        aperture_geometries: Optional[
            Sequence[Optional[VisualInnerApertureGeometry]]
        ] = None,
    ) -> "VisualDetectionFrame":
        """Build a batch from the receiver's exact timing ledger."""

        timing = getattr(snapshot, "timing", None)
        if timing is None:
            raise ValueError("vision snapshot lacks exact FrameTimingV1 provenance")
        camera_frame = getattr(snapshot, "camera_frame", None)
        image = getattr(camera_frame, "image", None)
        shape = getattr(image, "shape", None)
        if len(shape) < 2 if type(shape) is tuple else True:
            raise TypeError("snapshot camera image lacks exact dimensions")
        token = CameraFrameToken.from_vision_snapshot(snapshot)
        time_basis_id = getattr(timing, "host_clock_id", None)
        if type(time_basis_id) is not str or not time_basis_id:
            raise TypeError("timing.host_clock_id must be a non-empty exact string")
        return cls.from_detector_results(
            detections,
            generation=token.generation,
            frame_id=token.frame_id,
            publication_sequence=token.publication_sequence,
            stream_id=token.stream_id,
            final_unique_packet_monotonic_ns=_nonnegative_int(
                getattr(timing, "final_unique_packet_monotonic_ns", None),
                "timing.final_unique_packet_monotonic_ns",
            ),
            publish_monotonic_ns=_nonnegative_int(
                getattr(timing, "publish_monotonic_ns", None),
                "timing.publish_monotonic_ns",
            ),
            time_basis_id=time_basis_id,
            image_size_px=(
                _positive_int(shape[1], "snapshot image width"),
                _positive_int(shape[0], "snapshot image height"),
            ),
            edge_margin_px=edge_margin_px,
            appearances=appearances,
            aperture_geometries=aperture_geometries,
            camera_source_time_ns=_nonnegative_int(
                getattr(timing, "camera_source_time_ns", None),
                "timing.camera_source_time_ns",
            ),
        )


@dataclass(frozen=True, slots=True)
class VisualTrackSample:
    tracker_frame_sequence: int
    token: CameraFrameToken
    observation_monotonic_ns: int
    publication_monotonic_ns: Optional[int]
    provenance_basis: FrameProvenanceBasis
    camera_source_time_ns: Optional[int]
    source_index: int
    center_norm: tuple[float, float]
    bbox_norm: tuple[float, float, float, float]
    apparent_scale: float
    confidence: float
    clipping: FrameEdge
    center_censored: bool
    association_confidence: float
    accepted_association: Optional["AssociationEvidence"] = None
    inner_aperture: Optional[VisualInnerApertureGeometry] = None

    @property
    def bearing_norm(self) -> float:
        return self.center_norm[0]

    @property
    def elevation_norm(self) -> float:
        return -self.center_norm[1]


@dataclass(frozen=True, slots=True)
class VisualTrack:
    """Immutable public view of one stable visual identity."""

    track_id: str
    first_token: CameraFrameToken
    latest_token: CameraFrameToken
    center_norm: tuple[float, float]
    bbox_norm: tuple[float, float, float, float]
    apparent_scale: float
    center_velocity_norm_s: tuple[float, float]
    log_scale_rate_s: float
    confidence: float
    association_confidence: float
    consecutive_frame_count: int
    total_observation_count: int
    missed_frame_count: int
    clipping: FrameEdge
    center_censored: bool
    role: VisualTrackRole
    authoritative_gate_index: Optional[int]
    authority_race_status_sequence: Optional[int]
    authority_race_status_boot_ms: Optional[int]
    ambiguous: bool
    visible: bool
    history: tuple[VisualTrackSample, ...]

    @property
    def bearing_norm(self) -> float:
        return self.center_norm[0]

    @property
    def elevation_norm(self) -> float:
        return -self.center_norm[1]

    @property
    def bearing_rate_norm_s(self) -> float:
        return self.center_velocity_norm_s[0]

    @property
    def elevation_rate_norm_s(self) -> float:
        return -self.center_velocity_norm_s[1]


@dataclass(frozen=True, slots=True)
class AssociationEvidence:
    """Compact, inspectable evidence for one accepted association."""

    track_id: str
    previous_token: CameraFrameToken
    current_token: CameraFrameToken
    detection_source_index: int
    cost: float
    confidence: float
    predicted_center_residual_norm: float
    bbox_iou: float
    log_width_change: float
    log_height_change: float
    log_area_residual: float
    clipping_continuity: float
    temporal_consistency: float
    appearance_distance: Optional[float]
    ambiguous: bool
    missed_frame_count_before_association: int
    observation_gap_ns: int
    publication_gap_ns: Optional[int]
    track_ambiguous_before_association: bool


def visual_track_history_sha256(
    history: tuple[VisualTrackSample, ...],
) -> str:
    """Canonically freeze one exact, untruncated visual identity history."""

    if (
        type(history) is not tuple
        or not history
        or any(type(sample) is not VisualTrackSample for sample in history)
    ):
        raise TypeError(
            "visual track history digest requires exact nonempty samples"
        )
    payload = {
        "schema": "aigp-vq2-visual-track-history/1",
        # Preserve historical /1 digests when the opt-in aperture fitter was
        # not active.  New geometry is included when it was actually observed.
        "samples": [
            {
                key: value
                for key, value in asdict(sample).items()
                if key != "inner_aperture" or value is not None
            }
            for sample in history
        ],
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "visual track history is not canonically hashable"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class VisualTrackerUpdate:
    """One deterministic tracker transition."""

    tracker_frame_sequence: int
    token: CameraFrameToken
    observation_monotonic_ns: int
    publish_monotonic_ns: Optional[int]
    provenance_basis: FrameProvenanceBasis
    tracks: tuple[VisualTrack, ...]
    visible_track_ids: tuple[str, ...]
    created_track_ids: tuple[str, ...]
    associated_track_ids: tuple[str, ...]
    missed_track_ids: tuple[str, ...]
    retired_track_ids: tuple[str, ...]
    ambiguous_track_ids: tuple[str, ...]
    associations: tuple[AssociationEvidence, ...]

    def track(self, track_id: str) -> VisualTrack:
        for item in self.tracks:
            if item.track_id == track_id:
                return item
        raise KeyError(track_id)

    @property
    def visible_tracks(self) -> tuple[VisualTrack, ...]:
        visible = set(self.visible_track_ids)
        return tuple(item for item in self.tracks if item.track_id in visible)


@dataclass(frozen=True, slots=True)
class MultiTargetTrackerConfig:
    """Image-space association bounds; these never authorize flight commands."""

    min_detection_confidence: float = 0.10
    max_association_gap_ns: int = 300_000_000
    max_missed_frames: int = 12
    max_center_residual_norm: float = 0.34
    max_log_width_change: float = 0.80
    max_log_height_change: float = 0.80
    max_log_area_residual: float = 1.30
    min_bbox_iou: float = 0.01
    fallback_center_residual_norm: float = 0.18
    clipped_fragment_relaxation: float = 1.45
    max_assignment_cost: float = 0.82
    unmatched_cost: float = 0.86
    ambiguity_margin: float = 0.08
    ambiguity_clear_frames: int = 2
    velocity_smoothing: float = 0.55
    confidence_smoothing: float = 0.55
    history_limit: int = 256
    center_cost_weight: float = 0.28
    overlap_cost_weight: float = 0.14
    size_cost_weight: float = 0.22
    confidence_cost_weight: float = 0.08
    clipping_cost_weight: float = 0.13
    temporal_cost_weight: float = 0.08
    appearance_cost_weight: float = 0.07

    def __post_init__(self) -> None:
        for name in (
            "min_detection_confidence",
            "min_bbox_iou",
            "max_assignment_cost",
            "unmatched_cost",
            "ambiguity_margin",
            "velocity_smoothing",
            "confidence_smoothing",
        ):
            _finite(getattr(self, name), name, minimum=0.0, maximum=1.0)
        for name in (
            "max_center_residual_norm",
            "max_log_width_change",
            "max_log_height_change",
            "max_log_area_residual",
            "fallback_center_residual_norm",
            "clipped_fragment_relaxation",
        ):
            _finite(getattr(self, name), name, minimum=0.0, strictly_positive=True)
        if self.clipped_fragment_relaxation < 1.0:
            raise ValueError("clipped_fragment_relaxation must be >= 1")
        for name in (
            "max_association_gap_ns",
            "max_missed_frames",
            "ambiguity_clear_frames",
            "history_limit",
        ):
            _positive_int(getattr(self, name), name)
        if self.unmatched_cost <= self.max_assignment_cost:
            raise ValueError("unmatched_cost must exceed max_assignment_cost")
        weights = tuple(
            _finite(getattr(self, name), name, minimum=0.0)
            for name in (
                "center_cost_weight",
                "overlap_cost_weight",
                "size_cost_weight",
                "confidence_cost_weight",
                "clipping_cost_weight",
                "temporal_cost_weight",
                "appearance_cost_weight",
            )
        )
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("association cost weights must sum to 1")


@dataclass(slots=True)
class _TrackState:
    track_id: str
    history: list[VisualTrackSample]
    center_velocity_norm_s: tuple[float, float]
    log_scale_rate_s: float
    confidence: float
    association_confidence: float
    consecutive_frame_count: int
    total_observation_count: int
    missed_frame_count: int
    nominal_role: VisualTrackRole
    authoritative_gate_index: Optional[int]
    authority_race_status_sequence: Optional[int]
    authority_race_status_boot_ms: Optional[int]
    ambiguous: bool
    unambiguous_streak: int
    retired: bool
    appearance: Optional[tuple[float, ...]]

    @property
    def latest(self) -> VisualTrackSample:
        return self.history[-1]


@dataclass(frozen=True, slots=True)
class _PairScore:
    cost: float
    center_residual: float
    bbox_iou: float
    log_width_change: float
    log_height_change: float
    log_area_residual: float
    clipping_continuity: float
    temporal_consistency: float
    appearance_distance: Optional[float]


@dataclass(frozen=True, slots=True)
class _AssignmentPlan:
    active_states: tuple[_TrackState, ...]
    eligible_detections: tuple[VisualDetection, ...]
    pair_scores: dict[tuple[int, int], _PairScore]
    selected_pairs: dict[int, int]
    ambiguous_track_indexes: frozenset[int]
    ambiguous_detection_indexes: frozenset[int]


class MultiTargetVisualTracker:
    """Stateful deterministic tracker with no I/O or command side effects."""

    def __init__(
        self,
        config: Optional[MultiTargetTrackerConfig] = None,
    ) -> None:
        if config is None:
            config = DEFAULT_MULTI_TARGET_TRACKER_CONFIG
        if type(config) is not MultiTargetTrackerConfig:
            raise TypeError("config must be an exact MultiTargetTrackerConfig")
        self.config = config
        self._states: dict[str, _TrackState] = {}
        self._next_track_number = 1
        self._generation: Optional[int] = None
        self._time_basis_id: Optional[str] = None
        self._provenance_basis: Optional[FrameProvenanceBasis] = None
        self._stream_id: Optional[str] = None
        self._image_size_px: Optional[tuple[int, int]] = None
        self._last_token: Optional[CameraFrameToken] = None
        self._last_observation_ns: Optional[int] = None
        self._last_publish_ns: Optional[int] = None
        self._last_camera_source_time_ns: Optional[int] = None
        self._frame_sequence = 0
        self._processed_frame_times: dict[CameraFrameToken, int] = {}
        self._processed_publish_times: dict[CameraFrameToken, Optional[int]] = {}
        self._processed_provenance: dict[
            CameraFrameToken, FrameProvenanceBasis
        ] = {}
        self._processed_frame_keys: set[tuple[int, int]] = set()
        self._latest_update: Optional[VisualTrackerUpdate] = None

    @property
    def latest_update(self) -> Optional[VisualTrackerUpdate]:
        return self._latest_update

    @property
    def generation(self) -> Optional[int]:
        return self._generation

    def reset_generation(self, generation: int) -> None:
        """Start a new camera reset generation without carrying track identity."""

        _nonnegative_int(generation, "generation")
        if self._generation is not None and generation <= self._generation:
            raise ValueError("reset generation must strictly increase")
        self._states.clear()
        self._generation = generation
        self._time_basis_id = None
        self._provenance_basis = None
        self._stream_id = None
        self._image_size_px = None
        self._last_token = None
        self._last_observation_ns = None
        self._last_publish_ns = None
        self._last_camera_source_time_ns = None
        self._processed_frame_times.clear()
        self._processed_publish_times.clear()
        self._processed_provenance.clear()
        self._processed_frame_keys.clear()
        self._latest_update = None
        # Track IDs remain lifetime-unique across reset generations.

    def has_processed_token(self, token: CameraFrameToken) -> bool:
        if type(token) is not CameraFrameToken:
            raise TypeError("token must be an exact CameraFrameToken")
        return token in self._processed_frame_times

    def frame_observation_time_ns(self, token: CameraFrameToken) -> int:
        if type(token) is not CameraFrameToken:
            raise TypeError("token must be an exact CameraFrameToken")
        try:
            return self._processed_frame_times[token]
        except KeyError as exc:
            raise KeyError(f"unknown camera frame token {token.exact_tuple}") from exc

    def frame_publish_time_ns(self, token: CameraFrameToken) -> Optional[int]:
        if type(token) is not CameraFrameToken:
            raise TypeError("token must be an exact CameraFrameToken")
        try:
            return self._processed_publish_times[token]
        except KeyError as exc:
            raise KeyError(f"unknown camera frame token {token.exact_tuple}") from exc

    def latest_processed_token_published_by(
        self,
        publish_monotonic_ns: int,
    ) -> Optional[CameraFrameToken]:
        """Return the exact processed camera watermark at one live time.

        The processed-frame registry includes fresh frames with zero eligible
        detections, unlike per-track histories. Live publication times are
        strictly increasing within a reset generation.
        """

        cutoff_ns = _nonnegative_int(
            publish_monotonic_ns,
            "publish_monotonic_ns",
        )
        eligible = tuple(
            (published, token)
            for token, published in self._processed_publish_times.items()
            if published is not None and published <= cutoff_ns
        )
        if not eligible:
            return None
        return max(eligible, key=lambda item: item[0])[1]

    def frame_provenance_basis(
        self,
        token: CameraFrameToken,
    ) -> FrameProvenanceBasis:
        if type(token) is not CameraFrameToken:
            raise TypeError("token must be an exact CameraFrameToken")
        try:
            return self._processed_provenance[token]
        except KeyError as exc:
            raise KeyError(f"unknown camera frame token {token.exact_tuple}") from exc

    @property
    def time_basis_id(self) -> Optional[str]:
        return self._time_basis_id

    def preview_associations(
        self,
        frame: VisualDetectionFrame,
    ) -> dict[int, VisualTrack]:
        """Preview exact unambiguous existing-track assignments without mutation.

        Keys retain the detector's source indexes rather than the filtered
        eligible-detection positions used internally.  New-track detections,
        unmatched detections, and every near-tied assignment are omitted.
        Calling :meth:`update` with the same still-unconsumed frame will use
        the same selected pairs.
        """

        if type(frame) is not VisualDetectionFrame:
            raise TypeError("frame must be an exact VisualDetectionFrame")
        plan = self._plan_assignments(frame)
        return {
            plan.eligible_detections[detection_index].source_index: (
                self._snapshot(plan.active_states[track_index])
            )
            for track_index, detection_index in plan.selected_pairs.items()
            if (
                track_index not in plan.ambiguous_track_indexes
                and detection_index not in plan.ambiguous_detection_indexes
            )
        }

    def update(self, frame: VisualDetectionFrame) -> VisualTrackerUpdate:
        """Consume one fresh frame and update every eligible visual detection."""

        if type(frame) is not VisualDetectionFrame:
            raise TypeError("frame must be an exact VisualDetectionFrame")
        plan = self._plan_assignments(frame)
        eligible = plan.eligible_detections
        active_states = plan.active_states
        pair_scores = plan.pair_scores
        selected_pairs = plan.selected_pairs
        ambiguous_track_indexes = plan.ambiguous_track_indexes
        ambiguous_detection_indexes = plan.ambiguous_detection_indexes

        self._frame_sequence += 1
        associated_ids: list[str] = []
        missed_ids: list[str] = []
        retired_ids: list[str] = []
        created_ids: list[str] = []
        association_evidence: list[AssociationEvidence] = []
        used_detection_indexes: set[int] = set()

        for track_index, state in enumerate(active_states):
            detection_index = selected_pairs.get(track_index)
            if detection_index is None:
                state.missed_frame_count += 1
                state.consecutive_frame_count = 0
                state.association_confidence = 0.0
                if track_index in ambiguous_track_indexes:
                    state.ambiguous = True
                    state.unambiguous_streak = 0
                if state.missed_frame_count > self.config.max_missed_frames:
                    state.retired = True
                    state.nominal_role = VisualTrackRole.RETIRED
                    retired_ids.append(state.track_id)
                else:
                    missed_ids.append(state.track_id)
                continue

            detection = eligible[detection_index]
            used_detection_indexes.add(detection_index)
            pair = pair_scores[(track_index, detection_index)]
            ambiguous = (
                track_index in ambiguous_track_indexes
                or detection_index in ambiguous_detection_indexes
            )
            evidence = self._associate(
                state,
                detection,
                frame,
                pair,
                ambiguous=ambiguous,
            )
            associated_ids.append(state.track_id)
            association_evidence.append(evidence)

        for detection_index, detection in enumerate(eligible):
            if detection_index in used_detection_indexes:
                continue
            state = self._new_track(
                detection,
                frame,
                ambiguous=detection_index in ambiguous_detection_indexes,
            )
            self._states[state.track_id] = state
            created_ids.append(state.track_id)

        self._generation = frame.token.generation
        self._time_basis_id = frame.time_basis_id
        self._provenance_basis = frame.provenance_basis
        self._stream_id = frame.token.stream_id
        self._image_size_px = frame.image_size_px
        self._last_token = frame.token
        self._last_observation_ns = frame.observation_monotonic_ns
        self._last_publish_ns = frame.publish_monotonic_ns
        self._last_camera_source_time_ns = frame.camera_source_time_ns
        self._processed_frame_times[frame.token] = frame.observation_monotonic_ns
        self._processed_publish_times[frame.token] = frame.publish_monotonic_ns
        self._processed_provenance[frame.token] = frame.provenance_basis
        self._processed_frame_keys.add(frame.token.exact_tuple)
        update = self._make_update(
            frame,
            created_ids=created_ids,
            associated_ids=associated_ids,
            missed_ids=missed_ids,
            retired_ids=retired_ids,
            associations=association_evidence,
        )
        self._latest_update = update
        return update

    def assign_role(self, track_id: str, role: VisualTrackRole) -> None:
        """Assign a camera lifecycle role without inventing a gate index."""

        state = self._state(track_id)
        if type(role) is not VisualTrackRole:
            raise TypeError("role must be an exact VisualTrackRole")
        if role not in {
            VisualTrackRole.CURRENT,
            VisualTrackRole.NEXT,
            VisualTrackRole.UNKNOWN,
        }:
            raise ValueError("only current, next, or unknown may be assigned")
        if state.retired:
            raise ValueError("a retired track cannot receive a live role")
        state.nominal_role = role
        self._refresh_latest_update()

    def confirm_authoritative_gate(
        self,
        track_id: str,
        *,
        gate_index: int,
        race_status_sequence: Optional[int],
        race_status_boot_ms: int,
    ) -> None:
        """Attach a semantic index after a caller-proved race association."""

        state = self._state(track_id)
        _nonnegative_int(gate_index, "gate_index")
        if race_status_sequence is not None:
            _nonnegative_int(race_status_sequence, "race_status_sequence")
        _nonnegative_int(race_status_boot_ms, "race_status_boot_ms")
        if state.retired or state.nominal_role is not VisualTrackRole.CURRENT:
            raise ValueError("only the live current track may receive gate authority")
        if state.ambiguous:
            raise ValueError("an ambiguous track cannot receive gate authority")
        if state.authoritative_gate_index is not None:
            if state.authoritative_gate_index != gate_index:
                raise ValueError("one track cannot carry two authoritative gate indexes")
            if (
                state.authority_race_status_sequence is not None
                and race_status_sequence is not None
                and race_status_sequence < state.authority_race_status_sequence
            ):
                raise ValueError("race status sequence cannot regress")
            if (
                state.authority_race_status_boot_ms is not None
                and race_status_boot_ms < state.authority_race_status_boot_ms
            ):
                raise ValueError("race status boot time cannot regress")
        state.authoritative_gate_index = gate_index
        state.authority_race_status_sequence = race_status_sequence
        state.authority_race_status_boot_ms = race_status_boot_ms
        self._refresh_latest_update()

    def retire_track(self, track_id: str) -> None:
        """Retire a crossed current gate while retaining its bounded history."""

        state = self._state(track_id)
        state.retired = True
        state.nominal_role = VisualTrackRole.RETIRED
        state.missed_frame_count = max(state.missed_frame_count, 1)
        self._refresh_latest_update()

    def track(self, track_id: str) -> VisualTrack:
        return self._snapshot(self._state(track_id))

    def tracks(self) -> tuple[VisualTrack, ...]:
        return tuple(
            self._snapshot(state)
            for state in sorted(self._states.values(), key=lambda item: item.track_id)
        )

    def _plan_assignments(
        self,
        frame: VisualDetectionFrame,
    ) -> _AssignmentPlan:
        self._validate_frame_advance(frame)
        eligible = tuple(
            item
            for item in frame.detections
            if item.confidence >= self.config.min_detection_confidence
        )
        active_states = tuple(
            state
            for state in self._states.values()
            if not state.retired
        )
        pair_scores: dict[tuple[int, int], _PairScore] = {}
        for track_index, state in enumerate(active_states):
            for detection_index, detection in enumerate(eligible):
                score = self._pair_score(
                    state,
                    detection,
                    frame.observation_monotonic_ns,
                )
                if score is not None:
                    pair_scores[(track_index, detection_index)] = score

        assignments = _global_assignments(
            track_count=len(active_states),
            detection_count=len(eligible),
            pair_costs={
                key: score.cost
                for key, score in pair_scores.items()
                if score.cost <= self.config.max_assignment_cost
            },
            unmatched_cost=self.config.unmatched_cost,
        )
        selected_pairs = {
            track_index: detection_index
            for track_index, detection_index in assignments.items()
            if detection_index is not None
            and (track_index, detection_index) in pair_scores
            and pair_scores[(track_index, detection_index)].cost
            <= self.config.max_assignment_cost
        }
        (
            ambiguous_track_indexes,
            ambiguous_detection_indexes,
        ) = self._find_assignment_ambiguity(
            pair_scores,
            selected_pairs,
        )
        return _AssignmentPlan(
            active_states=active_states,
            eligible_detections=eligible,
            pair_scores=pair_scores,
            selected_pairs=selected_pairs,
            ambiguous_track_indexes=frozenset(
                ambiguous_track_indexes
            ),
            ambiguous_detection_indexes=frozenset(
                ambiguous_detection_indexes
            ),
        )

    def _validate_frame_advance(self, frame: VisualDetectionFrame) -> None:
        if (
            self._generation is not None
            and frame.token.generation != self._generation
        ):
            if frame.token.generation < self._generation:
                raise StaleVisualFrameError("camera generation regressed")
            raise StaleVisualFrameError(
                "camera generation changed; call reset_generation before updating"
            )
        if self._time_basis_id is not None and frame.time_basis_id != self._time_basis_id:
            raise ValueError("time_basis_id cannot change inside one camera generation")
        if (
            self._provenance_basis is not None
            and frame.provenance_basis is not self._provenance_basis
        ):
            raise ValueError("frame provenance basis cannot change inside a generation")
        if self._stream_id is not None and frame.token.stream_id != self._stream_id:
            raise ValueError("camera stream_id cannot change inside one generation")
        if self._image_size_px is not None and frame.image_size_px != self._image_size_px:
            raise ValueError("image size cannot change inside one camera generation")
        if frame.token.exact_tuple in self._processed_frame_keys:
            raise StaleVisualFrameError("camera frame token was already consumed")
        if self._last_token is not None:
            previous_publication = self._last_token.publication_sequence
            current_publication = frame.token.publication_sequence
            if (
                previous_publication is not None
                and current_publication is not None
                and current_publication <= previous_publication
            ):
                raise StaleVisualFrameError(
                    "camera publication sequence must strictly advance"
                )
        if (
            self._last_observation_ns is not None
            and frame.observation_monotonic_ns <= self._last_observation_ns
        ):
            raise StaleVisualFrameError(
                "observation monotonic time must strictly advance"
            )
        if (
            self._last_publish_ns is not None
            and frame.publish_monotonic_ns is not None
            and frame.publish_monotonic_ns <= self._last_publish_ns
        ):
            raise StaleVisualFrameError("camera publish time must strictly advance")
        if (
            self._last_camera_source_time_ns is not None
            and frame.camera_source_time_ns is not None
            and frame.camera_source_time_ns <= self._last_camera_source_time_ns
        ):
            raise StaleVisualFrameError(
                "opaque camera source ordering token must strictly advance"
            )
        if (
            self._last_token is not None
            and self._provenance_basis is FrameProvenanceBasis.LEGACY_CAPTURE
        ):
            previous_state = self._processed_frame_times.get(self._last_token)
            if (
                previous_state is not None
                and frame.observation_monotonic_ns <= previous_state
            ):
                raise StaleVisualFrameError(
                    "legacy capture receipt time must strictly advance"
                )

    def _pair_score(
        self,
        state: _TrackState,
        detection: VisualDetection,
        observation_ns: int,
    ) -> Optional[_PairScore]:
        latest = state.latest
        dt_ns = observation_ns - latest.observation_monotonic_ns
        if dt_ns <= 0 or dt_ns > self.config.max_association_gap_ns:
            return None
        dt_s = dt_ns / _NS_PER_S
        predicted_center = (
            latest.center_norm[0] + state.center_velocity_norm_s[0] * dt_s,
            latest.center_norm[1] + state.center_velocity_norm_s[1] * dt_s,
        )
        center_residual = math.hypot(
            detection.center_norm[0] - predicted_center[0],
            detection.center_norm[1] - predicted_center[1],
        )
        clipping_continuity = _clipping_continuity(
            latest.clipping,
            detection.clipping,
        )
        shared_clipped_fragment = bool(latest.clipping & detection.clipping)
        relaxation = (
            self.config.clipped_fragment_relaxation
            if shared_clipped_fragment
            else 1.0
        )
        center_bound = self.config.max_center_residual_norm * relaxation
        if center_residual > center_bound:
            return None

        old_width, old_height = _bbox_size(latest.bbox_norm)
        new_width, new_height = _bbox_size(detection.bbox_norm)
        log_width_change = math.log(new_width / old_width)
        log_height_change = math.log(new_height / old_height)
        predicted_log_scale = (
            math.log(latest.apparent_scale) + state.log_scale_rate_s * dt_s
        )
        log_area_residual = 2.0 * (
            detection.log_scale - predicted_log_scale
        )
        if (
            abs(log_width_change)
            > self.config.max_log_width_change * relaxation
            or abs(log_height_change)
            > self.config.max_log_height_change * relaxation
            or abs(log_area_residual)
            > self.config.max_log_area_residual * relaxation
        ):
            return None

        predicted_bbox = _shift_bbox(
            latest.bbox_norm,
            state.center_velocity_norm_s[0] * dt_s * 0.5,
            state.center_velocity_norm_s[1] * dt_s * 0.5,
        )
        bbox_iou = _bbox_iou(predicted_bbox, detection.bbox_norm)
        fallback_bound = (
            self.config.fallback_center_residual_norm * relaxation
        )
        if bbox_iou < self.config.min_bbox_iou and center_residual > fallback_bound:
            return None

        temporal_consistency = 1.0 / (1.0 + state.missed_frame_count)
        appearance_distance = _appearance_distance(
            state.appearance,
            detection.appearance,
        )
        appearance_cost = (
            0.5 if appearance_distance is None else min(1.0, appearance_distance)
        )
        size_cost = (
            abs(log_width_change)
            / (self.config.max_log_width_change * relaxation)
            + abs(log_height_change)
            / (self.config.max_log_height_change * relaxation)
            + abs(log_area_residual)
            / (self.config.max_log_area_residual * relaxation)
        ) / 3.0
        cost = (
            self.config.center_cost_weight * min(1.0, center_residual / center_bound)
            + self.config.overlap_cost_weight * (1.0 - bbox_iou)
            + self.config.size_cost_weight * min(1.0, size_cost)
            + self.config.confidence_cost_weight
            * abs(detection.confidence - state.confidence)
            + self.config.clipping_cost_weight * (1.0 - clipping_continuity)
            + self.config.temporal_cost_weight * (1.0 - temporal_consistency)
            + self.config.appearance_cost_weight * appearance_cost
        )
        return _PairScore(
            cost=cost,
            center_residual=center_residual,
            bbox_iou=bbox_iou,
            log_width_change=log_width_change,
            log_height_change=log_height_change,
            log_area_residual=log_area_residual,
            clipping_continuity=clipping_continuity,
            temporal_consistency=temporal_consistency,
            appearance_distance=appearance_distance,
        )

    def _find_assignment_ambiguity(
        self,
        pair_scores: dict[tuple[int, int], _PairScore],
        selected_pairs: dict[int, int],
    ) -> tuple[set[int], set[int]]:
        ambiguous_tracks: set[int] = set()
        ambiguous_detections: set[int] = set()
        for track_index, detection_index in selected_pairs.items():
            selected_cost = pair_scores[(track_index, detection_index)].cost
            row_alternatives = [
                (other_detection, score.cost)
                for (candidate_track, other_detection), score in pair_scores.items()
                if candidate_track == track_index
                and other_detection != detection_index
                and score.cost <= self.config.max_assignment_cost
            ]
            column_alternatives = [
                (other_track, score.cost)
                for (other_track, candidate_detection), score in pair_scores.items()
                if candidate_detection == detection_index
                and other_track != track_index
                and score.cost <= self.config.max_assignment_cost
            ]
            for other_detection, cost in row_alternatives:
                if cost - selected_cost < self.config.ambiguity_margin:
                    ambiguous_tracks.add(track_index)
                    ambiguous_detections.update((detection_index, other_detection))
            for other_track, cost in column_alternatives:
                if cost - selected_cost < self.config.ambiguity_margin:
                    ambiguous_tracks.update((track_index, other_track))
                    ambiguous_detections.add(detection_index)
        return ambiguous_tracks, ambiguous_detections

    def _associate(
        self,
        state: _TrackState,
        detection: VisualDetection,
        frame: VisualDetectionFrame,
        pair: _PairScore,
        *,
        ambiguous: bool,
    ) -> AssociationEvidence:
        latest = state.latest
        observation_gap_ns = (
            frame.observation_monotonic_ns - latest.observation_monotonic_ns
        )
        dt_s = observation_gap_ns / _NS_PER_S
        publication_gap_ns = (
            frame.publish_monotonic_ns - latest.publication_monotonic_ns
            if frame.publish_monotonic_ns is not None
            and latest.publication_monotonic_ns is not None
            else None
        )
        missed_frame_count_before_association = state.missed_frame_count
        track_ambiguous_before_association = state.ambiguous
        measured_velocity = (
            (detection.center_norm[0] - latest.center_norm[0]) / dt_s,
            (detection.center_norm[1] - latest.center_norm[1]) / dt_s,
        )
        measured_log_scale_rate = (
            detection.log_scale - math.log(latest.apparent_scale)
        ) / dt_s
        alpha = self.config.velocity_smoothing
        state.center_velocity_norm_s = (
            alpha * measured_velocity[0]
            + (1.0 - alpha) * state.center_velocity_norm_s[0],
            alpha * measured_velocity[1]
            + (1.0 - alpha) * state.center_velocity_norm_s[1],
        )
        state.log_scale_rate_s = (
            alpha * measured_log_scale_rate
            + (1.0 - alpha) * state.log_scale_rate_s
        )
        confidence_alpha = self.config.confidence_smoothing
        state.confidence = (
            confidence_alpha * detection.confidence
            + (1.0 - confidence_alpha) * state.confidence
        )
        association_confidence = max(
            0.0,
            min(1.0, 1.0 - pair.cost / self.config.max_assignment_cost),
        )
        if ambiguous:
            state.ambiguous = True
            state.unambiguous_streak = 0
            association_confidence *= 0.25
        else:
            state.unambiguous_streak += 1
            if state.unambiguous_streak >= self.config.ambiguity_clear_frames:
                state.ambiguous = False
        state.association_confidence = association_confidence
        state.consecutive_frame_count = (
            state.consecutive_frame_count + 1
            if state.missed_frame_count == 0
            else 1
        )
        state.total_observation_count += 1
        state.missed_frame_count = 0
        state.appearance = _blend_appearance(
            state.appearance,
            detection.appearance,
            alpha=confidence_alpha,
        )
        evidence = AssociationEvidence(
            track_id=state.track_id,
            previous_token=latest.token,
            current_token=frame.token,
            detection_source_index=detection.source_index,
            cost=pair.cost,
            confidence=association_confidence,
            predicted_center_residual_norm=pair.center_residual,
            bbox_iou=pair.bbox_iou,
            log_width_change=pair.log_width_change,
            log_height_change=pair.log_height_change,
            log_area_residual=pair.log_area_residual,
            clipping_continuity=pair.clipping_continuity,
            temporal_consistency=pair.temporal_consistency,
            appearance_distance=pair.appearance_distance,
            ambiguous=ambiguous,
            missed_frame_count_before_association=(
                missed_frame_count_before_association
            ),
            observation_gap_ns=observation_gap_ns,
            publication_gap_ns=publication_gap_ns,
            track_ambiguous_before_association=(
                track_ambiguous_before_association
            ),
        )
        sample = _sample(
            self._frame_sequence,
            frame,
            detection,
            association_confidence,
            accepted_association=evidence,
        )
        state.history.append(sample)
        if len(state.history) > self.config.history_limit:
            del state.history[: len(state.history) - self.config.history_limit]
        return evidence

    def _new_track(
        self,
        detection: VisualDetection,
        frame: VisualDetectionFrame,
        *,
        ambiguous: bool,
    ) -> _TrackState:
        track_id = f"vq2-track-{self._next_track_number:06d}"
        self._next_track_number += 1
        initial_association_confidence = 0.0 if ambiguous else detection.confidence
        return _TrackState(
            track_id=track_id,
            history=[
                _sample(
                    self._frame_sequence,
                    frame,
                    detection,
                    initial_association_confidence,
                )
            ],
            center_velocity_norm_s=(0.0, 0.0),
            log_scale_rate_s=0.0,
            confidence=detection.confidence,
            association_confidence=initial_association_confidence,
            consecutive_frame_count=1,
            total_observation_count=1,
            missed_frame_count=0,
            nominal_role=VisualTrackRole.UNKNOWN,
            authoritative_gate_index=None,
            authority_race_status_sequence=None,
            authority_race_status_boot_ms=None,
            ambiguous=ambiguous,
            unambiguous_streak=0 if ambiguous else 1,
            retired=False,
            appearance=detection.appearance,
        )

    def _make_update(
        self,
        frame: VisualDetectionFrame,
        *,
        created_ids: list[str],
        associated_ids: list[str],
        missed_ids: list[str],
        retired_ids: list[str],
        associations: list[AssociationEvidence],
    ) -> VisualTrackerUpdate:
        tracks = self.tracks()
        visible = tuple(
            track.track_id
            for track in tracks
            if track.visible and track.role is not VisualTrackRole.RETIRED
        )
        ambiguous = tuple(track.track_id for track in tracks if track.ambiguous)
        return VisualTrackerUpdate(
            tracker_frame_sequence=self._frame_sequence,
            token=frame.token,
            observation_monotonic_ns=frame.observation_monotonic_ns,
            publish_monotonic_ns=frame.publish_monotonic_ns,
            provenance_basis=frame.provenance_basis,
            tracks=tracks,
            visible_track_ids=visible,
            created_track_ids=tuple(created_ids),
            associated_track_ids=tuple(associated_ids),
            missed_track_ids=tuple(missed_ids),
            retired_track_ids=tuple(retired_ids),
            ambiguous_track_ids=ambiguous,
            associations=tuple(associations),
        )

    def _refresh_latest_update(self) -> None:
        update = self._latest_update
        if update is None:
            return
        tracks = self.tracks()
        self._latest_update = VisualTrackerUpdate(
            tracker_frame_sequence=update.tracker_frame_sequence,
            token=update.token,
            observation_monotonic_ns=update.observation_monotonic_ns,
            publish_monotonic_ns=update.publish_monotonic_ns,
            provenance_basis=update.provenance_basis,
            tracks=tracks,
            visible_track_ids=tuple(
                track.track_id
                for track in tracks
                if track.visible and track.role is not VisualTrackRole.RETIRED
            ),
            created_track_ids=update.created_track_ids,
            associated_track_ids=update.associated_track_ids,
            missed_track_ids=update.missed_track_ids,
            retired_track_ids=tuple(
                track.track_id
                for track in tracks
                if track.role is VisualTrackRole.RETIRED
            ),
            ambiguous_track_ids=tuple(
                track.track_id for track in tracks if track.ambiguous
            ),
            associations=update.associations,
        )

    def _snapshot(self, state: _TrackState) -> VisualTrack:
        latest = state.latest
        if state.retired:
            role = VisualTrackRole.RETIRED
        elif state.ambiguous:
            role = VisualTrackRole.AMBIGUOUS
        else:
            role = state.nominal_role
        return VisualTrack(
            track_id=state.track_id,
            first_token=state.history[0].token,
            latest_token=latest.token,
            center_norm=latest.center_norm,
            bbox_norm=latest.bbox_norm,
            apparent_scale=latest.apparent_scale,
            center_velocity_norm_s=state.center_velocity_norm_s,
            log_scale_rate_s=state.log_scale_rate_s,
            confidence=state.confidence,
            association_confidence=state.association_confidence,
            consecutive_frame_count=state.consecutive_frame_count,
            total_observation_count=state.total_observation_count,
            missed_frame_count=state.missed_frame_count,
            clipping=latest.clipping,
            center_censored=latest.center_censored,
            role=role,
            authoritative_gate_index=state.authoritative_gate_index,
            authority_race_status_sequence=state.authority_race_status_sequence,
            authority_race_status_boot_ms=state.authority_race_status_boot_ms,
            ambiguous=state.ambiguous,
            visible=state.missed_frame_count == 0 and not state.retired,
            history=tuple(state.history),
        )

    def _state(self, track_id: str) -> _TrackState:
        if type(track_id) is not str or not track_id:
            raise TypeError("track_id must be a non-empty exact string")
        try:
            return self._states[track_id]
        except KeyError as exc:
            raise KeyError(f"unknown visual track {track_id}") from exc


def _sample(
    frame_sequence: int,
    frame: VisualDetectionFrame,
    detection: VisualDetection,
    association_confidence: float,
    *,
    accepted_association: Optional[AssociationEvidence] = None,
) -> VisualTrackSample:
    return VisualTrackSample(
        tracker_frame_sequence=frame_sequence,
        token=frame.token,
        observation_monotonic_ns=frame.observation_monotonic_ns,
        publication_monotonic_ns=frame.publish_monotonic_ns,
        provenance_basis=frame.provenance_basis,
        camera_source_time_ns=frame.camera_source_time_ns,
        source_index=detection.source_index,
        center_norm=detection.center_norm,
        bbox_norm=detection.bbox_norm,
        apparent_scale=detection.apparent_scale,
        confidence=detection.confidence,
        clipping=detection.clipping,
        center_censored=detection.center_censored,
        association_confidence=association_confidence,
        accepted_association=accepted_association,
        inner_aperture=detection.inner_aperture,
    )


def _global_assignments(
    *,
    track_count: int,
    detection_count: int,
    pair_costs: dict[tuple[int, int], float],
    unmatched_cost: float,
) -> dict[int, Optional[int]]:
    """Minimum-cost rectangular assignment with one dummy per track."""

    if track_count == 0:
        return {}
    column_count = detection_count + track_count
    invalid_cost = unmatched_cost + 1_000_000.0
    matrix: list[list[float]] = []
    for track_index in range(track_count):
        row = [
            pair_costs.get((track_index, detection_index), invalid_cost)
            for detection_index in range(detection_count)
        ]
        row.extend(unmatched_cost for _ in range(track_count))
        matrix.append(row)

    # Hungarian algorithm for rows <= columns, using one-indexed potentials.
    u = [0.0] * (track_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)
    for row_index in range(1, track_count + 1):
        p[0] = row_index
        column0 = 0
        min_value = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column0] = True
            current_row = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current = (
                    matrix[current_row - 1][column - 1]
                    - u[current_row]
                    - v[column]
                )
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    result: dict[int, Optional[int]] = {
        index: None for index in range(track_count)
    }
    for column in range(1, column_count + 1):
        row = p[column]
        if row == 0:
            continue
        detection_index = column - 1
        if detection_index < detection_count:
            result[row - 1] = detection_index
    return result


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _shift_bbox(
    bbox: tuple[float, float, float, float],
    delta_unit_x: float,
    delta_unit_y: float,
) -> tuple[float, float, float, float]:
    return (
        bbox[0] + delta_unit_x,
        bbox[1] + delta_unit_y,
        bbox[2] + delta_unit_x,
        bbox[3] + delta_unit_y,
    )


def _clipping_continuity(previous: FrameEdge, current: FrameEdge) -> float:
    if previous == current:
        return 1.0
    if previous == FrameEdge.NONE or current == FrameEdge.NONE:
        return 0.55
    shared = int(previous & current).bit_count()
    union = int(previous | current).bit_count()
    if shared:
        return 0.65 + 0.35 * shared / union
    opposite = (
        (previous & FrameEdge.LEFT and current & FrameEdge.RIGHT)
        or (previous & FrameEdge.RIGHT and current & FrameEdge.LEFT)
        or (previous & FrameEdge.TOP and current & FrameEdge.BOTTOM)
        or (previous & FrameEdge.BOTTOM and current & FrameEdge.TOP)
    )
    return 0.0 if opposite else 0.25


def _appearance_distance(
    previous: Optional[tuple[float, ...]],
    current: Optional[tuple[float, ...]],
) -> Optional[float]:
    if previous is None or current is None or len(previous) != len(current):
        return None
    denominator = math.sqrt(sum(value * value for value in previous)) * math.sqrt(
        sum(value * value for value in current)
    )
    if denominator <= 1e-12:
        return math.sqrt(
            sum((left - right) ** 2 for left, right in zip(previous, current))
        )
    cosine = sum(left * right for left, right in zip(previous, current)) / denominator
    return max(0.0, min(2.0, 1.0 - cosine))


def _blend_appearance(
    previous: Optional[tuple[float, ...]],
    current: Optional[tuple[float, ...]],
    *,
    alpha: float,
) -> Optional[tuple[float, ...]]:
    if current is None:
        return previous
    if previous is None or len(previous) != len(current):
        return current
    return tuple(
        alpha * right + (1.0 - alpha) * left
        for left, right in zip(previous, current)
    )


def _bbox_size(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _bbox(
    value: object,
    label: str,
) -> tuple[float, float, float, float]:
    if type(value) is not tuple or len(value) != 4:
        raise TypeError(f"{label} must be an exact four-tuple")
    left, top, right, bottom = (
        _finite(item, f"{label}[{index}]", minimum=0.0, maximum=1.0)
        for index, item in enumerate(value)
    )
    if right <= left or bottom <= top:
        raise ValueError(f"{label} must have positive width and height")
    return left, top, right, bottom


def _finite_pair(
    value: object,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{label} must be an exact pair")
    return (
        _finite(value[0], f"{label}[0]", minimum=minimum, maximum=maximum),
        _finite(value[1], f"{label}[1]", minimum=minimum, maximum=maximum),
    )


def _finite(
    value: object,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise TypeError(f"{label} must be finite numeric data")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _exact_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def _nonnegative_int(
    value: object,
    label: str,
    *,
    maximum: Optional[int] = None,
) -> int:
    result = _exact_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be a non-empty exact string")
    return value


def _image_size(value: object) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError("image_size_px must be an exact pair")
    width = _positive_int(value[0], "image_size_px[0]")
    height = _positive_int(value[1], "image_size_px[1]")
    return width, height


DEFAULT_MULTI_TARGET_TRACKER_CONFIG = MultiTargetTrackerConfig()
