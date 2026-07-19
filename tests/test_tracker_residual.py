"""
Tests for the learned tracker residual (iter-001 A14, iter-031 v2).

These pin the safety contract:
  - The MLP forward pass produces the right shape for any 12-dim input.
  - Even a model whose raw output is huge (10, 10, 10) gets clamped at the
    consumer to ±residual_clamp_rad / ±residual_thrust_clamp.
  - With `use_residual=False`, the tracker output is byte-identical to the
    pre-A15 baseline.
  - With `use_residual=True` and zero-init weights, the residual is zero
    so behaviour matches baseline within float-epsilon (an extra branch
    runs but it adds zero).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolated_benchmark_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGP_CACHE_ROOT", str(tmp_path / "artifacts"))

from control.learned_residual import (
    DEFAULT_N_HIDDEN,
    DEFAULT_N_INPUTS,
    DEFAULT_N_OUTPUTS,
    TrackerResidualMLP,
    build_input_features,
)
from control.mpc_tracker import GeometricTracker, TrackerConfig
from planning.trajectory_optimizer import TrajectoryPoint


def _hover_ref() -> TrajectoryPoint:
    return TrajectoryPoint(
        time=0.0,
        position=(0.0, 0.0, -2.0),
        velocity=(0.0, 0.0, 0.0),
        acceleration=(0.0, 0.0, 0.0),
        jerk=(0.0, 0.0, 0.0),
        yaw=0.0,
        yaw_rate=0.0,
    )


# ---------------------------------------------------------------------------
# Iter-014 — feature-trace hook (ML training infrastructure)
# ---------------------------------------------------------------------------

def test_feature_trace_empty_by_default():
    """TrackerConfig.trace_features=False (the default) must produce
    an empty trace, even after many track() calls. No-overhead contract
    for production callers."""
    tracker = GeometricTracker(TrackerConfig())
    ref = _hover_ref()
    for _ in range(50):
        tracker.track(
            current_position=(0.1, 0.0, -1.9),
            current_velocity=(0.0, 0.0, 0.0),
            current_yaw=0.0,
            reference=ref,
        )
    assert tracker.feature_trace == []


def test_feature_trace_captures_when_enabled():
    """Iter-031 v2: with trace_features=True, every track() call appends
    a 13-tuple (features_12d, roll_nom, pitch_nom, thrust_nom, pos_err,
    vel_err, pos, vel, yaw_des, ref_pos, ref_vel, ref_accel,
    accel_des_baseline)."""
    tracker = GeometricTracker(TrackerConfig(trace_features=True))
    ref = _hover_ref()
    for _ in range(7):
        tracker.track(
            current_position=(0.2, 0.1, -1.8),
            current_velocity=(0.0, 0.0, 0.0),
            current_yaw=0.0,
            reference=ref,
        )
    assert len(tracker.feature_trace) == 7
    for entry in tracker.feature_trace:
        assert len(entry) == 13
        (features, roll, pitch, thrust, pos_err, vel_err,
         pos, vel, yaw_des, ref_pos, ref_vel, ref_accel,
         accel_des_baseline) = entry
        assert features.shape == (DEFAULT_N_INPUTS,)
        assert np.all(np.isfinite(features))
        assert -math.pi <= roll <= math.pi
        assert -math.pi <= pitch <= math.pi
        assert 0.0 <= thrust <= 1.5  # normalized; near ~1.0 at hover
        assert len(pos_err) == 3
        assert len(vel_err) == 3
        assert len(pos) == 3
        assert len(vel) == 3
        assert -math.pi <= yaw_des <= math.pi
        assert len(ref_pos) == 3
        assert len(ref_vel) == 3
        assert len(ref_accel) == 3
        assert len(accel_des_baseline) == 3


def test_feature_trace_save_load_roundtrip(tmp_path):
    """Iter-015: save_feature_trace + load_feature_trace must round-trip
    the trace exactly. Crucial for future ML training pipelines: the
    collected dataset must be persistable across processes."""
    from control.learned_residual import (
        save_feature_trace,
        load_feature_trace,
    )
    tracker = GeometricTracker(TrackerConfig(trace_features=True))
    ref = _hover_ref()
    for i in range(5):
        tracker.track(
            current_position=(0.1 * i, 0.0, -1.8),
            current_velocity=(0.0, 0.0, 0.0),
            current_yaw=0.0,
            reference=ref,
        )
    path = tmp_path / "trace.npz"
    save_feature_trace(tracker.feature_trace, path)
    loaded = load_feature_trace(path)
    assert loaded["features"].shape == (5, DEFAULT_N_INPUTS)
    assert loaded["roll_nom"].shape == (5,)
    assert loaded["pitch_nom"].shape == (5,)
    assert loaded["thrust_nom"].shape == (5,)
    assert loaded["pos_err"].shape == (5, 3)
    assert loaded["vel_err"].shape == (5, 3)
    assert loaded["pos"].shape == (5, 3)
    assert loaded["vel"].shape == (5, 3)
    assert loaded["yaw_des"].shape == (5,)
    assert loaded["ref_pos"].shape == (5, 3)
    assert loaded["ref_vel"].shape == (5, 3)
    assert loaded["ref_accel"].shape == (5, 3)
    assert loaded["accel_des_baseline"].shape == (5, 3)
    # Round-trip integrity: same values
    for i, entry in enumerate(tracker.feature_trace):
        (feats, r, p, t, pe, ve, po, va, yd, rp, rv, ra, adb) = entry
        np.testing.assert_array_equal(loaded["features"][i], feats)
        assert loaded["roll_nom"][i] == r
        assert loaded["pitch_nom"][i] == p
        assert loaded["thrust_nom"][i] == t
        np.testing.assert_array_equal(loaded["pos_err"][i], pe)
        np.testing.assert_array_equal(loaded["vel_err"][i], ve)
        np.testing.assert_array_equal(loaded["pos"][i], po)
        np.testing.assert_array_equal(loaded["vel"][i], va)
        assert loaded["yaw_des"][i] == yd
        np.testing.assert_array_equal(loaded["ref_pos"][i], rp)
        np.testing.assert_array_equal(loaded["ref_vel"][i], rv)
        np.testing.assert_array_equal(loaded["ref_accel"][i], ra)
        np.testing.assert_array_equal(loaded["accel_des_baseline"][i], adb)


def test_save_feature_trace_rejects_empty(tmp_path):
    """Empty trace should raise — saving zero rows wastes disk and
    breaks the load contract."""
    from control.learned_residual import save_feature_trace
    import pytest as _pytest
    with _pytest.raises(ValueError, match="empty"):
        save_feature_trace([], tmp_path / "x.npz")


@pytest.mark.benchmark
def test_bench_exposes_tracker_feature_trace_when_enabled():
    """Iter-024: the synthetic matrix bench exposes the GeometricTracker's
    feature_trace in the result dict so external scripts can collect
    real ML training data. With trace_features=True, the trace has
    samples in the canonical 12-dim v2 format; with the flag off the list
    is empty (no overhead for production callers)."""
    import json
    from pathlib import Path
    from scripts.benchmark import run_synthetic_benchmark

    cfg_path = Path(__file__).resolve().parent.parent / "sim_pybullet" / "configs" / "race_01.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Default path: empty trace.
    r_off = run_synthetic_benchmark(duration=1.0, config=cfg)
    assert r_off.get("tracker_feature_trace") == []

    # Enabled path: nonempty v2 trace.
    r_on = run_synthetic_benchmark(
        duration=1.0, config=cfg,
        tracker_config_overrides={"trace_features": True},
    )
    trace = r_on.get("tracker_feature_trace")
    assert isinstance(trace, list)
    assert len(trace) > 50  # 1s at 100Hz minus startup; should be ~100
    entry = trace[0]
    assert len(entry) == 13  # v2 schema
    features, roll, pitch, thrust, pos_err, vel_err = entry[:6]
    assert features.shape == (DEFAULT_N_INPUTS,)
    assert np.all(np.isfinite(features))
    assert -math.pi <= roll <= math.pi
    assert -math.pi <= pitch <= math.pi
    assert 0.0 <= thrust <= 1.5
    assert len(pos_err) == 3
    assert len(vel_err) == 3

    # JSON-backed cache hits must preserve the same in-memory feature-vector
    # contract as the cold evaluator path.
    r_cached = run_synthetic_benchmark(
        duration=1.0,
        config=cfg,
        tracker_config_overrides={"trace_features": True},
    )
    assert r_cached["cache"]["benchmark_result"] == "hit"
    assert r_cached["tracker_feature_trace"][0][0].shape == (
        DEFAULT_N_INPUTS,
    )


@pytest.mark.benchmark
def test_collect_residual_dataset_smoke(tmp_path, monkeypatch):
    """Iter-025: collect_residual_dataset.collect() runs the bench
    across non-skip tracks, concatenates traces, saves an .npz that
    round-trips through `load_feature_trace`. Smoke-tests the
    pipeline end-to-end; we monkeypatch `_SKIP` to only one track
    so the test stays under a few seconds."""
    from control.learned_residual import load_feature_trace
    from scripts import collect_residual_dataset as col

    # Skip all but race_01 to keep wall-time bounded.
    monkeypatch.setattr(
        col, "_SKIP",
        {"figure8", "aigp_default", "slalom", "grand_tour",
         "straight_hairpin", "vertical_cliff"},
    )
    out = tmp_path / "ds.npz"
    summary = col.collect(out, duration=2.0, allow_prefix=True)
    assert summary["total_samples"] > 50
    # Verify the file round-trips.
    loaded = load_feature_trace(out)
    assert loaded["features"].shape[1] == DEFAULT_N_INPUTS
    assert loaded["features"].shape[0] == summary["total_samples"]
    assert loaded["pos_err"].shape == (summary["total_samples"], 3)


def test_init_residual_weights_script_writes_loadable_zero_npz(tmp_path):
    """Iter-022 connect-the-dots: the init_residual_weights.py script
    must produce a .npz file that `TrackerResidualMLP.from_npz` can
    load AND whose forward pass is zero. This is the round-trip
    contract that lets the pipeline test
    `use_residual=True + this_file` ≡ baseline.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    out = tmp_path / "weights.npz"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "init_residual_weights.py"),
            "--out", str(out),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"script failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert out.exists()
    # Load + verify zero behavior.
    mlp = TrackerResidualMLP.from_npz(out)
    for sample in (
        np.zeros(DEFAULT_N_INPUTS),
        np.ones(DEFAULT_N_INPUTS),
        np.linspace(-3.0, 3.0, DEFAULT_N_INPUTS),
    ):
        y = mlp.forward(sample)
        assert y.shape == (DEFAULT_N_OUTPUTS,)
        np.testing.assert_array_equal(y, np.zeros(DEFAULT_N_OUTPUTS))


# ---------------------------------------------------------------------------
# MLP shape + sanity
# ---------------------------------------------------------------------------

def test_zero_init_forward_is_zero():
    mlp = TrackerResidualMLP.zero_init()
    x = np.linspace(-1.0, 1.0, DEFAULT_N_INPUTS)
    y = mlp.forward(x)
    assert y.shape == (DEFAULT_N_OUTPUTS,)
    assert np.allclose(y, 0.0)


def test_random_init_forward_shape():
    mlp = TrackerResidualMLP.random_init(seed=7)
    y = mlp.forward(np.zeros(DEFAULT_N_INPUTS))
    assert y.shape == (DEFAULT_N_OUTPUTS,)


def test_input_dim_mismatch_raises():
    mlp = TrackerResidualMLP.zero_init()
    with pytest.raises(ValueError):
        mlp.forward(np.zeros(DEFAULT_N_INPUTS - 1))


@pytest.mark.parametrize("field", ["W1", "b1", "W2", "b2"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_model_rejects_nonfinite_parameters(field, bad_value):
    values = {
        "W1": np.zeros((DEFAULT_N_INPUTS, DEFAULT_N_HIDDEN)),
        "b1": np.zeros(DEFAULT_N_HIDDEN),
        "W2": np.zeros((DEFAULT_N_HIDDEN, DEFAULT_N_OUTPUTS)),
        "b2": np.zeros(DEFAULT_N_OUTPUTS),
    }
    values[field].flat[0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        TrackerResidualMLP(**values)


@pytest.mark.parametrize("bad_std", [0.0, -1.0, float("nan"), float("inf")])
def test_model_rejects_invalid_feature_standard_deviation(bad_std):
    with pytest.raises(ValueError, match="feat_std"):
        TrackerResidualMLP(
            W1=np.zeros((DEFAULT_N_INPUTS, DEFAULT_N_HIDDEN)),
            b1=np.zeros(DEFAULT_N_HIDDEN),
            W2=np.zeros((DEFAULT_N_HIDDEN, DEFAULT_N_OUTPUTS)),
            b2=np.zeros(DEFAULT_N_OUTPUTS),
            feat_mean=np.zeros(DEFAULT_N_INPUTS),
            feat_std=np.full(DEFAULT_N_INPUTS, bad_std),
        )


@pytest.mark.parametrize("bad_clamp", [0.0, -1.0, float("nan"), float("inf")])
def test_model_rejects_invalid_output_clamp(bad_clamp):
    with pytest.raises(ValueError, match="output_clamp"):
        TrackerResidualMLP(
            W1=np.zeros((DEFAULT_N_INPUTS, DEFAULT_N_HIDDEN)),
            b1=np.zeros(DEFAULT_N_HIDDEN),
            W2=np.zeros((DEFAULT_N_HIDDEN, DEFAULT_N_OUTPUTS)),
            b2=np.zeros(DEFAULT_N_OUTPUTS),
            output_clamp=np.full(DEFAULT_N_OUTPUTS, bad_clamp),
        )


def test_model_rejects_nonfinite_input_before_command_generation():
    mlp = TrackerResidualMLP.zero_init()
    sample = np.zeros(DEFAULT_N_INPUTS)
    sample[0] = np.nan
    with pytest.raises(ValueError, match="input.*finite"):
        mlp.forward(sample)


@pytest.mark.parametrize("n_outputs", [1, 2, 4])
def test_tracker_rejects_selected_model_with_wrong_output_dimension(
    tmp_path: Path, n_outputs: int
):
    malformed = TrackerResidualMLP.zero_init(n_outputs=n_outputs)
    path = tmp_path / f"wrong-output-{n_outputs}.npz"
    malformed.to_npz(path)

    with pytest.raises(RuntimeError, match="selected residual weights are invalid"):
        GeometricTracker(
            TrackerConfig(use_residual=True, residual_weights_path=str(path))
        )


def test_tracker_rejects_explicit_missing_residual_weights(tmp_path: Path):
    with pytest.raises(RuntimeError, match="selected residual weights are invalid"):
        GeometricTracker(
            TrackerConfig(
                use_residual=True,
                residual_weights_path=str(tmp_path / "missing.npz"),
            )
        )


def test_tracker_rejects_explicit_corrupt_residual_weights(tmp_path: Path):
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not a NumPy archive")
    with pytest.raises(RuntimeError, match="selected residual weights are invalid"):
        GeometricTracker(
            TrackerConfig(use_residual=True, residual_weights_path=str(path))
        )


def test_build_input_features_is_12_dim_and_ordered():
    """Iter-031: 12-dim feature vector — adds sin(yaw), cos(yaw)."""
    feats = build_input_features(
        pos_err=np.array([1.0, 2.0, 3.0]),
        vel_err=np.array([4.0, 5.0, 6.0]),
        ref_accel=np.array([7.0, 8.0, 9.0]),
        thrust_normalized=0.5,
        yaw_des=0.0,
    )
    assert feats.shape == (DEFAULT_N_INPUTS,)
    # yaw=0 → sin=0, cos=1
    assert list(feats) == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.5, 0.0, 1.0,
    ]
    # yaw=π/2 → sin=1, cos=0
    feats_90 = build_input_features(
        pos_err=np.zeros(3), vel_err=np.zeros(3),
        ref_accel=np.zeros(3), thrust_normalized=0.0,
        yaw_des=math.pi / 2,
    )
    assert abs(feats_90[10] - 1.0) < 1e-9
    assert abs(feats_90[11] - 0.0) < 1e-9


def test_npz_roundtrip(tmp_path: Path):
    mlp = TrackerResidualMLP.random_init(seed=42)
    path = tmp_path / "rt.npz"
    mlp.to_npz(path)
    reloaded = TrackerResidualMLP.from_npz(path)
    assert np.allclose(mlp.W1, reloaded.W1)
    assert np.allclose(mlp.b1, reloaded.b1)
    assert np.allclose(mlp.W2, reloaded.W2)
    assert np.allclose(mlp.b2, reloaded.b2)


def test_no_torch_at_import_time():
    """The control loop must not pay a torch import tax at runtime."""
    import sys
    import importlib

    # Pretend torch is unavailable; the residual module must still import.
    saved = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        # Reload to force re-evaluation of any conditional imports.
        if "control.learned_residual" in sys.modules:
            importlib.reload(sys.modules["control.learned_residual"])
        else:
            import control.learned_residual  # noqa: F401
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)


# ---------------------------------------------------------------------------
# Hard clamp at the consumer
# ---------------------------------------------------------------------------

def test_residual_output_clamped_at_consumer(tmp_path: Path):
    """Even if the model outputs 10.0 on every channel, the consumer must
    clip the delta to ±residual_clamp_rad / ±residual_thrust_clamp."""
    # Build a model whose forward pass returns a constant huge value.
    # Easiest construction: zero hidden weights but a large b2.
    mlp = TrackerResidualMLP.zero_init()
    mlp.b2 = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    npz_path = tmp_path / "huge.npz"
    mlp.to_npz(npz_path)

    cfg_residual = TrackerConfig(
        use_residual=True,
        residual_weights_path=str(npz_path),
        residual_clamp_rad=0.05,
        residual_thrust_clamp=0.05,
    )
    cfg_baseline = TrackerConfig(use_residual=False)
    tracker_res = GeometricTracker(cfg_residual)
    tracker_base = GeometricTracker(cfg_baseline)

    pos = (0.0, 0.0, -2.0)
    vel = (0.0, 0.0, 0.0)
    yaw = 0.0
    ref = _hover_ref()

    cmd_res = tracker_res.track(pos, vel, yaw, ref)
    cmd_base = tracker_base.track(pos, vel, yaw, ref)

    # Delta must not exceed the clamp magnitude.
    assert abs(cmd_res.roll_rad - cmd_base.roll_rad) <= 0.05 + 1e-9
    assert abs(cmd_res.pitch_rad - cmd_base.pitch_rad) <= 0.05 + 1e-9
    assert abs(cmd_res.thrust - cmd_base.thrust) <= 0.05 + 1e-9
    # And tilt limits remain enforced.
    assert abs(cmd_res.roll_rad) <= cfg_residual.max_tilt_rad
    assert abs(cmd_res.pitch_rad) <= cfg_residual.max_tilt_rad
    assert cfg_residual.min_thrust_normalized <= cmd_res.thrust <= cfg_residual.max_thrust_normalized


# ---------------------------------------------------------------------------
# Off-by-default safety
# ---------------------------------------------------------------------------

def test_residual_off_is_byte_identical_to_baseline():
    """`use_residual=False` (the default) must give the same command as
    a tracker built with no config tweaks at all."""
    cfg_default = TrackerConfig()
    cfg_explicit_off = TrackerConfig(use_residual=False)
    t1 = GeometricTracker(cfg_default)
    t2 = GeometricTracker(cfg_explicit_off)

    pos = (1.0, 0.5, -2.5)
    vel = (0.5, -0.2, 0.0)
    yaw = 0.3
    ref = TrajectoryPoint(
        time=0.0,
        position=(2.0, 0.0, -2.0),
        velocity=(1.0, 0.0, 0.0),
        acceleration=(0.5, 0.0, 0.0),
        jerk=(0.0, 0.0, 0.0),
        yaw=0.0,
        yaw_rate=0.0,
    )

    c1 = t1.track(pos, vel, yaw, ref)
    c2 = t2.track(pos, vel, yaw, ref)
    assert c1.roll_rad == c2.roll_rad
    assert c1.pitch_rad == c2.pitch_rad
    assert c1.yaw_rad == c2.yaw_rad
    assert c1.thrust == c2.thrust


def test_residual_on_with_no_selected_weights_matches_baseline_within_float_epsilon(
    monkeypatch,
):
    """Turn the feature on with NO weights file present -> the safety
    fallback initialises zero-init weights, so the tracker output matches
    baseline to within numerical noise.

    Iter-031 added auto-resolve at `<repo>/control/residual_weights.npz`
    when `residual_weights_path=None`. Force that optional default absent so
    this test remains independent of locally generated ignored weights."""
    import control.mpc_tracker as tracker_module

    default_weights = Path(tracker_module.__file__).resolve().parent / "residual_weights.npz"
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: False if self.resolve() == default_weights else original_exists(self),
    )
    cfg_off = TrackerConfig(use_residual=False)
    cfg_on_zero = TrackerConfig(use_residual=True, residual_weights_path=None)
    t_off = GeometricTracker(cfg_off)
    t_on = GeometricTracker(cfg_on_zero)

    pos = (1.0, 0.5, -2.5)
    vel = (0.5, -0.2, 0.0)
    yaw = 0.3
    ref = TrajectoryPoint(
        time=0.0,
        position=(2.0, 0.0, -2.0),
        velocity=(1.0, 0.0, 0.0),
        acceleration=(0.5, 0.0, 0.0),
        jerk=(0.0, 0.0, 0.0),
        yaw=0.0,
        yaw_rate=0.0,
    )

    c_off = t_off.track(pos, vel, yaw, ref)
    c_on = t_on.track(pos, vel, yaw, ref)
    # Should agree to ~1e-12 — the residual adds exactly 0.0, so any
    # difference comes from numpy float reassociation in the branch.
    assert abs(c_on.roll_rad - c_off.roll_rad) < 1e-9
    assert abs(c_on.pitch_rad - c_off.pitch_rad) < 1e-9
    assert abs(c_on.thrust - c_off.thrust) < 1e-9
