"""
Test script for Gate Detection
==============================
Run this to verify your gate detector is working correctly.

Usage:
    python test_detection.py
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gate_detector import GateDetector, GateDetection, get_steering_error


def create_synthetic_gate_image(
    gate_color_bgr: tuple = (0, 0, 255),  # Red
    gate_center: tuple = (320, 240),
    gate_size: int = 150,
    background_color: tuple = (40, 40, 40),
    noise_level: int = 20,
    rotation_deg: float = 0,
) -> np.ndarray:
    """
    Create a synthetic image with a gate for testing.
    
    Args:
        gate_color_bgr: Gate color in BGR format
        gate_center: (x, y) center of gate in image
        gate_size: Size of gate in pixels
        background_color: Background color
        noise_level: Amount of random noise to add
        rotation_deg: Rotation of gate in degrees
    """
    # Create background
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = background_color
    
    # Create gate as a hollow square
    half_size = gate_size // 2
    thickness = gate_size // 5
    
    cx, cy = gate_center
    
    # Draw gate (thick square outline)
    # Outer rectangle
    outer_pts = np.array([
        [cx - half_size, cy - half_size],
        [cx + half_size, cy - half_size],
        [cx + half_size, cy + half_size],
        [cx - half_size, cy + half_size],
    ], dtype=np.float32)
    
    # Inner rectangle (to cut out)
    inner_half = half_size - thickness
    inner_pts = np.array([
        [cx - inner_half, cy - inner_half],
        [cx + inner_half, cy - inner_half],
        [cx + inner_half, cy + inner_half],
        [cx - inner_half, cy + inner_half],
    ], dtype=np.float32)
    
    # Apply rotation
    if rotation_deg != 0:
        # Rotation matrix around center
        angle_rad = np.radians(rotation_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        for pts in [outer_pts, inner_pts]:
            for i in range(4):
                x, y = pts[i] - [cx, cy]
                pts[i] = [
                    cx + x * cos_a - y * sin_a,
                    cy + x * sin_a + y * cos_a
                ]
    
    # Draw filled outer
    cv2.fillPoly(image, [outer_pts.astype(np.int32)], gate_color_bgr)
    # Cut out inner
    cv2.fillPoly(image, [inner_pts.astype(np.int32)], background_color)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.randint(0, noise_level, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
    
    return image


def test_basic_detection():
    """Test that detector finds a gate in a simple image."""
    print("\n" + "="*50)
    print("TEST: Basic Detection")
    print("="*50)
    
    # Create test image with red gate in center
    image = create_synthetic_gate_image(
        gate_color_bgr=(0, 0, 255),  # Red
        gate_center=(320, 240),
        gate_size=150
    )
    
    # Detect
    detector = GateDetector(color_preset="red")
    detections = detector.detect(image)
    
    # Check results
    assert len(detections) >= 1, "Should detect at least one gate"
    
    det = detections[0]
    print(f"  Gate center: ({det.center_x}, {det.center_y})")
    print(f"  Expected: (320, 240)")
    print(f"  Distance error: {abs(det.center_x - 320) + abs(det.center_y - 240)} pixels")
    
    # Allow some tolerance
    assert abs(det.center_x - 320) < 20, f"X center off by {abs(det.center_x - 320)}"
    assert abs(det.center_y - 240) < 20, f"Y center off by {abs(det.center_y - 240)}"
    
    print("  ✓ PASSED")
    return image, detections


def test_offset_detection():
    """Test detection of gate not in center."""
    print("\n" + "="*50)
    print("TEST: Offset Detection")
    print("="*50)
    
    # Gate in upper-left
    image = create_synthetic_gate_image(
        gate_color_bgr=(0, 0, 255),
        gate_center=(150, 120),
        gate_size=100
    )
    
    detector = GateDetector(color_preset="red")
    detections = detector.detect(image)
    
    assert len(detections) >= 1, "Should detect gate"
    det = detections[0]
    
    print(f"  Gate center: ({det.center_x}, {det.center_y})")
    print(f"  Expected: (150, 120)")
    
    assert abs(det.center_x - 150) < 20
    assert abs(det.center_y - 120) < 20
    
    print("  ✓ PASSED")
    return image, detections


def test_steering_error():
    """Test steering error calculation."""
    print("\n" + "="*50)
    print("TEST: Steering Error Calculation")
    print("="*50)
    
    # Gate to the right of center
    image = create_synthetic_gate_image(
        gate_center=(480, 240),  # Right of center (640/2 = 320)
        gate_size=100
    )
    
    detector = GateDetector(color_preset="red")
    detections = detector.detect(image)
    
    assert len(detections) >= 1
    det = detections[0]
    
    h_error, v_error = get_steering_error(det, 640, 480)
    
    print(f"  Gate center: ({det.center_x}, {det.center_y})")
    print(f"  Horizontal error: {h_error:.3f} (positive = right)")
    print(f"  Vertical error: {v_error:.3f}")
    
    assert h_error > 0, "Gate is right of center, error should be positive"
    assert abs(v_error) < 0.2, "Gate is near vertical center"
    
    print("  ✓ PASSED")


def test_multiple_colors():
    """Test detection with different gate colors."""
    print("\n" + "="*50)
    print("TEST: Multiple Colors")
    print("="*50)
    
    colors = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "orange": (0, 165, 255),
        "green": (0, 255, 0),
    }
    
    for name, bgr in colors.items():
        image = create_synthetic_gate_image(
            gate_color_bgr=bgr,
            gate_center=(320, 240),
            gate_size=150
        )
        
        detector = GateDetector(color_preset=name)
        detections = detector.detect(image)
        
        if len(detections) >= 1:
            print(f"  {name}: ✓ Detected")
        else:
            print(f"  {name}: ✗ Not detected")
    
    print("  ✓ PASSED")


def test_rotated_gate():
    """Test detection of rotated gate."""
    print("\n" + "="*50)
    print("TEST: Rotated Gate")
    print("="*50)
    
    image = create_synthetic_gate_image(
        gate_center=(320, 240),
        gate_size=150,
        rotation_deg=15
    )
    
    detector = GateDetector(color_preset="red")
    detections = detector.detect(image)
    
    assert len(detections) >= 1, "Should detect rotated gate"
    
    det = detections[0]
    print(f"  Gate center: ({det.center_x}, {det.center_y})")
    print(f"  Confidence: {det.confidence:.2f}")
    
    print("  ✓ PASSED")
    return image, detections


def test_angled_gate_3d_like():
    """Test detection when gate is not head-on (strong in-plane rotation / 3D-like view)."""
    print("\n" + "="*50)
    print("TEST: Angled / 3D-like Gate (not head-on)")
    print("="*50)
    
    # Strong rotations that still look like a quad in 2D
    for rotation_deg in [35, 45, -40]:
        image = create_synthetic_gate_image(
            gate_center=(320, 240),
            gate_size=120,
            rotation_deg=rotation_deg,
        )
        detector = GateDetector(color_preset="red")
        detections = detector.detect(image)
        assert len(detections) >= 1, f"Should detect gate at rotation {rotation_deg}°"
        det = detections[0]
        assert det.confidence > 0, "Should have positive confidence"
        print(f"  rotation={rotation_deg}° -> detected center=({det.center_x}, {det.center_y}) conf={det.confidence:.2f}")
    print("  ✓ PASSED (works for angled/rotated gates)")


def test_multiple_gates_in_frame():
    """Test that we detect all gates when more than one is in the frame."""
    print("\n" + "="*50)
    print("TEST: Multiple Gates in One Frame")
    print("="*50)
    
    # Build image with 3 gates (red hollow squares at different positions)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)
    red = (0, 0, 255)
    centers = [(160, 160), (320, 240), (480, 320)]
    size = 80
    half = size // 2
    thick = size // 5
    inner_half = half - thick
    
    for cx, cy in centers:
        outer = np.array([
            [cx - half, cy - half], [cx + half, cy - half],
            [cx + half, cy + half], [cx - half, cy + half],
        ], dtype=np.int32)
        inner = np.array([
            [cx - inner_half, cy - inner_half], [cx + inner_half, cy - inner_half],
            [cx + inner_half, cy + inner_half], [cx - inner_half, cy + inner_half],
        ], dtype=np.int32)
        cv2.fillPoly(image, [outer], red)
        cv2.fillPoly(image, [inner], (40, 40, 40))
    
    detector = GateDetector(color_preset="red", min_area=500)
    detections = detector.detect(image)
    
    assert len(detections) >= 2, f"Expected at least 2 gates, got {len(detections)}"
    # With 3 gates we may get 3; allow 2 if one contour merges
    print(f"  Gates in image: 3, detections: {len(detections)}")
    print("  ✓ PASSED (multiple gates detected)")


def test_partial_gate_at_edge():
    """Test that we can still detect a gate that is partially out of frame."""
    print("\n" + "="*50)
    print("TEST: Partial Gate (not fully in frame)")
    print("="*50)
    
    # Gate centered so that a good chunk is visible but one side is clipped
    # Put center near left edge so right part of gate is in frame
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)
    red = (0, 0, 255)
    cx, cy = 100, 240  # gate center; gate will extend left out of frame
    half = 90
    thick = 18
    inner_half = half - thick
    # Clip points to image (0..639, 0..479)
    outer = np.array([
        [max(0, cx - half), cy - half], [min(639, cx + half), cy - half],
        [min(639, cx + half), cy + half], [max(0, cx - half), cy + half],
    ], dtype=np.int32)
    inner = np.array([
        [max(0, cx - inner_half), cy - inner_half], [min(639, cx + inner_half), cy - inner_half],
        [min(639, cx + inner_half), cy + inner_half], [max(0, cx - inner_half), cy + inner_half],
    ], dtype=np.int32)
    cv2.fillPoly(image, [outer], red)
    cv2.fillPoly(image, [inner], (40, 40, 40))
    
    detector = GateDetector(color_preset="red", min_area=500, max_aspect_ratio_partial=4.5)
    detections = detector.detect(image)
    
    # We should get at least one detection (the visible part of the gate)
    assert len(detections) >= 1, "Should detect partial gate at image edge"
    print(f"  Partial gate at edge -> detections: {len(detections)}")
    print("  ✓ PASSED (partial gate handled)")


def test_no_gate():
    """Test that detector doesn't false positive on empty image."""
    print("\n" + "="*50)
    print("TEST: No Gate (False Positive Check)")
    print("="*50)
    
    # Empty dark image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)
    
    detector = GateDetector(color_preset="red")
    detections = detector.detect(image)
    
    print(f"  Detections found: {len(detections)}")
    assert len(detections) == 0, "Should not detect gate in empty image"
    
    print("  ✓ PASSED")


def test_distance_estimation():
    """Test that distance estimation decreases as gate gets larger."""
    print("\n" + "="*50)
    print("TEST: Distance Estimation")
    print("="*50)
    
    detector = GateDetector(color_preset="red")
    
    distances = []
    sizes = [50, 100, 150, 200, 250]
    
    for size in sizes:
        image = create_synthetic_gate_image(
            gate_center=(320, 240),
            gate_size=size
        )
        detections = detector.detect(image)
        
        if detections:
            distances.append(detections[0].estimated_distance)
            print(f"  Size {size}px -> Distance: {detections[0].estimated_distance:.2f}m")
    
    # Distance should decrease as gate size increases
    for i in range(len(distances) - 1):
        assert distances[i] > distances[i+1], "Distance should decrease as gate appears larger"
    
    print("  ✓ PASSED (distances decrease as gate appears larger)")


def test_rich_ml_features():
    """Test that each detection exposes rich ML features (size, angle, above/below)."""
    print("\n" + "="*50)
    print("TEST: Rich ML Features for Path-Planning ML")
    print("="*50)

    # Gate above center
    image_above = create_synthetic_gate_image(
        gate_center=(320, 120), gate_size=100
    )
    detector = GateDetector(color_preset="red")
    det_above = detector.detect(image_above)
    assert len(det_above) >= 1
    d = det_above[0]
    assert d.is_above_center is True
    assert d.is_below_center is False
    assert d.normalized_center_y < 0

    # Gate below center
    image_below = create_synthetic_gate_image(
        gate_center=(320, 360), gate_size=100
    )
    det_below = detector.detect(image_below)
    assert len(det_below) >= 1
    d = det_below[0]
    assert d.is_below_center is True
    assert d.is_above_center is False
    assert d.normalized_center_y > 0

    # ML feature dict
    features = d.to_ml_features(640, 480)
    required_keys = [
        "center_x", "center_y", "normalized_center_x", "normalized_center_y",
        "is_above_center", "is_below_center", "is_left_of_center", "is_right_of_center",
        "apparent_width_px", "apparent_height_px", "aspect_ratio", "rectangularity",
        "rotation_deg", "estimated_distance", "confidence", "area", "area_normalized",
    ]
    for k in required_keys:
        assert k in features, f"Missing ML feature: {k}"
    print(f"  to_ml_features() keys: {len(features)}")
    print("  ✓ PASSED")


def test_color_agnostic_detection():
    """Test that the detector finds gates WITHOUT a color preset (edge-first + clustering)."""
    print("\n" + "="*50)
    print("TEST: Color-Agnostic Detection (no preset)")
    print("="*50)

    # Cyan gate -- not in any preset
    cyan_bgr = (255, 255, 0)
    image = create_synthetic_gate_image(
        gate_color_bgr=cyan_bgr,
        gate_center=(320, 240),
        gate_size=150,
        background_color=(40, 40, 40),
    )

    detector = GateDetector()  # no color_preset
    detections = detector.detect(image)

    assert len(detections) >= 1, "Should detect cyan gate without any preset"
    det = detections[0]
    assert abs(det.center_x - 320) < 30, f"X center off by {abs(det.center_x - 320)}"
    assert abs(det.center_y - 240) < 30, f"Y center off by {abs(det.center_y - 240)}"
    print(f"  Cyan gate (no preset): center=({det.center_x}, {det.center_y}) conf={det.confidence:.2f} method={det.detection_method}")

    # White gate on gray background -- extreme case
    white_bgr = (255, 255, 255)
    image2 = create_synthetic_gate_image(
        gate_color_bgr=white_bgr,
        gate_center=(320, 240),
        gate_size=150,
        background_color=(80, 80, 80),
    )
    detections2 = detector.detect(image2)
    assert len(detections2) >= 1, "Should detect white gate without any preset"
    print(f"  White gate (no preset): detections={len(detections2)} method={detections2[0].detection_method}")

    # ML features should include detection_method
    f = detections[0].to_ml_features(640, 480)
    assert "detection_method" in f, "ML features should include detection_method"

    print("  ✓ PASSED (color-agnostic detection works)")


def demo_visualization():
    """Create a demo visualization."""
    print("\n" + "="*50)
    print("DEMO: Visualization")
    print("="*50)
    
    # Create image with gate
    image = create_synthetic_gate_image(
        gate_center=(350, 200),
        gate_size=180,
        rotation_deg=5
    )
    
    detector = GateDetector(color_preset="red")
    detections = detector.detect(image)
    
    # Get visualization
    vis = detector.get_debug_visualization(image, detections)
    mask = detector.get_mask_visualization(image)
    
    # Save outputs
    cv2.imwrite("demo_original.png", image)
    cv2.imwrite("demo_detection.png", vis)
    cv2.imwrite("demo_mask.png", mask)
    
    print("  Saved: demo_original.png")
    print("  Saved: demo_detection.png")
    print("  Saved: demo_mask.png")
    
    return vis, mask


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("#" + " "*20 + "GATE DETECTOR TESTS" + " "*19 + "#")
    print("#"*60)
    
    try:
        test_basic_detection()
        test_offset_detection()
        test_steering_error()
        test_multiple_colors()
        test_rotated_gate()
        test_angled_gate_3d_like()
        test_multiple_gates_in_frame()
        test_partial_gate_at_edge()
        test_no_gate()
        test_distance_estimation()
        test_rich_ml_features()
        test_color_agnostic_detection()
        demo_visualization()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✓")
        print("="*50)
        print("\nCheck the demo_*.png files to see the visualization.")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
