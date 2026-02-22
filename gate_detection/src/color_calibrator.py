"""
HSV Color Calibration Tool
===========================
Interactive tool to find the right HSV thresholds for gate detection.

Run this with a sample image or video from the simulator to tune
the color thresholds for your specific gate colors.

Usage:
    python color_calibrator.py --image path/to/gate_image.png
    python color_calibrator.py --video path/to/video.mp4
    python color_calibrator.py --camera 0  # Use webcam for testing
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


class HSVCalibrator:
    """Interactive HSV threshold calibration using trackbars."""
    
    def __init__(self):
        self.window_name = "HSV Calibrator"
        self.mask_window = "Mask"
        
        # Default values (start with broad range)
        self.h_low = 0
        self.h_high = 180
        self.s_low = 100
        self.s_high = 255
        self.v_low = 100
        self.v_high = 255
        
        self._setup_windows()
    
    def _setup_windows(self):
        """Create windows and trackbars."""
        cv2.namedWindow(self.window_name)
        cv2.namedWindow(self.mask_window)
        
        # Create trackbars
        cv2.createTrackbar("H Low", self.window_name, self.h_low, 180, self._on_h_low)
        cv2.createTrackbar("H High", self.window_name, self.h_high, 180, self._on_h_high)
        cv2.createTrackbar("S Low", self.window_name, self.s_low, 255, self._on_s_low)
        cv2.createTrackbar("S High", self.window_name, self.s_high, 255, self._on_s_high)
        cv2.createTrackbar("V Low", self.window_name, self.v_low, 255, self._on_v_low)
        cv2.createTrackbar("V High", self.window_name, self.v_high, 255, self._on_v_high)
    
    def _on_h_low(self, val): self.h_low = val
    def _on_h_high(self, val): self.h_high = val
    def _on_s_low(self, val): self.s_low = val
    def _on_s_high(self, val): self.s_high = val
    def _on_v_low(self, val): self.v_low = val
    def _on_v_high(self, val): self.v_high = val
    
    def get_thresholds(self):
        """Get current threshold values as numpy arrays."""
        lower = np.array([self.h_low, self.s_low, self.v_low])
        upper = np.array([self.h_high, self.s_high, self.v_high])
        return lower, upper
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply current thresholds to frame and return mask."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = self.get_thresholds()
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def run_on_image(self, image_path: str):
        """Run calibration on a static image."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        print("\n" + "="*50)
        print("HSV CALIBRATION MODE")
        print("="*50)
        print("Adjust the trackbars to isolate the gate color.")
        print("Press 'p' to print current values")
        print("Press 's' to save values to file")
        print("Press 'q' to quit")
        print("="*50 + "\n")
        
        while True:
            mask = self.process_frame(image)
            
            # Show results
            result = cv2.bitwise_and(image, image, mask=mask)
            
            # Add info overlay
            info_image = image.copy()
            lower, upper = self.get_thresholds()
            cv2.putText(info_image, f"Lower: {lower}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_image, f"Upper: {upper}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Find contours and draw them
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(info_image, contours, -1, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, info_image)
            cv2.imshow(self.mask_window, mask)
            cv2.imshow("Filtered", result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self._print_values()
            elif key == ord('s'):
                self._save_values()
        
        cv2.destroyAllWindows()
    
    def run_on_video(self, video_path: str):
        """Run calibration on a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video from {video_path}")
            return
        
        print("\n" + "="*50)
        print("HSV CALIBRATION MODE (Video)")
        print("="*50)
        print("Press 'p' to print current values")
        print("Press 's' to save values")
        print("Press 'space' to pause/unpause")
        print("Press 'q' to quit")
        print("="*50 + "\n")
        
        paused = False
        frame = None
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
            
            if frame is None:
                continue
            
            mask = self.process_frame(frame)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Add info overlay
            info_image = frame.copy()
            lower, upper = self.get_thresholds()
            cv2.putText(info_image, f"Lower: {lower}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_image, f"Upper: {upper}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if paused:
                cv2.putText(info_image, "PAUSED", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(self.window_name, info_image)
            cv2.imshow(self.mask_window, mask)
            cv2.imshow("Filtered", result)
            
            key = cv2.waitKey(30 if not paused else 1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self._print_values()
            elif key == ord('s'):
                self._save_values()
            elif key == ord(' '):
                paused = not paused
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_on_camera(self, camera_id: int = 0):
        """Run calibration using webcam (for testing without simulator)."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("\n" + "="*50)
        print("HSV CALIBRATION MODE (Camera)")
        print("="*50)
        print("Hold a colored object (like the gate color) in front of camera")
        print("Press 'p' to print current values")
        print("Press 's' to save values")
        print("Press 'q' to quit")
        print("="*50 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            mask = self.process_frame(frame)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Add crosshair at center
            h, w = frame.shape[:2]
            cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 1)
            cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 1)
            
            # Show HSV value at center
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            center_hsv = hsv[h//2, w//2]
            cv2.putText(frame, f"Center HSV: {center_hsv}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            lower, upper = self.get_thresholds()
            cv2.putText(frame, f"Lower: {lower}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Upper: {upper}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, frame)
            cv2.imshow(self.mask_window, mask)
            cv2.imshow("Filtered", result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self._print_values()
            elif key == ord('s'):
                self._save_values()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _print_values(self):
        """Print current threshold values in copy-paste format."""
        lower, upper = self.get_thresholds()
        print("\n" + "-"*40)
        print("Current HSV Thresholds:")
        print(f"  hsv_lower = np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
        print(f"  hsv_upper = np.array([{upper[0]}, {upper[1]}, {upper[2]}])")
        print("-"*40 + "\n")
    
    def _save_values(self, filename: str = "hsv_thresholds.txt"):
        """Save current values to a file."""
        lower, upper = self.get_thresholds()
        with open(filename, 'w') as f:
            f.write(f"# HSV Thresholds for Gate Detection\n")
            f.write(f"# Format: H_low, S_low, V_low, H_high, S_high, V_high\n")
            f.write(f"{lower[0]}, {lower[1]}, {lower[2]}, {upper[0]}, {upper[1]}, {upper[2]}\n")
        print(f"Saved thresholds to {filename}")


def create_test_image_with_gate(color: str = "red") -> np.ndarray:
    """Create a synthetic test image with a colored gate for testing."""
    
    # Create a dark background (simulating a race environment)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)  # Dark gray background
    
    # Define gate color in BGR
    colors = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "orange": (0, 165, 255),
        "green": (0, 255, 0),
    }
    gate_color = colors.get(color, (0, 0, 255))
    
    # Draw a square gate frame
    gate_outer = [(200, 140), (440, 140), (440, 340), (200, 340)]
    gate_inner = [(230, 170), (410, 170), (410, 310), (230, 310)]
    
    # Draw outer rectangle
    cv2.rectangle(image, gate_outer[0], gate_outer[2], gate_color, -1)
    # Cut out inner part (make it black/transparent)
    cv2.rectangle(image, gate_inner[0], gate_inner[2], (40, 40, 40), -1)
    
    # Add some noise to make it more realistic
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="HSV Color Calibration Tool")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, help="Camera ID (usually 0)")
    parser.add_argument("--test", type=str, choices=["red", "blue", "orange", "green"],
                       help="Generate a test image with specified gate color")
    
    args = parser.parse_args()
    
    calibrator = HSVCalibrator()
    
    if args.image:
        calibrator.run_on_image(args.image)
    elif args.video:
        calibrator.run_on_video(args.video)
    elif args.camera is not None:
        calibrator.run_on_camera(args.camera)
    elif args.test:
        # Create and save a test image
        test_image = create_test_image_with_gate(args.test)
        test_path = f"test_gate_{args.test}.png"
        cv2.imwrite(test_path, test_image)
        print(f"Created test image: {test_path}")
        calibrator.run_on_image(test_path)
    else:
        # Default: create a red test image
        print("No input specified. Creating a test image with a red gate...")
        test_image = create_test_image_with_gate("red")
        cv2.imwrite("test_gate_red.png", test_image)
        calibrator.run_on_image("test_gate_red.png")


if __name__ == "__main__":
    main()
