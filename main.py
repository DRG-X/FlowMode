"""
Productivity Monitoring System
Detects user attention based on face presence, head pose, and eye gaze.

Author: Senior Developer
Date: 2026-02-07
"""

import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import logging
from typing import Optional, Tuple
import sys

# Try to import logging config, but make it optional
try:
    import log_config as log_config
    HAS_LOG_CONFIG = True
except ImportError:
    HAS_LOG_CONFIG = False
    print("Warning: Modules.log_config not found. Using basic logging.")


class FacePresenceDetector:
    """Detects if a face is present in the frame using MediaPipe Face Detection."""
    
    def __init__(self, model_path: str = "blaze_face_short_range.tflite"):
        self.model_path = model_path
        self.current_state = False
        self.candidate_state = None
        self.candidate_since = None
        self.present_to_away_time = 2.0
        self.away_to_present_time = 0.7
        
        # Initialize MediaPipe Face Detector
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        
        self.detector = FaceDetector.create_from_options(options)
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detect face presence with debouncing.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Tuple of (is_present, annotated_frame)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        
        detected_state = len(result.detections) > 0
        
        # Draw bounding boxes
        annotated_frame = frame.copy()
        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Debouncing logic
        now = datetime.now()
        
        if detected_state == self.current_state:
            self.candidate_state = None
            self.candidate_since = None
        else:
            if self.candidate_state is None:
                self.candidate_state = detected_state
                self.candidate_since = now
            else:
                if detected_state != self.candidate_state:
                    self.candidate_state = None
                    self.candidate_since = None
                else:
                    elapsed = (now - self.candidate_since).total_seconds()
                    threshold = self.present_to_away_time if self.current_state else self.away_to_present_time
                    
                    if elapsed >= threshold:
                        self.current_state = self.candidate_state
                        self.candidate_state = None
                        self.candidate_since = None
        
        return self.current_state, annotated_frame


class HeadPoseDetector:
    """Detects head pose (yaw and pitch) using MediaPipe Face Mesh."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark points for pose estimation
        self.landmark_points = [1, 152, 33, 263, 61, 291]
        
        # 3D model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip (1)
            (0.0, -330.0, -65.0),     # Chin (152)
            (-225.0, 170.0, -135.0),  # Left eye outer corner (33)
            (225.0, 170.0, -135.0),   # Right eye outer corner (263)
            (-150.0, -150.0, -125.0), # Left mouth corner (61)
            (150.0, -150.0, -125.0)   # Right mouth corner (291)
        ], dtype=np.float64)
        
        # Calibration
        self.yaw_baseline = None
        self.pitch_baseline = None
        self.is_calibrated = False
        
        # Debouncing
        self.current_state = False
        self.candidate_state = None
        self.candidate_since = None
        self.distracted_to_attentive_time = 0.3
        self.attentive_to_distracted_time = 0.7
        
        # Thresholds (degrees)
        self.yaw_threshold = 20
        self.pitch_threshold = 20
    
    def calibrate(self, yaw: float, pitch: float):
        """Set current pose as baseline."""
        self.yaw_baseline = yaw
        self.pitch_baseline = pitch
        self.is_calibrated = True
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Detect head pose and determine if user is attentive.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Tuple of (is_attentive, yaw, pitch)
        """
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        
        if not result.multi_face_landmarks:
            return self.current_state, None, None
        
        for facial_landmarks in result.multi_face_landmarks:
            # Extract 2D image points
            image_points = []
            for i in self.landmark_points:
                pt = facial_landmarks.landmark[i]
                x = int(pt.x * width)
                y = int(pt.y * height)
                image_points.append((x, y))
            
            image_points = np.array(image_points, dtype=np.float64)
            
            # Camera matrix
            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return self.current_state, None, None
            
            # Get rotation matrix and decompose
            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            pitch = angles[0]
            yaw = angles[1]
            
            # Auto-calibrate on first detection if not calibrated
            if not self.is_calibrated:
                self.calibrate(yaw, pitch)
            
            # Calculate corrected angles
            yaw_corr = yaw - self.yaw_baseline
            pitch_corr = pitch - self.pitch_baseline
            
            # Determine detected state
            detected_state = (abs(yaw_corr) < self.yaw_threshold) and (abs(pitch_corr) < self.pitch_threshold)
            
            # Debouncing logic
            now = datetime.now()
            
            if detected_state == self.current_state:
                self.candidate_state = None
                self.candidate_since = None
            else:
                if self.candidate_state is None:
                    self.candidate_state = detected_state
                    self.candidate_since = now
                else:
                    if detected_state != self.candidate_state:
                        self.candidate_state = None
                        self.candidate_since = None
                    else:
                        elapsed = (now - self.candidate_since).total_seconds()
                        threshold = self.attentive_to_distracted_time if self.current_state else self.distracted_to_attentive_time
                        
                        if elapsed >= threshold:
                            self.current_state = self.candidate_state
                            self.candidate_state = None
                            self.candidate_since = None
            
            return self.current_state, yaw, pitch
        
        return self.current_state, None, None


class EyeGazeDetector:
    """Detects eye gaze direction using iris tracking."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Important for iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Iris landmark points
        self.iris_points_left = [474, 475, 476, 477]
        self.iris_points_right = [469, 470, 471, 472]
        
        # Eye boundary landmarks
        self.left_eye_top = 159
        self.left_eye_bottom = 145
        self.right_eye_top = 386
        self.right_eye_bottom = 374
        
        # Calibration
        self.ref_eye_down_score = None
        self.is_calibrated = False
        
        # Smoothing
        self.eye_smooth = 0.0
        self.smoothing_factor = 0.85
        
        # Threshold
        self.distraction_threshold = -0.18
    
    def _calculate_iris_center_y(self, landmarks, iris_points: list, height: int) -> Optional[float]:
        """Calculate average Y coordinate of iris points (normalized)."""
        sum_y = 0
        for i in iris_points:
            pt_y = landmarks.landmark[i].y
            sum_y += pt_y
        return sum_y / len(iris_points)
    
    def _calculate_eye_down_score(self, landmarks, height: int, top_idx: int, 
                                  bottom_idx: int, center_y: float) -> Optional[float]:
        """Calculate how far down the iris is within the eye."""
        pt_top = landmarks.landmark[top_idx]
        pt_bottom = landmarks.landmark[bottom_idx]
        
        iris_offset = (center_y * height) - (pt_top.y * height)
        eye_height = (pt_bottom.y * height) - (pt_top.y * height)
        
        if eye_height != 0:
            return iris_offset / eye_height
        return None
    
    def calibrate(self, frame: np.ndarray):
        """Calibrate eye gaze based on current frame."""
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        
        if not result.multi_face_landmarks:
            return False
        
        for landmarks in result.multi_face_landmarks:
            left_center_y = self._calculate_iris_center_y(landmarks, self.iris_points_left, height)
            right_center_y = self._calculate_iris_center_y(landmarks, self.iris_points_right, height)
            
            if left_center_y is None or right_center_y is None:
                return False
            
            left_score = self._calculate_eye_down_score(landmarks, height, self.left_eye_top, 
                                                       self.left_eye_bottom, left_center_y)
            right_score = self._calculate_eye_down_score(landmarks, height, self.right_eye_top, 
                                                         self.right_eye_bottom, right_center_y)
            
            if left_score is not None and right_score is not None:
                self.ref_eye_down_score = (left_score + right_score) / 2
                self.eye_smooth = 0.0
                self.is_calibrated = True
                return True
        
        return False
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect eye gaze and determine if user is attentive.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Tuple of (is_attentive, smoothed_eye_score)
        """
        if not self.is_calibrated:
            return True, 0.0  # Default to attentive if not calibrated
        
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        
        if not result.multi_face_landmarks:
            return True, self.eye_smooth  # Keep previous state
        
        for landmarks in result.multi_face_landmarks:
            left_center_y = self._calculate_iris_center_y(landmarks, self.iris_points_left, height)
            right_center_y = self._calculate_iris_center_y(landmarks, self.iris_points_right, height)
            
            if left_center_y is None or right_center_y is None:
                return True, self.eye_smooth
            
            left_score = self._calculate_eye_down_score(landmarks, height, self.left_eye_top, 
                                                       self.left_eye_bottom, left_center_y)
            right_score = self._calculate_eye_down_score(landmarks, height, self.right_eye_top, 
                                                         self.right_eye_bottom, right_center_y)
            
            if left_score is not None and right_score is not None:
                final_score = (left_score + right_score) / 2
                calibrated_score = final_score - self.ref_eye_down_score
                
                # Apply smoothing
                self.eye_smooth = self.smoothing_factor * self.eye_smooth + (1 - self.smoothing_factor) * calibrated_score
                
                is_attentive = self.eye_smooth >= self.distraction_threshold
                return is_attentive, self.eye_smooth
        
        return True, self.eye_smooth


class ProductivityMonitor:
    """Main class that orchestrates all detection components."""
    
    def __init__(self, model_path: str = "blaze_face_short_range.tflite"):
        # Initialize detectors
        try:
            self.face_detector = FacePresenceDetector(model_path)
        except Exception as e:
            print(f"Error initializing face detector: {e}")
            print("Make sure 'blaze_face_short_range.tflite' is in the current directory.")
            sys.exit(1)
        
        self.head_detector = HeadPoseDetector()
        self.eye_detector = EyeGazeDetector()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)
        
        # Set camera resolution (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Logging
        if HAS_LOG_CONFIG:
            self.debug_log = log_config.setup_logger('microscope.log', logging.INFO)
            self.stats_log = log_config.setup_logger('dashboard.log', logging.INFO)
        else:
            self.debug_log = logging.getLogger('debug')
            self.stats_log = logging.getLogger('stats')
            logging.basicConfig(level=logging.INFO)
        
        # State tracking
        self.overall_state = "AWAY"
        self.last_logged_state = None
        
        # UI settings
        self.show_debug_info = True
        self.frame_counter = 0
    
    def log_state_change(self, new_state: str):
        """Log state changes."""
        if new_state != self.last_logged_state:
            now = datetime.now()
            self.debug_log.info(f"Productivity state changed to {new_state} at {now}")
            self.stats_log.info(f"Productivity state changed to {new_state} at {now}")
            self.last_logged_state = new_state
    
    def draw_status_panel(self, frame: np.ndarray, face_present: bool, head_aligned: bool, 
                         eyes_attentive: bool, yaw: Optional[float], pitch: Optional[float], 
                         eye_score: float) -> np.ndarray:
        """Draw comprehensive status panel on frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Status panel background
        cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        y_offset = 40
        line_height = 35
        
        # Title
        cv2.putText(frame, "PRODUCTIVITY MONITOR", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += line_height + 10
        
        # Face presence
        face_status = "PRESENT ✓" if face_present else "AWAY ✗"
        face_color = (0, 255, 0) if face_present else (0, 0, 255)
        cv2.putText(frame, f"Face: {face_status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
        y_offset += line_height
        
        # Head pose
        if face_present:
            head_status = "ALIGNED ✓" if head_aligned else "DISTRACTED ✗"
            head_color = (0, 255, 0) if head_aligned else (0, 165, 255)
            cv2.putText(frame, f"Head: {head_status}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, head_color, 2)
            y_offset += line_height
            
            # Show angles if available
            if yaw is not None and pitch is not None:
                cv2.putText(frame, f"  Yaw: {yaw:.1f}°  Pitch: {pitch:.1f}°", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += line_height - 5
            
            # Eye gaze
            if head_aligned:
                eye_status = "FOCUSED ✓" if eyes_attentive else "WANDERING ✗"
                eye_color = (0, 255, 0) if eyes_attentive else (0, 165, 255)
                cv2.putText(frame, f"Eyes: {eye_status}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
                y_offset += line_height
                
                if self.eye_detector.is_calibrated:
                    cv2.putText(frame, f"  Score: {eye_score:.3f}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_offset += line_height - 5
        
        y_offset += 10
        
        # Overall status
        if self.overall_state == "ATTENTIVE":
            status_text = "STATUS: ATTENTIVE ✓"
            status_color = (0, 255, 0)
        else:
            status_text = f"STATUS: {self.overall_state}"
            status_color = (0, 165, 255)
        
        cv2.putText(frame, status_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Instructions
        y_offset = height - 100
        cv2.putText(frame, "Controls:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, "C - Calibrate head & eyes", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(frame, "D - Toggle debug info", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(frame, "Q - Quit", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Calibration warnings
        if not self.head_detector.is_calibrated or not self.eye_detector.is_calibrated:
            warning_y = height - 200
            cv2.putText(frame, "⚠ PRESS 'C' TO CALIBRATE", (width // 2 - 200, warning_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, key: int) -> np.ndarray:
        """Process a single frame through the detection pipeline."""
        self.frame_counter += 1
        
        # Handle calibration
        if key == ord('c') or key == ord('C'):
            # Calibrate head pose
            _, yaw_temp, pitch_temp = self.head_detector.detect(frame)
            if yaw_temp is not None and pitch_temp is not None:
                self.head_detector.calibrate(yaw_temp, pitch_temp)
                print("✓ Head pose calibrated")
            
            # Calibrate eye gaze
            if self.eye_detector.calibrate(frame):
                print("✓ Eye gaze calibrated")
            else:
                print("✗ Eye gaze calibration failed - no face detected")
        
        # Toggle debug info
        if key == ord('d') or key == ord('D'):
            self.show_debug_info = not self.show_debug_info
        
        # Step 1: Check face presence
        face_present, annotated_frame = self.face_detector.detect(frame)
        
        head_aligned = False
        eyes_attentive = False
        yaw = None
        pitch = None
        eye_score = 0.0
        
        if face_present:
            # Step 2: Check head pose
            head_aligned, yaw, pitch = self.head_detector.detect(frame)
            
            if head_aligned:
                # Step 3: Check eye gaze
                eyes_attentive, eye_score = self.eye_detector.detect(frame)
            else:
                # Reset eye smoothing when head is not aligned
                self.eye_detector.eye_smooth = 0.0
        
        # Determine overall state
        if not face_present:
            self.overall_state = "AWAY"
        elif not head_aligned:
            self.overall_state = "HEAD DISTRACTED"
        elif not eyes_attentive:
            self.overall_state = "EYES DISTRACTED"
        else:
            self.overall_state = "ATTENTIVE"
        
        # Log state changes
        self.log_state_change(self.overall_state)
        
        # Draw status panel
        if self.show_debug_info:
            output_frame = self.draw_status_panel(annotated_frame, face_present, head_aligned, 
                                                  eyes_attentive, yaw, pitch, eye_score)
        else:
            output_frame = annotated_frame
            # Just show overall status
            status_color = (0, 255, 0) if self.overall_state == "ATTENTIVE" else (0, 165, 255)
            cv2.putText(output_frame, self.overall_state, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        
        return output_frame
    
    def run(self):
        """Main loop for the productivity monitor."""
        print("=" * 60)
        print("PRODUCTIVITY MONITORING SYSTEM")
        print("=" * 60)
        print("\nLook at the camera and press 'C' to calibrate")
        print("Press 'Q' to quit\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Get keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Process frame
                output_frame = self.process_frame(frame, key)
                
                # Display
                cv2.imshow("Productivity Monitor", output_frame)
                
                # Check for quit
                if key == ord('q') or key == ord('Q'):
                    print("\nShutting down...")
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released. Goodbye!")


def main():
    """Entry point for the application."""
    # You can specify a different model path here if needed
    monitor = ProductivityMonitor(model_path="blaze_face_short_range.tflite")
    monitor.run()


if __name__ == "__main__":
    main()
