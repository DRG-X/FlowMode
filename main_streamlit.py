"""
Productivity Monitoring System with Session Analytics (Streamlit Version)
Detects user attention based on face presence, head pose, and eye gaze.
Optimized for Streamlit UI integration.

Author: Senior Developer
Date: 2026-02-08
"""

import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import logging
from typing import Optional, Tuple, Dict
import sys
import csv
import os
from collections import defaultdict
import threading
import time

# Try to import logging config, but make it optional
try:
    import log_config as log_config
    HAS_LOG_CONFIG = True
except ImportError:
    HAS_LOG_CONFIG = False


class SessionAnalytics:
    """Tracks and analyzes attention metrics for a session."""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.session_end = None
        
        # State tracking
        self.current_state = "AWAY"
        self.state_start_time = datetime.now()
        
        # Time tracking (in seconds)
        self.state_durations = defaultdict(float)
        
        # Detailed state history for analysis
        self.state_history = []
        
        # Timeline for time-series graph (state at each timestamp)
        self.timeline = []
        
        # Longest continuous periods
        self.longest_attentive_period = 0.0
        self.longest_distracted_period = 0.0
        self.current_period_start = datetime.now()
        self.current_period_state = "AWAY"
        
        # Frame counter for sampling
        self.total_frames = 0
        self.attentive_frames = 0
    
    def record_timeline_point(self):
        """Record current state for time-series visualization."""
        self.timeline.append({
            'timestamp': datetime.now(),
            'state': self.current_state,
            'is_attentive': self.current_state == "ATTENTIVE"
        })
    
    def update_state(self, new_state: str):
        """Update the current state and track durations."""
        now = datetime.now()
        
        # Calculate duration of previous state
        duration = (now - self.state_start_time).total_seconds()
        self.state_durations[self.current_state] += duration
        
        # Record state transition
        self.state_history.append({
            'state': self.current_state,
            'start_time': self.state_start_time,
            'end_time': now,
            'duration': duration
        })
        
        # Track longest continuous periods
        if self.current_state == "ATTENTIVE":
            period_duration = (now - self.current_period_start).total_seconds()
            if period_duration > self.longest_attentive_period:
                self.longest_attentive_period = period_duration
        elif self.current_state in ["AWAY", "HEAD DISTRACTED", "EYES DISTRACTED"]:
            period_duration = (now - self.current_period_start).total_seconds()
            if period_duration > self.longest_distracted_period:
                self.longest_distracted_period = period_duration
        
        # Update state
        self.current_state = new_state
        self.state_start_time = now
        
        # Reset period tracking if state category changed
        if (new_state == "ATTENTIVE" and self.current_period_state != "ATTENTIVE") or \
           (new_state != "ATTENTIVE" and self.current_period_state == "ATTENTIVE"):
            self.current_period_start = now
            self.current_period_state = new_state
    
    def record_frame(self, is_attentive: bool):
        """Record frame-level data for additional accuracy."""
        self.total_frames += 1
        if is_attentive:
            self.attentive_frames += 1
    
    def end_session(self):
        """Finalize the session and calculate final metrics."""
        self.session_end = datetime.now()
        
        # Add final state duration
        duration = (self.session_end - self.state_start_time).total_seconds()
        self.state_durations[self.current_state] += duration
        
        self.state_history.append({
            'state': self.current_state,
            'start_time': self.state_start_time,
            'end_time': self.session_end,
            'duration': duration
        })
    
    def get_total_duration(self) -> float:
        """Get total session duration in seconds."""
        if self.session_end:
            return (self.session_end - self.session_start).total_seconds()
        return (datetime.now() - self.session_start).total_seconds()
    
    def get_attentive_time(self) -> float:
        """Get total attentive time in seconds."""
        return self.state_durations.get("ATTENTIVE", 0.0)
    
    def get_distracted_time(self) -> float:
        """Get total distracted time (all non-attentive states) in seconds."""
        total_distracted = 0.0
        for state, duration in self.state_durations.items():
            if state != "ATTENTIVE":
                total_distracted += duration
        return total_distracted
    
    def get_attentiveness_score(self) -> float:
        """Calculate attentiveness percentage (0-100)."""
        total = self.get_total_duration()
        if total == 0:
            return 0.0
        attentive = self.get_attentive_time()
        return (attentive / total) * 100
    
    def get_breakdown_by_state(self) -> Dict[str, float]:
        """Get time breakdown by each state in seconds."""
        return dict(self.state_durations)
    
    def format_duration(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def get_summary_dict(self) -> Dict:
        """Get session summary as a dictionary for Streamlit display."""
        total_seconds = self.get_total_duration()
        attentive_seconds = self.get_attentive_time()
        distracted_seconds = self.get_distracted_time()
        score = self.get_attentiveness_score()
        
        return {
            'session_start': self.session_start,
            'session_end': self.session_end,
            'total_duration_seconds': total_seconds,
            'total_duration_formatted': self.format_duration(total_seconds),
            'attentive_seconds': attentive_seconds,
            'attentive_formatted': self.format_duration(attentive_seconds),
            'distracted_seconds': distracted_seconds,
            'distracted_formatted': self.format_duration(distracted_seconds),
            'attentiveness_score': score,
            'longest_attentive_period': self.longest_attentive_period,
            'longest_attentive_formatted': self.format_duration(self.longest_attentive_period),
            'longest_distracted_period': self.longest_distracted_period,
            'longest_distracted_formatted': self.format_duration(self.longest_distracted_period),
            'state_breakdown': self.get_breakdown_by_state(),
            'timeline': self.timeline,
            'state_history': self.state_history
        }
    
    def save_to_csv(self, filename: str = "session_history.csv"):
        """Save session summary to CSV file."""
        file_exists = os.path.isfile(filename)
        
        total_seconds = self.get_total_duration()
        attentive_seconds = self.get_attentive_time()
        distracted_seconds = self.get_distracted_time()
        score = self.get_attentiveness_score()
        
        # Prepare row data
        row_data = {
            'session_start': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'session_end': self.session_end.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration_seconds': round(total_seconds, 2),
            'total_duration_formatted': self.format_duration(total_seconds),
            'attentive_seconds': round(attentive_seconds, 2),
            'attentive_time_formatted': self.format_duration(attentive_seconds),
            'distracted_seconds': round(distracted_seconds, 2),
            'distracted_time_formatted': self.format_duration(distracted_seconds),
            'attentiveness_score': round(score, 2),
            'longest_attentive_period_seconds': round(self.longest_attentive_period, 2),
            'longest_attentive_period_formatted': self.format_duration(self.longest_attentive_period),
            'longest_distracted_period_seconds': round(self.longest_distracted_period, 2),
            'longest_distracted_period_formatted': self.format_duration(self.longest_distracted_period),
        }
        
        # Add state breakdown
        breakdown = self.get_breakdown_by_state()
        for state, duration in breakdown.items():
            state_key = state.lower().replace(' ', '_')
            row_data[f'{state_key}_seconds'] = round(duration, 2)
            row_data[f'{state_key}_formatted'] = self.format_duration(duration)
        
        # Write to CSV
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row_data)


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


class ProductivityMonitorStreamlit:
    """Main class for Streamlit integration - returns frames instead of displaying them."""
    
    def __init__(self, model_path: str = "blaze_face_short_range.tflite"):
        # Initialize detectors
        try:
            self.face_detector = FacePresenceDetector(model_path)
        except Exception as e:
            raise Exception(f"Error initializing face detector: {e}")
        
        self.head_detector = HeadPoseDetector()
        self.eye_detector = EyeGazeDetector()
        
        # Camera
        self.cap = None
        
        # Session analytics
        self.analytics = SessionAnalytics()
        
        # State tracking
        self.overall_state = "AWAY"
        self.last_logged_state = None
        
        # Control flags
        self.is_running = False
        self.should_calibrate = False
        
        # Current frame
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Timeline recording interval (record every N frames)
        self.timeline_record_interval = 30  # Record every 30 frames (~1 second at 30fps)
        self.frame_counter = 0
    
    def start_camera(self):
        """Initialize camera."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def stop_camera(self):
        """Release camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def calibrate(self):
        """Trigger calibration on next frame."""
        self.should_calibrate = True
    
    def draw_status_panel(self, frame: np.ndarray, face_present: bool, head_aligned: bool, 
                         eyes_attentive: bool, yaw: Optional[float], pitch: Optional[float], 
                         eye_score: float) -> np.ndarray:
        """Draw comprehensive status panel on frame with session stats."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Status panel background
        cv2.rectangle(overlay, (10, 10), (450, 340), (0, 0, 0), -1)
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
        
        # Session stats (compact)
        y_offset += line_height + 15
        session_duration = self.analytics.get_total_duration()
        session_score = self.analytics.get_attentiveness_score()
        
        cv2.putText(frame, f"Session: {self.analytics.format_duration(session_duration)}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += 25
        cv2.putText(frame, f"Score: {session_score:.1f}%", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Calibration warnings
        if not self.head_detector.is_calibrated or not self.eye_detector.is_calibrated:
            warning_y = height - 80
            cv2.putText(frame, "⚠ CLICK CALIBRATE BUTTON", (width // 2 - 220, warning_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame and return annotated frame and status."""
        self.frame_counter += 1
        
        # Handle calibration (triggered from Streamlit UI)
        if self.should_calibrate:
            # Calibrate head pose
            _, yaw_temp, pitch_temp = self.head_detector.detect(frame)
            if yaw_temp is not None and pitch_temp is not None:
                self.head_detector.calibrate(yaw_temp, pitch_temp)
            
            # Calibrate eye gaze
            self.eye_detector.calibrate(frame)
            
            self.should_calibrate = False
        
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
        
        # Update analytics
        if self.overall_state != self.last_logged_state:
            self.analytics.update_state(self.overall_state)
            self.last_logged_state = self.overall_state
        
        # Record frame
        is_attentive = (self.overall_state == "ATTENTIVE")
        self.analytics.record_frame(is_attentive)
        
        # Record timeline point periodically
        if self.frame_counter % self.timeline_record_interval == 0:
            self.analytics.record_timeline_point()
        
        # Draw status panel
        output_frame = self.draw_status_panel(annotated_frame, face_present, head_aligned, 
                                              eyes_attentive, yaw, pitch, eye_score)
        
        # Status info for Streamlit
        status = {
            'state': self.overall_state,
            'face_present': face_present,
            'head_aligned': head_aligned,
            'eyes_attentive': eyes_attentive,
            'is_calibrated': self.head_detector.is_calibrated and self.eye_detector.is_calibrated,
            'session_duration': self.analytics.get_total_duration(),
            'attentiveness_score': self.analytics.get_attentiveness_score()
        }
        
        return output_frame, status
    
    def is_tracking_active(self):
        """Check if tracking is still active."""
        return self.is_running
    
    def get_current_frame(self):
        """Get the current processed frame (thread-safe)."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def run_loop(self):
        """Main processing loop (runs in background thread)."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            output_frame, _ = self.process_frame(frame)
            
            # Display in OpenCV window
            cv2.imshow("Productivity Monitor", output_frame)
            
            # Check for window close or 'Q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                self.is_running = False
                break
            
            # Check if window was closed by user (clicking X)
            if cv2.getWindowProperty("Productivity Monitor", cv2.WND_PROP_VISIBLE) < 1:
                self.is_running = False
                break
            
            # Store frame (thread-safe)
            with self.frame_lock:
                self.current_frame = output_frame
            
            # Small delay to prevent CPU overuse
            time.sleep(0.01)
    
    def start_tracking(self):
        """Start the tracking session."""
        if not self.is_running:
            self.start_camera()
            self.analytics = SessionAnalytics()  # Reset analytics
            self.is_running = True
            
            # Start processing in background thread
            self.thread = threading.Thread(target=self.run_loop, daemon=True)
            self.thread.start()
    
    def stop_tracking(self):
        """Stop the tracking session."""
        if self.is_running:
            self.is_running = False
            if hasattr(self, 'thread'):
                self.thread.join(timeout=2.0)
            
            self.analytics.end_session()
            
            # Close OpenCV window
            cv2.destroyAllWindows()
            
            self.stop_camera()
    
    def get_session_summary(self):
        """Get session summary for display."""
        return self.analytics.get_summary_dict()
    
    def save_session(self):
        """Save session to CSV."""
        self.analytics.save_to_csv()
