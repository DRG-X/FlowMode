from datetime import datetime
import os
import cv2 as cv
import mediapipe as mp
import numpy as np

# =============================
# MODEL PATH (SAFE + CORRECT)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "blaze_face_short_range.tflite")

# =============================
# MEDIAPIPE TASKS (CORRECT API)
# =============================
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

BaseOptions = base_options.BaseOptions
FaceDetector = vision.FaceDetector
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

# =============================
# FACE DETECTOR
# =============================
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

detector = FaceDetector.create_from_options(options)

# =============================
# FACE MESH
# =============================
mp_faceMesh = mp.solutions.face_mesh
face_mesh = mp_faceMesh.FaceMesh()

landmark_points = [1, 152, 33, 263, 61, 291]

# =============================
# CAMERA
# =============================
cap = cv.VideoCapture(0)

# =============================
# FACE PRESENCE STATE
# =============================
presence_state = False
presence_candidate = None
presence_since = None

present_to_away_time = 2
away_to_present_time = 0.7

# =============================
# HEAD POSE STATE
# =============================
head_state = False  # False = DISTRACTED, True = ATTENTIVE
head_candidate = None
head_since = None

attentive_to_distracted_time = 2
distracted_to_attentive_time = 0.7

yaw_current = None
pitch_current = None

# =============================
# MAIN LOOP
# =============================
while True:
    now = datetime.now()
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    key = cv.waitKey(1) & 0xFF

    # =============================
    # FACE PRESENCE (BlazeFace)
    # =============================
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result_presence = detector.detect(mp_image)
    detected_presence = bool(result_presence.detections)

    if detected_presence:
        for det in result_presence.detections:
            bbox = det.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if detected_presence == presence_state:
        presence_candidate = None
        presence_since = None
    else:
        if presence_candidate is None:
            presence_candidate = detected_presence
            presence_since = now
        else:
            elapsed = (now - presence_since).total_seconds()
            threshold = present_to_away_time if presence_state else away_to_present_time
            if elapsed >= threshold:
                presence_state = presence_candidate
                presence_candidate = None
                presence_since = None

    presence_label = "PRESENT" if presence_state else "AWAY"
    cv.putText(frame, presence_label, (30, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # =============================
    # HEAD POSE (FaceMesh)
    # =============================
    if presence_state:
        result_mesh = face_mesh.process(rgb)
        if result_mesh.multi_face_landmarks:
            for lm in result_mesh.multi_face_landmarks:
                image_points = []

                for i in landmark_points:
                    pt = lm.landmark[i]
                    x = int(pt.x * width)
                    y = int(pt.y * height)
                    image_points.append((x, y))
                    cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

                image_points = np.array(image_points, dtype=np.float64)

                model_points = np.array([
                    (0.0, 0.0, 0.0),
                    (0.0, -330.0, -65.0),
                    (-225.0, 170.0, -135.0),
                    (225.0, 170.0, -135.0),
                    (-150.0, -150.0, -125.0),
                    (150.0, -150.0, -125.0)
                ], dtype=np.float64)

                focal_length = width
                center = (width / 2, height / 2)

                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)

                success, rvec, _ = cv.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    np.zeros((4, 1)),
                    flags=cv.SOLVEPNP_ITERATIVE
                )

                if success:
                    rmat, _ = cv.Rodrigues(rvec)
                    angles, *_ = cv.RQDecomp3x3(rmat)

                    pitch, yaw = angles[0], angles[1]

                    if key == ord('c'):
                        pitch_current = pitch
                        yaw_current = yaw

                    if yaw_current is None:
                        yaw_current = yaw
                        pitch_current = pitch

                    yaw_corr = yaw - yaw_current
                    pitch_corr = pitch - pitch_current

                    detected_head = (abs(yaw_corr) < 20) and (abs(pitch_corr) < 20)

                    if detected_head == head_state:
                        head_candidate = None
                        head_since = None
                    else:
                        if head_candidate is None:
                            head_candidate = detected_head
                            head_since = now
                        else:
                            elapsed = (now - head_since).total_seconds()
                            threshold = attentive_to_distracted_time if head_state else distracted_to_attentive_time
                            if elapsed >= threshold:
                                head_state = head_candidate
                                head_candidate = None
                                head_since = None

    head_label = "ATTENTIVE" if head_state else "DISTRACTED"
    cv.putText(frame, head_label, (30, 80),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv.imshow("FocusOS – Presence + Head Pose", frame)

    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()