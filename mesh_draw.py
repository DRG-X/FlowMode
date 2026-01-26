import cv2
import mediapipe as mp
import time

MODEL_PATH = "models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------- Globals for results ----------
latest_result = None

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


# ---------- Create Landmarker ----------
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=on_result,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

landmarker = FaceLandmarker.create_from_options(options)

# ---------- Drawing utils ----------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh  # only for connection constants (FACEMESH_TESSELATION etc.)

IMPORTANT_LANDMARKS = {
    "nose_tip": 1,
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "chin": 152
}

def draw_important_points(frame_bgr, face_landmarks, color=(0, 255, 0)):
    h, w = frame_bgr.shape[:2]

    for name, idx in IMPORTANT_LANDMARKS.items():
        lm = face_landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(frame_bgr, (x, y), 4, color, -1)
        cv2.putText(frame_bgr, name, (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ---------- FPS setup ----------
prev_time = time.time()
fps = 0.0

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)  # mirror view (optional but feels natural)

    # Feed frame to landmarker async
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    # ---------- Draw results if available ----------
    if latest_result is not None and len(latest_result.face_landmarks) > 0:
        # Get first face
        face_landmarks_list = latest_result.face_landmarks[0]

        # Convert to "NormalizedLandmarkList" so drawing_utils can draw it
        normalized_landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for lm in face_landmarks_list:
            normalized_landmark_list.landmark.add(x=lm.x, y=lm.y, z=lm.z)

        # Draw mesh connections
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=normalized_landmark_list,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # Draw contours (face outline, lips, eyes)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=normalized_landmark_list,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

        # Draw irises (optional cool detail)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=normalized_landmark_list,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )

        # Mark important points
        draw_important_points(frame, face_landmarks_list, color=(255, 255, 0))

    # ---------- FPS calc ----------
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time
    fps = 1.0 / dt if dt > 0 else fps

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face Mesh (MediaPipe Tasks)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
