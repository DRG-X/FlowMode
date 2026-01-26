from datetime import datetime
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from log_config import logger


MODEL_PATH = "blaze_face_short_range.tflite"



BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

detector = FaceDetector.create_from_options(options)
cap = cv2.VideoCapture(0)
frame_counter = 0

current_state = False
candidate_state = None
candidate_since = None


present_to_away_time = 1
away_to_present_time = 0.7


while True:
    ret, frame = cap.read()
    frame_counter += 1
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)
    # print(result)
    if result.detections:
        detected_state = True  # Detected_state = True if face is detected and false if not detected
        for detection in result.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    else:
        # print("Face Not Detected ")
        detected_state = False
    

    now = datetime.now()

    if detected_state == current_state:
        # stable -> cancel any pending switch
        candidate_state = None
        candidate_since = None
    else:
        # detector suggests a different state
        if candidate_state is None:
            # start candidate timer
            candidate_state = detected_state
            candidate_since = now
        else:
            # already waiting to switch
            if detected_state != candidate_state:
                # noisy flip -> restart
                candidate_state = None
                candidate_since = None
            else:
                elapsed = (now - candidate_since).total_seconds()

                # choose threshold based on direction
                if current_state is True and candidate_state is False:
                    threshold = present_to_away_time
                else:
                    threshold = away_to_present_time

                if elapsed >= threshold:
                    current_state = candidate_state
                    log_label = "PRESENT" if current_state else "AWAY"
                    logger.info(f"Person's state changed to {log_label} at time {now}")
                    
                    candidate_state = None
                    candidate_since = None
    
    label = "PRESENT" if current_state else "AWAY"
    # logger.info(f"Current_state was {label} at {now}")
    cv2.putText(frame , label , (30,40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,255) , 2)

    cv2.imshow("MediaPipe Tasks Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
