from datetime import datetime
import cv2 as cv
import mediapipe as mp
import logging
import log_config

debug_log = log_config.setup_logger('microscope.log' , logging.INFO)



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

current_state = False
candidate_state = None
candidate_since = None

present_to_away_time = 2
away_to_present_time = 0.7


def update(frame , now) -> str:
    
    global current_state, candidate_state, candidate_since
    frame_rgb = cv.cvtColor(frame , cv.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = detector.detect(mp_image)
    if result.detections:
        detected_state = True  # Detected_state = True if face is detected and false if not detected
        for detection in result.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    else:
        # print("Face Not Detected ")
        detected_state = False
    


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
                    debug_log.info(f"Person's state changed to {log_label} at time {now}")

                    candidate_state = None
                    candidate_since = None
                    return log_label

    return "PRESENT" if current_state else "AWAY"

