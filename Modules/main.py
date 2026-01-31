import cv2 as cv
import mediapipe as mp 
import numpy as np 
from datetime import datetime
import logging 
import  log_config



import face_prescence_module
import head_pose_module
import eye_gaze_module


debug_log = log_config.setup_logger('microscope.log' , logging.INFO)
stats_log = log_config.setup_logger('dashboard.log' , logging.INFO)

capture = cv.VideoCapture(0)

while True:
    ret , frame  = capture.read()
    now = datetime.now()
    flipped_frame = cv.flip(frame , 1)
    cv.imshow("FocusOS V1", flipped_frame)
    if not ret:
        break

    key = cv.waitKey(1) & 0xFF
    
    presence_label = face_prescence_module.update(frame , now)
    if presence_label == "PRESENT":
        head_pose_label = head_pose_module.update(frame , now , key)
        if head_pose_label == "ATTENTIVE":
            eye_gaze_label = eye_gaze_module.update(frame , key , now)
            if eye_gaze_label == "attentive Eyes":
                print("User is really active and doing some productive work")
                debug_log.info(f"Doing Productive work at:{now}")
                stats_log.info(f"Doing Productive work at:{now}")
                # some logging here 
                # some maths here to find the time user was productive and shit
            else:
                debug_log.info(f"Doing Productive work at:{now}")
                stats_log.info(f"Doing Productive work at:{now}")
                continue
        else:
            debug_log.info(f"Doing Productive work at:{now}")
            stats_log.info(f"Doing Productive work at:{now}")
            continue
    else:
        debug_log.info(f"Doing Productive work at:{now}")
        stats_log.info(f"Doing Productive work at:{now}")
        continue
    
    if ( key == ord('q')):
        break


capture.release()
cv.destroyAllWindows()