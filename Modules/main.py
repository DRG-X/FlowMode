import cv2 as cv
import mediapipe as mp 
import numpy as np 
from datetime import datetime


import face_prescence_module
import head_pose_module
import eye_gaze_module



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
                # some logging here 
                # some maths here to find the time user was productive and shit
            else:
                continue
        else:
            continue
    else:
        continue
    
    if ( key == ord('q')):
        break


capture.release()
cv.destroyAllWindows()