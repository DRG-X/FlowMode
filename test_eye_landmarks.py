import cv2 as cv
import mediapipe as mp
import numpy as np 

capture = cv.VideoCapture(0)

mp_faceMesh = mp.solutions.face_mesh
face_mesh = mp_faceMesh.FaceMesh(refine_landmarks = True)

iris_points = [470]

'''

469 = right_eye_left_corner
468 = right_eye_centre
471 = right_eye_right_corner
473 = left_eye_centre
474 = left_eye_left_corner
476 = left_eye_right_corner
472 = right_eye_bottom
470 = right_eye_top
475 = left_eye_top
477 = left_eye_bottom 

'''

while True:
    _ , frame = capture.read()

    height , width , channel = frame.shape

    frame_rgb = cv.cvtColor(frame , cv.COLOR_BGR2RGB)
    # frame_flipped = cv.flip(frame , 1)

    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        for FacialLandmarks in result.multi_face_landmarks:

            for i in iris_points:
                pt_i = FacialLandmarks.landmark[i]

                x = pt_i.x
                y = pt_i.y

                x_cordinate = int(x*width)
                y_cordinate = int(y*height)

                cv.circle(frame , (x_cordinate , y_cordinate) , 2, (0,255,255) , -1)

    cv.imshow("My webcam" , frame)
    if (cv.waitKey(1) == ord('q')):
        break


capture.release()
cv.destroyAllWindows()