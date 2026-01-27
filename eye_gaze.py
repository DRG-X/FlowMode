import cv2 as cv
import mediapipe as mp
import numpy as np 

capture = cv.VideoCapture(0)

mp_faceMesh = mp.solutions.face_mesh
face_mesh = mp_faceMesh.FaceMesh(refine_landmarks = True)

iris_points = [469 , 471 , 474 , 476]

'''
469 = right_eye_left_corner
468 = right_eye_centre
471 = right_eye_right_corner
473 = left_eye_centre
474 = left_eye_left_corner
476 = left_eye_right_corner
'''


def calc_eye_down_score(height  , top_eye , bottom_eye , centre_eye):
    result = face_mesh.process(frame_rgb)
    if result.multi_face_landmarks:
        for FacialLandmarks in result.multi_face_landmarks:

            pt_top_eye = FacialLandmarks.landmark[top_eye]
            pt_bottom_eye = FacialLandmarks.landmark[bottom_eye]
            pt_centre_eye = FacialLandmarks.landmark[centre_eye]

            iris_offset = (   int(pt_centre_eye.y * height)   ) - (   int(pt_top_eye.y * height)  )
            eye_height =  (  int(pt_bottom_eye.y * height)  ) - (  int(pt_top_eye.y * height) )

            eye_down_score = iris_offset / eye_height
            return eye_down_score

# Eye down Score = iris_offset / eye_height 
# iris_offset = iris_y - top_y

while True:
    _ , frame = capture.read()

    height , width , channel = frame.shape

    frame_rgb = cv.cvtColor(frame , cv.COLOR_BGR2RGB)
    # frame_flipped = cv.flip(frame , 1)

    left_eye_score = calc_eye_down_score(height , 475 , 477 , 473)
    right_eye_score = calc_eye_down_score(height , 470 , 472 , 468 )

    if left_eye_score is not None and right_eye_score is not None:
        final_eye_score = (left_eye_score + right_eye_score) / 2

        if final_eye_score > 0.65:
            cv.putText(frame , "Distracted Eyes" , (20 , 110) , cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        else:
            cv.putText(frame , "Attentive Eyes" , (20 , 110) , cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # print(left_eye_score)
    # print(right_eye_score)
    # print(final_eye_score)

    cv.imshow("My webcam" , frame)
    if (cv.waitKey(1) == ord('q')):
        break


capture.release()
cv.destroyAllWindows()