import cv2 as cv
import mediapipe as mp
import numpy as np 

capture = cv.VideoCapture(0)

mp_faceMesh = mp.solutions.face_mesh
face_mesh = mp_faceMesh.FaceMesh(refine_landmarks = True)

iris_points_left_eye = [474, 475 , 476 , 477]
iris_points_right_eye = [469 , 470 , 471 , 472]

recalibrate_warning = "Press C to Recalibrate"
'''
469 = right_eye_left_corner
468 = right_eye_centre
471 = right_eye_right_corner
473 = left_eye_centre
474 = left_eye_left_corner
476 = left_eye_right_corner
'''
ref_eye_down_score = None


def center_eye_avg(iris_points : list):
    sum_y = 0
    if result.multi_face_landmarks:
        for FacialLandmarks in result.multi_face_landmarks:
            for i in iris_points:
                pt_i_y = (FacialLandmarks.landmark[i].y)
                sum_y += pt_i_y

            center_eye_avg_pt = sum_y /4
            # x_cor = int(center_eye_avg_pt.x * width)
            # y_cor = int(center_eye_avg.y * height)
            return center_eye_avg_pt # already a y co-ordinate

def calc_eye_down_score(height  , top_eye , bottom_eye , centre_eye):
    
    if result.multi_face_landmarks:
        for FacialLandmarks in result.multi_face_landmarks:

            pt_top_eye = FacialLandmarks.landmark[int(top_eye)]
            pt_bottom_eye = FacialLandmarks.landmark[int(bottom_eye)]
            

            iris_offset = (   (centre_eye * height)   ) - (   (pt_top_eye.y * height)  )
            eye_height =  (  (pt_bottom_eye.y * height)  ) - (  (pt_top_eye.y * height) )

            if eye_height != 0 :
                eye_down_score = iris_offset / eye_height
                return eye_down_score

# Eye down Score = iris_offset / eye_height 
# iris_offset = iris_y - top_y

while True:
    _ , frame = capture.read()

    height , width , channel = frame.shape

    cv.putText(frame , recalibrate_warning , (30 , 180) ,cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2 )
    frame_rgb = cv.cvtColor(frame , cv.COLOR_BGR2RGB)
    # frame_flipped = cv.flip(frame , 1)
    result = face_mesh.process(frame_rgb)

    key = cv.waitKey(1) & 0xFF
    left_eye_y_co_centre = center_eye_avg(iris_points_left_eye)
    right_eye_y_co_centre = center_eye_avg(iris_points_right_eye)
    left_eye_score = calc_eye_down_score(height , 159 , 145 , left_eye_y_co_centre)
    right_eye_score = calc_eye_down_score(height , 386 , 374 , right_eye_y_co_centre )

    if key == ord('c') or key == ord("C"):


        

        if left_eye_score is not None and right_eye_score is not None:
            ref_eye_down_score = ((calc_eye_down_score(height , 159 , 145 , left_eye_y_co_centre)) + (calc_eye_down_score(height , 386 , 374 , right_eye_y_co_centre )))/2
            recalibrate_warning = ""

    # if left_eye_score is not None and right_eye_score is not None:
    if ref_eye_down_score is not None and left_eye_score is not None and right_eye_score is not None:
        final_eye_score = (left_eye_score + right_eye_score) / 2
        callibrated_eye_down_score = final_eye_score - ref_eye_down_score
        print(callibrated_eye_down_score)

        if  callibrated_eye_down_score < -0.18: 
            cv.putText(frame, "Distracted Eyes", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            print("Distracted Eyes")
        else:
            cv.putText(frame, "Attentive Eyes", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            print("attentive Eyes")
    else:
        cv.putText(frame, "Press C to calibrate", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # print(left_eye_score)
    # print(right_eye_score)
    # print(final_eye_score)

    cv.imshow("My webcam" , frame)
    if (key == ord('q')):
        break


capture.release()
cv.destroyAllWindows()