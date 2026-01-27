import cv2 
import mediapipe as mp
import numpy as np 


capture = cv2.VideoCapture(0)

#FACE MESH

mp_faceMesh = mp.solutions.face_mesh
face_mesh = mp_faceMesh.FaceMesh()

landmark_points = [1, 152 , 33 , 263 , 61 , 291]
calibrate_warning = "Please Press C to calibrate"
'''
1 = nose tip 
152 = chin 
33 = left eye outer corner 
263 = right eye outer corner 
61 = left mouth corner 
291 = right mouth corner
'''

yaw_current = None
pitch_current = None

# yaw0 = None
# pitch0 = None

# calibration is to set the current pitch and yaw as baseline and then substract from it later.

while True:
    _ , frame = capture.read()
    height , width , _ = frame.shape
    # print(f"Height = {height} , width = {width}")
    rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)
    flipped_frame = cv2.flip(frame , 1)

    key = cv2.waitKey(1) & 0xFF


    # cv2.imshow("My Webcam in rgb ", rgb_frame)
    if result.multi_face_landmarks:
        for FacialLandmarks in result.multi_face_landmarks:

            image_points = []

            for i in landmark_points:
                pt_i = FacialLandmarks.landmark[i]
                x = int(pt_i.x * width)
                y = int(pt_i.y * height)

                image_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            image_points = np.array(image_points, dtype=np.float64)

            # 3D model points (approx face model)
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose tip (1)
                (0.0, -330.0, -65.0),     # Chin (152)
                (-225.0, 170.0, -135.0),  # Left eye outer corner (33)
                (225.0, 170.0, -135.0),   # Right eye outer corner (263)
                (-150.0, -150.0, -125.0), # Left mouth corner (61)
                (150.0, -150.0, -125.0)   # Right mouth corner (291)
            ], dtype=np.float64)

            focal_length = width
            center = (width / 2, height / 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                pitch = angles[0]
                yaw = angles[1]
                roll = angles[2]

                if(key == ord('c') or key == ord('C')):
                    pitch_current = pitch
                    yaw_current = yaw
                    calibrate_warning = " "

                if yaw_current is None:
                    yaw_current = yaw
                    pitch_current = pitch

                yaw_corr = yaw - yaw_current
                pitch_corr = pitch - pitch_current

                cv2.putText(flipped_frame, f"Yaw: {yaw:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(flipped_frame, f"Pitch: {pitch:.1f}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                attentive = (abs(yaw_corr) < 20) and (abs(pitch_corr) < 20)
                status = "ATTENTIVE ✅" if attentive else "DISTRACTED ❌"

                cv2.putText(flipped_frame, status, (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


    cv2.putText(flipped_frame , calibrate_warning , (30 , 180) , cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow("My Webcam in bgr" , flipped_frame)

    


    if(key == ord('q')):
        break


capture.release()
cv2.destroyAllWindows()