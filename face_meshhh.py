import cv2 
import mediapipe as mp

capture = cv2.VideoCapture(0)

#FACE MESH

mp_faceMesh = mp.solutions.face_mesh
face_mesh = mp_faceMesh.FaceMesh()

landmark_points = [1, 152 , 33 , 263 , 61 , 291]

'''
1 = nose tip 
152 = chin 
33 = left eye outer corner 
263 = right eye outer corner 
61 = left mouth corner 
291 = right mouth corner
'''




while True:
    _ , frame = capture.read()
    height , width , _ = frame.shape
    # print(f"Height = {height} , width = {width}")
    rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)


    # cv2.imshow("My Webcam in rgb ", rgb_frame)
    if result.multi_face_landmarks:
        for FacialLandmarks in result.multi_face_landmarks:

            for i in landmark_points:

                pt_i = FacialLandmarks.landmark[i]
                x = int(pt_i.x * width)
                y = int(pt_i.y * height)


                cv2.circle(frame , (x,y) , 2 , (0 , 0 , 255) , -1)
            # print(f"x = {x} , y = {y}")

    cv2.imshow("My Webcam in bgr" , frame)
    if(cv2.waitKey(1) == ord('q')):
        break


capture.release()
cv2.destroyAllWindows()