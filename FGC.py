import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
import time
cam = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


screen_w, screen_h = pydirectinput.size()
pydirectinput.FAILSAFE = False
mode = 2
control = ""
while True:
    success, image = cam.read() 
    image = cv2.flip(image, 1)
    start = time.time()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    landmark_points = output.multi_face_landmarks
    image_h, image_w, image_c = image.shape
    face_2d = []
    face_3d = []
    left_eye = []
    
    if landmark_points:               
            landmarks = landmark_points[0].landmark
            for idx, lm in enumerate(landmarks):
                
                if idx in [145,159]:
                    x = int(lm.x * image_w)   
                    y = int(lm.y * image_h)
                    cv2.circle(image, (x,y), 3, (0, 255, 0))
                    
                    left_eye.append(lm.y)
                #right corner of the right eye,left corner of the left eye,nose tip, corners of the mouth,chin
                if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                    
                    if idx ==1:  #nose tip
                        nose_2d = (lm.x * image_w,lm.y * image_h)
                        nose_3d = (lm.x * image_w,lm.y * image_h,lm.z * image_c)
                        x = int(lm.x * screen_w)   
                        y = int(lm.y * screen_h)        #did screen instead of image as it will go
                                                            # default pos of cursor would go to left where my vid is displaying
                        #print([x,y])
                        pydirectinput.moveTo(x,y)
                        

                    x,y = int(lm.x * image_w),int(lm.y * image_h)
                    cv2.circle(image, (x,y), 3, (255, 255, 0))
                    face_2d.append([x,y])
                    face_3d.append(([x,y,lm.z]))
                
             
            if (left_eye[0]-left_eye[1]<0.004):
                control = "click"
                pydirectinput.click()
                if mode==2:
                    pydirectinput.press('space')   
                else:
                    pydirectinput.press('x')       
               
            
            face_2d = np.array(face_2d,dtype=np.float64)

            face_3d = np.array(face_3d,dtype=np.float64)

            cam_matrix = np.array([[image_w,0,image_h/2],
                                  [0,image_w,image_w/2],
                                  [0,0,1]])
            distortion_matrix = np.zeros((4,1),dtype=np.float64)

            success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


            #getting rot matrix out of rot vector using rodrigues (rmat here)
            rmat,_ = cv2.Rodrigues(rotation_vec)
            # get euler angles using RQDecomp
            angles,_,_,_,_,_ = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            #print([x,y,z])
            if mode == 2:
                if y < -10:                         #yaw about y axis (left right)
                    control = "left"
                    pydirectinput.press('a')
                if y > 10:
                    control="right"
                    pydirectinput.press('d')
                
                if x < -10:                       #pitch about x axis (up and down)
                    control="down"
                    pydirectinput.press('s')

                if x > 17:
                    control = "up"
                    pydirectinput.press('w')

            #3D Controls
            else:
                if y<-20:
                    control = "tilt left"            #move left and tilt complete left
                    pydirectinput.moveTo(20,330)
                    pydirectinput.press('w')
                """if y < -10:                         #yaw about y axis (left right)
                    print("left")
                    pydirectinput.press('a')"""
                if y>15:
                    control = "tilt right"              #move right and tilt complete right
                    pydirectinput.moveTo(1200,400)
                    pydirectinput.press('w')
                """if y > 10:
                    print("right")
                    pydirectinput.press('d') """
                
                if x < -10:                       #pitch about x axis (up and down)
                    control = "down"
                    pydirectinput.press('c')
                    
                    
                elif x > 15:
                    control = "up"
                    pydirectinput.press('j')

    cv2.putText(image,"{}".format(control) , (20,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)   
    cv2.putText(image,"mode: {}D".format(mode) , (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)    
    cv2.imshow('face game controller',image)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('3'):
        mode = 3
    elif key == ord('2'):
        mode = 2
    elif key ==ord('q'):
        break
cam.release() 


