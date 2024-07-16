from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance
import dlib,time,cv2,os,imutils
#from controlkeys import right_pressed,left_pressed,up_pressed,down_pressed
#from controlkeys import KeyOn, KeyOff
import pyautogui
def face_init(i,leftThresh,rightThresh,jaw_thresh,hzLipsThresh,lbroweyeThresh,rbroweyeThresh,browToEyeThresh,mouth,leftBrow,leftEye,rightBrow,rightEye,jaw,nose):
       leftThresh += distance.euclidean(mouth[49-49],jaw[5-1])
       rightThresh += distance.euclidean(mouth[55-49],jaw[13-1])
       hzLipsThresh+= distance.euclidean(mouth[49-49],mouth[55-49])
       lbroweyeThresh += (distance.euclidean(leftEye[43-43],leftBrow[25-23])+distance.euclidean(leftEye[43-43],leftBrow[26-23]))/2
       rbroweyeThresh += (distance.euclidean(rightEye[40-37],rightBrow[20-18])+distance.euclidean(rightEye[40-37],rightBrow[19-18]))/2
       jaw_thresh += distance.euclidean(jaw[1-1],jaw[17-1])
       if(i==31):
              jaw_thresh = jaw_thresh/30
              leftThresh = (leftThresh/30)/jaw_thresh          #divinding by jaw length to make things constant even if user is moving
              rightThresh = (rightThresh/30)/jaw_thresh
              hzLipsThresh = (hzLipsThresh/30)/jaw_thresh
              browToEyeThresh = ((lbroweyeThresh+rbroweyeThresh)/(30*2))/jaw_thresh
       return (leftThresh,rightThresh,jaw_thresh,hzLipsThresh,browToEyeThresh,lbroweyeThresh,rbroweyeThresh)
       

def hzLip_calc(mouth,jaw):
       d1 = distance.euclidean(mouth[49-49],mouth[55-49])  #see coords of face
       d1 = d1/distance.euclidean(jaw[1-1],jaw[17-1])
       return d1
def browEye_calc(leftEye,leftBrow,rightEye,rightBrow,jaw):
       d1 = (distance.euclidean(leftEye[43-43],leftBrow[25-23])+distance.euclidean(leftEye[43-43],leftBrow[26-23]))/2
       d2 = (distance.euclidean(rightEye[40-37],rightBrow[20-18])+distance.euclidean(rightEye[40-37],rightBrow[19-18]))/2
       d = (d1+d2)/2
       d = d/ distance.euclidean(jaw[1-1],jaw[17-1])
       return d

def left_calc(mouth,jaw):
       d = distance.euclidean(mouth[49-49],jaw[5-1])
       d = d/ distance.euclidean(jaw[1-1],jaw[17-1])
       return d

def right_calc(mouth,jaw):
       d = distance.euclidean(mouth[55-49],jaw[13-1])
       d = d/ distance.euclidean(jaw[1-1],jaw[17-1])
       return d


def start_detector():
    shape_predictor= r"C:\Users\Himanshu\Downloads\shape_predictor_68_face_landmarks.dat"
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    #vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    vs = VideoStream(src=0).start()
    """left_key_pressed=left_pressed
    right_key_pressed=right_pressed
    up_key_pressed=up_pressed
    down_key_pressed=down_pressed
    """
    time.sleep(2.0)
    
    
    hzLipsThresh = 0
    browToEyeThresh = 0
    leftThresh = 0
    rightThresh = 0
    #minframe_check = 5
    (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    (lbStart, lbEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eyebrow"]
    (rbStart, rbEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eyebrow"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]
    (nStart,nEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]

    flag_smile=0
    flag_browLift=0
    flag_left = 0
    flag_right = 0
    # loop over the frames from the video stream
    lbroweyeThresh = 0                   
    rbroweyeThresh = 0
    jaw_thresh = 0
    i=-1
    start_time = time.time()
    while True:
            # grab the frame from the threaded video stream, resize it to
            # have a maximum width of 400 pixels, and convert it to
            # grayscale
            frame = vs.read()
            frame = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            cv2.imshow("Frame", frame)
            for rect in rects:
                     # determine the facial landmarks for the face region, then
                     # convert the facial landmark (x, y)-coordinates to a NumPy arry
                     shape = predictor(gray, rect)
                     shape = face_utils.shape_to_np(shape)
                     for(x,y) in shape:
                           cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                     # shape has cooordinates which are mapped through our known points of eye and brows in 68_shape
                     leftEye = shape[leStart:leEnd]
                     rightEye = shape[reStart:reEnd]
                     leftBrow =  shape[lbStart:lbEnd]
                     rightBrow = shape[rbStart:rbEnd]
                     mouth = shape[mStart:mEnd]
                     jaw = shape[jStart:jEnd]
                     nose = shape[nStart:nEnd]
                     if(i==-1):
                      if time.time() - start_time < 3.0:
                            cv2.putText(frame, "KEEP A STILL FACE", (20,80),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 1)
                      else:
                            i+=1
                     elif(i<=31):
                            (leftThresh,rightThresh,jaw_thresh,hzLipsThresh,browToEyeThresh,
                             lbroweyeThresh,rbroweyeThresh) = face_init(i,leftThresh,rightThresh,jaw_thresh,hzLipsThresh,
                             lbroweyeThresh,rbroweyeThresh,browToEyeThresh,mouth,leftBrow,leftEye,rightBrow,rightEye,jaw,nose)
                            i+=1
                     else:
                            hzlip_dist = hzLip_calc(mouth,jaw)
                            browToEyeDist = browEye_calc(leftEye,leftBrow,rightEye,rightBrow,jaw)
                            leftDist = left_calc(mouth,jaw)
                            rightDist = right_calc(mouth,jaw)
                            cv2.putText(frame, "width of lips: {}".format(hzlip_dist), (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                            cv2.putText(frame, "Eye brow to Eye dist: {}".format(browToEyeDist), (5, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                            cv2.putText(frame, "left  {}".format(leftDist), (5, 110),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                            cv2.putText(frame, "right {}".format(rightDist), (5, 140),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                            
                            cv2.putText(frame, "lips thresh {}".format(hzLipsThresh), (5, 170),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                            cv2.putText(frame, "brows thresh {}".format(browToEyeThresh), (5, 200),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                            cv2.putText(frame, "left thresh {}".format(leftThresh), (5, 230),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                            cv2.putText(frame, "right thresh {}".format(rightThresh), (5, 250),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                            
                            #print([browToEyeDist,browToEyeThresh])
                            if leftDist >= (leftThresh+0.10):
                                   flag_left +=1
                                   if(flag_left>=1):
                                          print("left")
                                          cv2.putText(frame, "left", (50, 280),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                                          flag_left = 0
                                          flag_smile = 1
                                          flag_right = 1
                                          
                                          pyautogui.press("left")
                            else:
                                   flag_left = 0 

                            if rightDist >= (rightThresh+0.10):
                                   flag_right +=1
                                   if(flag_right>=1):
                                          print("right")
                                          cv2.putText(frame, "right", (50, 280),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                                          flag_right = 0
                                          pyautogui.press("right")
                            else:
                                   flag_right = 0 
                                          
                            if browToEyeDist >= (browToEyeThresh+0.022):
                                   #print("detected brow raise")
                                   flag_browLift+=1
                                   if(flag_browLift>=1):
                                          print("brow up")
                                          cv2.putText(frame, "brows up", (50, 280),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                                          flag_browLift = 0
                                          flag_smile = 1
                                          flag_right = 1
                                          flag_left = 1
                                          pyautogui.press("up")

                            else:
                                   flag_browLift = 0         
                                   
                            if hzlip_dist >= (hzLipsThresh+0.08):
                                   flag_smile +=1
                                   if(flag_smile>=1):
                                          print("smile")
                                          cv2.putText(frame, "smile", (50, 280),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                                          flag_smile = 0
                                          flag_right = 1
                                          flag_left = 1
                                          pyautogui.press("down")
                            else:
                                   flag_smile = 0 

                            

            cv2.imshow("Frame", frame)
        
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                    break
    
    # do a bit of cleanup
    VideoStream(src=0).stop()
    cv2.destroyAllWindows()


def main():
	start_detector()

if __name__ == '__main__':
	main()