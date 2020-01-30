# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import cv2.aruco as aruco


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 180
camera.iso = 1600
#camera.exposure_mode = "sports"
camera.resolution = (640, 480)
camera.framerate = 30

#define the camera distortion arrays
cam_matrix = np.array([[309.65140551, 0, 299.7942552], [0, 309.63299386, 236.80161718], [ 0, 0, 1]])
dist_coef = np.array([-0.32061628, 0.13711123, 0.0058947, 0.00258218, -0.03117783])

rawCapture = PiRGBArray(camera, size=(640, 480))

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters_create()

i = 0
mean = 0
fps_max = 0
fps_min = 1000

def draw_fps(fps):
    # print fps
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (470, 50)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(gray,'FPS: ' + str(int(fps)), 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)


# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    
    start_time = time.time()
    
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.undistort(gray, cam_matrix, dist_coef)
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    gray = aruco.drawDetectedMarkers(gray, corners, ids)
        
    # measure fps
    fps = 1/(time.time() - start_time)
    
    draw_fps(fps)

    # show the frame
    cv2.imshow("Frame", gray)

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #print(i)
    mean += fps
    if fps > fps_max : 
        fps_max = fps
    if fps < fps_min :
        fps_min = fps

    # break when frame limit is reached 
    if i == 250:
        break
        
    i += 1    
        
cv2.destroyAllWindows()

mean = mean/(i+1)

print("i={}, mean={:.2f}, min={:.2f}, max={:.2f}".format(i, mean, fps_min, fps_max))

