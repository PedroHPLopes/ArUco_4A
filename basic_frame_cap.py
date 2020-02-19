# Credit: Tiziano Fiorenzani https://github.com/tizianofiorenzani/how_do_drones_work

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils.video import FPS
import cv2.aruco as aruco
import numpy as np
import time, math, cv2, pickle, os

#--- DEFINE the camera distortion arrays
camera_matrix = np.array([[309.65140551, 0, 299.7942552], [0, 309.63299386, 236.80161718], [ 0, 0, 1]])
camera_distortion = np.array([-0.32061628, 0.13711123, 0.0058947, 0.00258218, -0.03117783])

#--- initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 180
camera.iso = 1600 # max ISO to force exposure time to minimum to get less motion blur
#camera.exposure_mode = "sports"
#camera.resolution = (1280, 720)
camera.resolution = (640, 480) 
#camera.resolution = (1640, 922)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=camera.resolution)

#--- start imutils fps counter
fps = FPS().start()

#--- LOOP - capture frames from the camera
for frame_pi in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame_pi.array
    
    # show the frame
    #frame = cv2.undistort(frame, camera_matrix, camera_distortion)
    cv2.imshow("Frame", frame)
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    fps.update()

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))        
cv2.destroyAllWindows()

