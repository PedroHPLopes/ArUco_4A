# Credit: Tiziano Fiorenzani https://github.com/tizianofiorenzani/how_do_drones_work

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import cv2, time


#--- initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 180
camera.iso = 1600 # max ISO to force exposure time to minimum to get less motion blur
camera.resolution = (1280, 960)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=camera.resolution)

i=0

time.sleep(2)

perspectiveTransform = np.array([[ 1.48149190e+00,  8.32169939e-01, -3.23371591e+02],
       [-3.07292188e-02,  2.43886163e+00, -2.60605765e+02],
       [-2.87068424e-05,  1.30392453e-03,  1.00000000e+00]])


#--- LOOP - capture frames from the camera
for frame_pi in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame_pi.array
    frame = cv2.warpPerspective(frame, perspectiveTransform, (1280, 960))
    
    # show the frame
    cv2.imshow("Frame", frame)
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_write_str = "calib_pic_" + str(i) + ".jpg"
    cv2.imwrite(frame_write_str, frame)    
    time.sleep(2)
    
    if i > 100: break
    
    i += 1
    
    
cv2.destroyAllWindows()

print("[INFO] saved " + str(i) + " frames")

