# Credit: Tiziano Fiorenzani https://github.com/tizianofiorenzani/how_do_drones_work
#https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils.video import FPS
from threading import Thread
import cv2.aruco as aruco
import numpy as np
import time, math, cv2, pickle, os

class PiVideoStream:
	def __init__(self, resolution=(1280, 960), framerate=30, iso=400, rotation=180):
		# initialize the camera and stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
        self.camera.rotation = rotation
        self.camera.iso = iso
        #self.camera.exposure_mode = "sports"
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
            
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the frame most recently read
        return self.frame
    
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

#--- define functions
#-- Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
#-- Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

#--- DEFINE
x0, y0 = -222, -45
coord = np.zeros((len(ids_to_find), 7), dtype = np.int16)
old_coord = np.zeros((len(ids_to_find), 7), dtype = np.int16)
font = cv2.FONT_HERSHEY_PLAIN
#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] = -1.0
R_flip[2,2] = -1.0

#--- DEFINE Tag
ids_to_find  = [2, 17, 42]
marker_size  = 14 #- [mm]

#--- DEFINE the camera distortion arrays
camera_matrix = np.array([[613.80715183, 0, 671.24584852], [0, 614.33915691, 494.57901986], [0, 0, 1]])#*0.5
camera_distortion = np.array([[-0.30736199, 0.09435416, -0.00032245, -0.00106545, -0.01286428]])

#--- DEFINE dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters_create()

# created a *threaded *video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from `picamera` module...")
stream = PiVideoStream().start()
time.sleep(2.0)
#--- start imutils fps counter
fps = FPS().start()

#--- LOOP - capture frames from the camera
for frame_pi in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = stream.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #-- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image=frame, dictionary=aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    
    if ids is not None:
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
        rvec, tvec = ret[0], ret[1]
        
        aruco.drawDetectedMarkers(frame, corners)
        
        i = 0
        for id in ids_to_find:
            id_pos = np.where(ids==id)
            coord[i][0] = id
            
            try:
                id_pos = int(id_pos[0])
                aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec[id_pos][0], tvec[id_pos][0], 50)
                coord[i][1] = (tvec[id_pos][0][0] - x0)
                coord[i][2] = (-tvec[id_pos][0][1] - y0)
                coord[i][3] = tvec[id_pos][0][2]
                
            except:
                coord[i][1:] = old_coord[i][1:]
            
            i = i+1
    else: 
        coord[:][1:] = old_coord[:][1:]
    
    old_coord = coord
    
    # show the frame
    #frame = cv2.undistort(frame, camera_matrix, camera_distortion)
    cv2.imshow("Frame", frame)
    
    os.system('clear')
    print(coord)
    
    coord_dump = pickle.dumps(coord)

    fps.update()

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))        
cv2.destroyAllWindows()
stream.stop()