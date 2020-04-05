# Credit: Tiziano Fiorenzani https://github.com/tizianofiorenzani/how_do_drones_work
# https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils.video import FPS
from threading import Thread
import cv2.aruco as aruco
import numpy as np
import time, math, cv2, pickle, os

class PiVideoStream:
    def __init__(self, resolution=(1280, 960), framerate=30, iso=400, rotation=0):
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

    sy = math.sqrt(R[0, 0]* R[0, 0] + R[1, 0]* R[1, 0])

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

def calibrate(calib_frames):
    i = 0
    rvec_mtx = np.empty((1, 1, 3), dtype=np.float32)
    tvec_mtx = np.empty((1, 1, 3), dtype=np.float32)
    while (i<calib_frames):
        frame = stream.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejected = aruco.detectMarkers(image=frame, dictionary=aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=camera_distortion)
        
        if ids is not None:
            ret = aruco.estimatePoseSingleMarkers(corners, calib_size, camera_matrix, camera_distortion)
            rvec, tvec = ret[0], ret[1]
            
            aruco.drawDetectedMarkers(frame, corners)
            
            try:
                calib_pos = np.where(ids==id0)
                calib_pos = int(calib_pos[0])

                aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec[calib_pos][0], tvec[calib_pos][0], 50)
                rvec_mtx = np.append(rvec_mtx, rvec[calib_pos], axis=0)
                tvec_mtx = np.append(tvec_mtx, rvec[calib_pos], axis=0)
                
                i += 1
            except:
                continue 
        
        # show the frame
        cv2.imshow("Frame", frame)

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rvec_avg = np.average(rvec_mtx, axis=0)
    tvec_avg = np.average(tvec_mtx, axis=0)

    M_calib = basis_change_mtx(rvec, tvec)

    return M_calib #- return mean of x0 and y0 

def basis_change_mtx(rvec, tvec):
    M = np.zeros((4, 4), dtype=np.float32)

    M[:3, :3] = cv2.Rodrigues(rvec)[0]
    M[:3, 3] = tvec
    M[3, 3] = 1

    return M

def get_coord(M_calib, frame, old_coord, corners, market_size, camera_matrix, camera_distortion):
    coord = np.zeros((len(ids_to_find), 7), dtype = np.int16)

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
                aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec[id_pos], tvec[id_pos], 50)
                
                M_i = basis_change_mtx(rvec[id_pos], tvec[id_pos])
                inv_M_calib = np.linalg.inv(M_calib)
                M_t = np.dot(M_i, inv_M_calib)
                new_tvec = np.ones((4, 1))
                new_tvec[:3, 0]
                posvec = np.dot(M_t, new_tvec)
                
                coord[i, 1:4] = posvec
                
            except:
                coord[i][1:] = old_coord[i][1:]
            
            i = i+1
    else: 
        coord[:][1:] = old_coord[:][1:]

    return coord

#--- DEFINE parameters
ids_to_find  = [2, 17]
id_calib = 42
marker_size  = 14 #- [mm]
calib_size = 20 #- [mm]
calib_frames = 50 #- nb of frames used for calibration at the start

#--- DEFINE
coord = np.zeros((len(ids_to_find), 7), dtype = np.int16)
old_coord = np.zeros((len(ids_to_find), 7), dtype = np.int16)
font = cv2.FONT_HERSHEY_PLAIN
#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] = -1.0
R_flip[2,2] = -1.0

#--- DEFINE the camera distortion arrays
camera_matrix = np.array([[613.80715183, 0, 671.24584852], [0, 614.33915691, 494.57901986], [0, 0, 1]])#*0.5
camera_distortion = np.array([[-0.30736199, 0.09435416, -0.00032245, -0.00106545, -0.01286428]])

#--- DEFINE dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters_create()

# created a*threaded*video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from `picamera` module...")
stream = PiVideoStream().start()
time.sleep(2.0)

#--- CALIBRATION - Find the central tag and set x0 and y0
M_calib = calibrate(50)

#--- start imutils fps counter
fps = FPS().start()

#--- LOOP - Send coordinates to clients
while True:
    frame = stream.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #-- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image=frame, dictionary=aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    coord = get_coord(M_calib, frame, old_coord, corners, marker_size, camera_matrix, camera_distortion)
    old_coord = coord
    
    # show the frame
    # frame = cv2.undistort(frame, camera_matrix, camera_distortion)
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
