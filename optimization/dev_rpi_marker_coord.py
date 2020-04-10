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
    
def get_coord(tag_id, ret, ids):
    tag_id_pos = np.where(ids==tag_id)[0]

    try:
        tag_id_pos = int(tag_id_pos)
    except:
        return False, False
    
    rvec = ret[0][tag_id_pos]
    tvec = ret[1][tag_id_pos]

    return rvec, tvec
    

def calibrate(calib_frames, id_calib):
    print("[INFO] Started calibration")
    rvec_list = np.empty((1, 3), dtype=np.float64)
    tvec_list = np.empty((1, 3), dtype=np.float64)
    
    for i in range(calib_frames):
        frame = stream.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        corners, ids, rejected = aruco.detectMarkers(image=frame, dictionary=aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=camera_distortion)
        ret = aruco.estimatePoseSingleMarkers(corners, calib_size, camera_matrix, camera_distortion)
        rvec, tvec = get_coord(id_calib, ret, ids)
    
        if ((type(rvec) and type(tvec)) is np.ndarray):
            rvec_list = np.append(rvec_list, rvec, axis=0)
            tvec_list = np.append(tvec_list, tvec, axis=0)
            
            
    rvec_list = rvec_list[1:, :]
    tvec_list = tvec_list[1:, :]
    print(rvec_list)
                    
    M_calib = basis_change_mtx(np.average(rvec_list, axis=0), np.average(tvec_list, axis=0))

    return M_calib 

def basis_change_mtx(rvec, tvec):
    M = np.zeros((4, 4), dtype=np.float32)

    M[:3, :3] = cv2.Rodrigues(rvec)[0]
    M[:3, 3] = tvec
    M[3, 3] = 1

    return M

def pad_vect(tvec):
    ret = np.ones((4,1))
    ret[:3, :] = tvec.T
    return ret 

#--- DEFINE parameters
ids_to_find  = [2]
id_calib = 42
marker_size  = 14 #- [mm]
calib_size = 20 #- [mm]
calib_frames = 50 #- nb of frames used for calibration at the start

#--- DEFINE
coord = np.zeros((len(ids_to_find), 4), dtype = np.int16)
old_coord = np.zeros((len(ids_to_find), 4), dtype = np.int16)
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
M_calib = calibrate(50, id_calib)
inv_M_calib = np.linalg.inv(M_calib)

print("done calib...")
time.sleep(5)
#--- start imutils fps counter
fps = FPS().start()

coord[:, 0] = ids_to_find
old_coord[:, 0] = ids_to_find
#--- LOOP - Send coordinates to clients
while True:
    frame = stream.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #-- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image=frame, dictionary=aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
    
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners)
        
        i=0
        for id in ids_to_find:
            rvec, tvec = get_coord(id, ret, ids)

            if ((type(rvec) and type(tvec)) is np.ndarray):
                aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 50)
                
                M_p = np.linalg.inv(M_calib)
                tvec = pad_vect(tvec)

                coord[i, 1:] = np.dot(M_p, tvec).T[:, :3]
                old_coord[i, 1:] = coord[i, 1:] 

            else:
                coord[i, 1:] = old_coord[i, 1:]
            
            i += 1
    
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
