# Credit: Tiziano Fiorenzani https://github.com/tizianofiorenzani/how_do_drones_work

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils.video import FPS
import cv2.aruco as aruco
import numpy as np
import time, math, cv2, pickle, os


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
    
def drawCorners(corners, frame, color):
    i = 0
    for corner in corners:
        corner_tuple = tuple(corner.tolist())
        cv2.circle(frame, (int(corner_tuple[0]), int(corner_tuple[1])), 5, color, -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(i), (int(corner_tuple[0]), int(corner_tuple[1])), font, 1, color, 2,cv2.LINE_AA)
        i += 1

#--- DEFINE
x0, y0 = -213-9, 65


#--- DEFINE Tag
#ids_to_find  = [2, 4, 6, 17]
ids_to_find  = [2, 17, 42]
marker_size  = 14 #- [mm]

coord = np.zeros((len(ids_to_find), 7), dtype = np.int16)
old_coord = np.zeros((len(ids_to_find), 7), dtype = np.int16)

#--- DEFINE the camera distortion arrays
#camera_matrix = np.array([[309.65140551, 0, 299.7942552], [0, 309.63299386, 236.80161718], [ 0, 0, 1]])*0.5
#camera_distortion = np.array([-0.32061628, 0.13711123, 0.0058947, 0.00258218, -0.03117783])

camera_matrix = np.array([[471.14915486,   0.        , 637.52234699],
       [  0.        , 448.93343667, 784.36266076],
       [  0.        ,   0.        ,   1.        ]])

camera_distortion = np.array([[-0.16381624,  0.01563538, -0.06420826,  0.00137489, -0.00082846]])


#--- DEFINE dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters_create()

#--- initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 180
camera.iso = 100 # max ISO to force exposure time to minimum to get less motion blur
#camera.exposure_mode = "sports"
#camera.resolution = (1280, 720)
#camera.resolution = (640, 480) 
camera.resolution = (1280, 960)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=camera.resolution)

#--- initialise variables
#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

#--- start imutils fps counter
fps = FPS().start()

initial_corners = np.array([
		[611., 308.],
		[648., 309.],
		[648., 336.],
		[609., 335.]
	], dtype = np.float32)
	
initial_terrain_corners = np.array([
		[254., 200.],
		[1028., 209.],
		[1244., 746.],
		[35., 739.]
	], dtype = np.float32)
       
offset = -110

destination = np.array([
		[1280/2 - 39/2, 960/2 - 39/2 + offset],
		[1280/2 + 39/2, 960/2 - 39/2 + offset],
		[1280/2 + 39/2, 960/2 + 39/2 + offset],
		[1280/2 - 39/2, 960/2 + 39/2 + offset]
    ], dtype = np.float32)
    
destination_terrain = np.array([
		[175, 175],
		[1280-175, 175],
		[1280-175, 960-175],
		[175, 960-175]
    ], dtype = np.float32)


perspectiveTransform = cv2.getPerspectiveTransform(initial_terrain_corners, destination_terrain)

#--- LOOP - capture frames from the camera
for frame_pi in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame_pi.array
    frame = cv2.warpPerspective(frame, perspectiveTransform, (1280, 960))
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
                
                coord[i][1] = coord[i][1]
                coord[i][2] = coord[i][2]

                """
                R_ct = np.matrix(cv2.Rodrigues(rvec[id_pos][0]))
                R_tc = R_ct.T

                #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
                coord[i][4:]= rotationMatrixToEulerAngles(R_flip*R_tc)
                """
                
            except:
                coord[i][1:] = old_coord[i][1:]
            
            
            i = i+1
    else: 
        coord[:][1:] = old_coord[:][1:]
    
    old_coord = coord
    
    # show the frame
    
    
    drawCorners(initial_terrain_corners, frame, (0, 0, 254))
    drawCorners(destination_terrain, frame, (0, 254, 0))
    frame = cv2.undistort(frame, camera_matrix, camera_distortion)
    cv2.imshow("Frame", frame)
    
    os.system('clear')
    print(coord)
    
    coord_dump = pickle.dumps(coord)

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

