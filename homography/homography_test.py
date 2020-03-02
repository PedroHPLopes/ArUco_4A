import cv2
import numpy as np
import cv2.aruco as aruco

def drawCorners(corners, frame):
    i = 0
    for corner in corners:
        corner_tuple = tuple(corner.tolist())
        cv2.circle(frame, (int(corner_tuple[0]), int(corner_tuple[1])), 5, (0, 255, 0), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(i), (int(corner_tuple[0]), int(corner_tuple[1])), font, 1, (0,255,0), 2,cv2.LINE_AA)
        i += 1

frame = cv2.imread("img3.jpg")
original = frame
marker_size = 10

camera_matrix = np.array([[613.80715183, 0, 671.24584852], [0, 614.33915691, 494.57901986], [0, 0, 1]])#*0.5
camera_distortion = np.array([[-0.30736199, 0.09435416, -0.00032245, -0.00106545, -0.01286428]])

#--- DEFINE dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters_create()

corners, ids, rejected = aruco.detectMarkers(image=frame, dictionary=aruco_dict, parameters=parameters)#, cameraMatrix=camera_matrix, distCoeff=camera_distortion)
corners_calib_tag = corners[1][0]

print("[INFO] corners:\n", corners_calib_tag)
aruco.drawDetectedMarkers(frame, corners)

maxWidth = int(corners_calib_tag[2][0] - corners_calib_tag[3][0])
print("[INFO] maxWidth: ", maxWidth)

corr = 0
"""
destination = np.array([
		[corners_calib_tag[3][0], corners_calib_tag[3][1]-maxWidth+corr],
		[corners_calib_tag[3][0] + maxWidth, corners_calib_tag[3][1]-maxWidth+corr],
		[corners_calib_tag[3][0] + maxWidth, corners_calib_tag[3][1]+corr],
		[corners_calib_tag[3][0], corners_calib_tag[3][1]+corr]
    ], dtype = "float32")
"""

offset = -110

destination = np.array([
		[1280/2 - 39/2, 960/2 - 39/2 + offset],
		[1280/2 + 39/2, 960/2 - 39/2 + offset],
		[1280/2 + 39/2, 960/2 + 39/2 + offset],
		[1280/2 - 39/2, 960/2 + 39/2 + offset]
    ], dtype = "float32")


perspectiveTransform = cv2.getPerspectiveTransform(corners_calib_tag, destination)


frame = cv2.warpPerspective(frame, perspectiveTransform, (1280, 960))

#ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
#tvec = ret[1]

drawCorners(corners_calib_tag, original)

drawCorners(destination, original)


cv2.circle(frame, (615, 292), 5, (0, 255, 0), -1)
#frame = cv2.undistort(frame, np.dot(perspectiveTransform, camera_matrix), camera_distortion)
cv2.circle(frame, (int(1280/2), int(960/2)), 5, (0, 255, 0), -1)
cv2.imshow("image", frame)
cv2.imshow("original", original)
cv2.waitKey(0)
