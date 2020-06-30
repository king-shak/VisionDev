import numpy as np
import cv2 as cv
import glob
import json

# set this to zero if you want to want to go through each individual image
DELAY_PERIOD = 0

PATH_TO_IMAGES = 'data/calibration/xbox_live_vision_cam/*.jpg'

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 60, 0.0001)

GRID_WIDTH = 6
GRID_HEIGHT = 9

RESOLUTION = (640, 480)
FPS = 90
SENSOR_SIZE = (3.68, 2.76)

# prepare object points, initialze object and image point arrays
objp = np.zeros((GRID_WIDTH*GRID_HEIGHT, 3), np.float32)
objp[:,:2] = np.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH].T.reshape(-1, 2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(PATH_TO_IMAGES)

for fname in images:
    # grab the image, convert it to grayscale
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # find the chess board corners (image points)
    ret, corners = cv.findChessboardCorners(gray, (GRID_HEIGHT,GRID_WIDTH),None)

    if ret == True:
        # add the object point
        objpoints.append(objp)

        # refine the image points
        tempCorners = corners.copy()
        cv.cornerSubPix(gray,corners,(9,9),(-1,-1),criteria)
        imgpoints.append(corners)

        # draw and display the corners
        cv.drawChessboardCorners(img, (GRID_HEIGHT,GRID_WIDTH), corners,ret)
        cv.imshow(fname,img)
        cv.waitKey(DELAY_PERIOD)

# now that we have our image and object points we can calibrate
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, RESOLUTION, None, None)

print('Camera Matrix:\n',mtx)
print('Distortion Coefficients:\n',dist)

# now go through each image and undistort them using the values we just generated
for fname in images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newCameraMtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow(fname,dst)
    cv.waitKey(DELAY_PERIOD)

# calculate and report our re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("Total error: ", mean_error/len(objpoints))

# save our camera matrix and distortion coefficients
np.savez("data/configs/camera_properties", mtx, dist)
cameraConfigFile = open('data/configs/xbox_live_vision_cam_config.json', 'w+')

cameraConfig = {
    'resolution': list(RESOLUTION),
    'fps': FPS,
    'distortionCoefficients': dist.tolist(),
    'cameraMatrix': mtx.tolist(),
    'sensorSize': list(SENSOR_SIZE)
}

json.dump(cameraConfig, cameraConfigFile)
cv.destroyAllWindows()