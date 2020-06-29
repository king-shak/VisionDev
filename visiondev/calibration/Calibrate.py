import numpy as np
import cv2
import glob
import json

# set this to zero if you want to want to go through each individual image
DELAY_PERIOD = 0

PATH_TO_IMAGES = 'src/Calibration/images/*.jpg'

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.0001)

GRID_WIDTH = 6
GRID_HEIGHT = 9

# prepare object points, initialze object and image point arrays
objp = np.zeros((GRID_WIDTH*GRID_HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:GRID_HEIGHT,0:GRID_WIDTH].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(PATH_TO_IMAGES)

for fname in images:
    # grab the image, convert it to grayscale
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find the chess board corners (image points)
    ret, corners = cv2.findChessboardCorners(gray, (GRID_HEIGHT,GRID_WIDTH),None)

    if ret == True:
        # add the object point
        objpoints.append(objp)

        # refine the image points
        tempCorners = corners.copy()
        cv2.cornerSubPix(gray,corners,(9,9),(-1,-1),criteria)
        imgpoints.append(corners)

        # draw and display the corners
        cv2.drawChessboardCorners(img, (GRID_HEIGHT,GRID_WIDTH), corners,ret)
        cv2.imshow(fname,img)
        cv2.waitKey(DELAY_PERIOD)

# now that we have our image and object points we can calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print('Camera Matrix:\n',mtx)
print('Distortion Coefficients:\n',dist)

# now go through each image and undistort them using the values we just generated
# for fname in images:
#     img = cv2.imread(fname)
#     h,  w = img.shape[:2]
#     newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#     # undistort
#     dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

#     # crop the image
#     x,y,w,h = roi
#     dst = dst[y:y+h, x:x+w]
#     cv2.imshow(fname,dst)
    # cv2.waitKey(DELAY_PERIOD)

# calculate and report our re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("Total error: ", mean_error/len(objpoints))

# save our camera matrix and distortion coefficients
np.savez("src/camera_properties", mtx, dist)
cameraConfigFile = open('PiDefaultLens.json', 'w+')
cameraConfig = {
    'resolution': [320, 240],
    'fps': 90,
    'distortionCoefficients': dist.tolist(),
    'cameraMatrix': mtx.tolist(),
    'sensorSize': [3.68, 2.76]
}
json.dump(cameraConfig, cameraConfigFile)
cv2.destroyAllWindows()