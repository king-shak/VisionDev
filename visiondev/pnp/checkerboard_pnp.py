import numpy as np
import cv2
import glob

# set this to zero if you want to want to go through each individual image
DELAY_PERIOD = 0

PATH_TO_IMAGES = 'data/calibration/xbox_live_vision_cam/*.jpg'

GRID_WIDTH = 6
GRID_HEIGHT = 9

with np.load('data/calibration/configs/camera_properties.npz') as X:
    mtx, dist = [X[i] for i in ('arr_0','arr_1')]

def dot(a, b, mode):
    return (a[0] * b[0]) + (a[1] * b[1])

def norm(vector):
    return vector / np.linalg.norm(vector)

def getAngleToTarget(rvecs):
    return np.arccos(dot(norm([rvecs[2][0], rvecs[2][2]]), [0, 1], False))

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((GRID_WIDTH*GRID_HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:GRID_HEIGHT,0:GRID_WIDTH].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

for fname in glob.glob(PATH_TO_IMAGES):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (GRID_HEIGHT,GRID_WIDTH),None)

    if ret == True:
        cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)

        retval, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)

        rotationVector = cv2.Rodrigues(rvecs)     
        angleToTarget = getAngleToTarget(rotationVector[0])
        print(np.degrees(angleToTarget))

        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners,imgpts)
        cv2.imshow(fname,img)
        cv2.waitKey(DELAY_PERIOD)

cv2.destroyAllWindows()