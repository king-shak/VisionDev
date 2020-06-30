from networktables import NetworkTables
import numpy as np
import cv2
import glob
import logging
import collections

with np.load('data/calibration/configs/camera_properties.npz') as X:
    mtx, dist = [X[i] for i in ('arr_0','arr_1')]

logging.basicConfig(level=logging.DEBUG)
ip = '10.29.10.2'
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable('limelight')

cap = cv2.VideoCapture('http://10.29.10.11:5800/')

HEIGHT_OF_CAMERA = 6.25
HEIGHT_OF_TARGET = 31.5
DELTA_HEIGHT = HEIGHT_OF_TARGET - HEIGHT_OF_CAMERA
MOUNTING_ANGLE = 121


horizontal = [1., 0.]

objp = np.array([
    [5.9363, 2.9128, 0],
    [4, 2.4120, 0],
    [-4, 2.4120, 0],
    [-5.9363, 2.9128, 0],
    [-7.3134, -2.4120, 0],
    [-5.3771, -2.9128, 0],
    [5.3771, -2.9128, 0],
    [7.3134, -2.4120, 0]
], dtype=np.float32)

NUM_OF_POINTS = len(objp)

corners = np.zeros((NUM_OF_POINTS, 2), dtype=np.float32)

axis = np.float32([[0,0,0], [5,0,0], [0,5,0], [0,0,5]]).reshape(-1,3)

def getPrincipalAxes(imgp):
    mean = np.empty((0))
    mean, eigen = cv2.PCACompute(imgp, mean)
    x = [eigen[0][0], eigen[1][0]]
    y = [eigen[0][1], eigen[1][1]]
    rotation = getAngle(horizontal, x, False)
    return x, y, np.ravel(mean), rotation

def rotatePoint(pt, angle):
    x = (pt[0] * np.cos(angle)) - (pt[1] * np.sin(angle))
    y = (pt[1] * np.cos(angle)) + (pt[0] * np.sin(angle))
    return [x, y]

def dot(a, b):     
    return (a[0] * b[0]) + (a[1] * b[1])

def norm(vector):
    return vector / np.linalg.norm(vector)

def getAngleToTarget(rvecs):
    angle = np.pi - np.arccos(dot(norm([rvecs[2][0], rvecs[2][2]]), [0, 1]))
    crossProduct = np.cross([0, 0, 1], rvecs[2])
    if (crossProduct[1] < 0):
        angle*=-1
    return angle

def getLength(vector):
    return np.sqrt((np.power(vector[0], 2) + np.power(vector[1], 2)))

def getAngle(a, b, mode = None):
    rotation = np.arccos(dot(a, b) / (getLength(a) * getLength(b)))
    if mode is not None:
        if (b[1] < 0):
            if mode:
                rotation = (2 * np.pi) - rotation
            else :
                rotation*=-1
    return rotation

def sortImgPts(imgpts, x, y, midpt, rotation):
    pts = {}
    for i in range(NUM_OF_POINTS):
        pt = norm(rotatePoint([imgpts[i][0] - midpt[0], midpt[1] - imgpts[i][1]], -1 * rotation))
        angle = getAngle(horizontal, pt, True)
        pts[angle] = imgpts[i]
    
    pts = collections.OrderedDict(sorted(pts.items()))
    
    test = np.zeros((NUM_OF_POINTS, 2), dtype=np.float32)
    j = 0
    for i in pts:
        test[j] = pts[i]
        j+=1

    return test

def draw(img, imgpts):
    origin = tuple(imgpts[0].ravel())
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 3)
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 3)
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (0,255,255), 3)
    return img

while (True):
    ret, img = cap.read()
    tv = sd.getNumber('tv', 0)

    if (ret == True) and (tv == 1):
        tcornx = sd.getNumberArray('tcornx', [0, 0])
        tcorny = sd.getNumberArray('tcorny', [0, 0])

        if (len(tcornx) == NUM_OF_POINTS) and (len(tcorny) == NUM_OF_POINTS):
            for x in range(NUM_OF_POINTS):
                corners[x][0] = tcornx[x]
                corners[x][1] = tcorny[x]

            x, y, midPoint, rotation = getPrincipalAxes(corners)
            corners = sortImgPts(corners, x, y, midPoint, rotation)
            
            retval, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
            rotMat, jacobian = cv2.Rodrigues(rvecs)     
            angleToTarget = getAngleToTarget(rotMat)
            print(np.degrees(angleToTarget), tvecs)

            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            img = draw(img,imgpts)
            # blue, green, red, white, light blue, pink, yellow, gray
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (100, 100, 100)]
            for x in range (8):
                cv2.circle(img, (corners[x][0], corners[x][1]), 4, colors[x], -2)
            cv2.imshow('stream',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()