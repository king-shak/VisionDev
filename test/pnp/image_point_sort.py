import numpy as np
import cv2

corners = np.array([
    [211, 132],
    [100, 141],
    [108, 100],
    [198, 94]
], dtype=np.float32)

def norm(vector):
    return vector / np.linalg.norm(vector)

def getPrincipalAxes(imgp):
    mean = np.empty((0))
    mean, eigen = cv2.PCACompute(imgp, mean)
    x = [eigen[0][0], eigen[1][0]]
    y = [eigen[0][1], eigen[1][1]]
    return x, y, np.ravel(mean)

def sortImgPts(imgpts, x, y, midpt):
    sortedPts = np.zeros((4, 2))
    pts = np.zeros((4, 2))
    
    for x in range(len(imgpts)):
        pts[x] = norm([imgpts[x][0] - midpt[0], midpt[1] - imgpts[x][1]])
        if (pts[x][0] < 0) and (pts[x][1] < 0):
            sortedPts[0] = imgpts[x]
        elif (pts[x][0] > 0) and (pts[x][1] < 0):
            sortedPts[1] = imgpts[x]
        elif (pts[x][0] < 0) and (pts[x][1] > 0):
            sortedPts[2] = imgpts[x]
        elif (pts[x][0] > 0) and (pts[x][1] > 0):
            sortedPts[3] = imgpts[x]
        else:
            print('Angle of Rotation is to extreme')

    return sortedPts

x, y, midPoint = getPrincipalAxes(corners)
points = sortImgPts(corners, x, y, midPoint)
print(points)