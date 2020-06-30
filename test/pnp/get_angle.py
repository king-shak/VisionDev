# This program gets its data from the second image shown when checkerboard_pnp.py is run
import numpy as np

rvecs = [
    [-0.97442951, -0.07951381, 0.21015397],
    [-0.06528138, -0.79476276, -0.60339912],
    [ 0.21500111, -0.60168905,  0.76924951]]

def dot(a, b, mode):
    return (a[0] * b[0]) + (a[1] * b[1])

def norm(vector):
    return vector / np.linalg.norm(vector)

def getAngleToTarget(rvecs):
    return np.arccos(dot(norm([rvecs[2][0], rvecs[2][2]]), [0, 1], False))

print(np.degrees(getAngle(rvecs)))