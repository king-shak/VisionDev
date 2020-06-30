import cv2 as cv
import numpy as np
import time

img = cv.imread('data/targets/2019Hatch/test1.jpg')

cv.imshow('img', img)
cv.waitKey(0)

imgBytes = cv.imencode('.jpg', img)[1].tobytes()

start = time.time()
imgBytesArray = np.frombuffer(imgBytes, dtype=np.uint8)
imgFromBytes = cv.imdecode(imgBytesArray, cv.IMREAD_UNCHANGED)
end = time.time()
delta = (end - start) * 1000
print(delta)

cv.imshow('img from bytes', imgFromBytes)
cv.waitKey(0)