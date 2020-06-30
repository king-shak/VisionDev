from time import sleep
from networktables import NetworkTables
import logging
import numpy as np
import cv2

cap = cv2.VideoCapture('http://10.29.10.11:5800/')

logging.basicConfig(level=logging.DEBUG)
ip = '10.29.10.2'
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable('limelight')

# while True:
#     print(sd.getNumber('tv', 0.0))

corners = np.zeros((8, 2), dtype=np.float32)


tv = sd.getNumber('tv', 0)
while (True):
    tv = sd.getNumber('tv', 0)
    if (tv == 1):
        ret, img = cap.read()
        cv2.imwrite('test\test.jpg', img)
        print("saved image")
        tcornx = sd.getNumberArray('tcornx', [0, 0])
        tcorny = sd.getNumberArray('tcorny', [0, 0])
        print('got corners')

        if (len(tcornx) == 8) and (len(tcorny) == 8):
            for i in range (8):
                corners[i][0] = tcornx[i]
                corners[i][1] = tcorny[i]

            print(corners)
            print("\n")
            sleep(.01)

# when everyting is done, release the capture
cap.release()