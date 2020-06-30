from networktables import NetworkTables
import numpy as np
import cv2

# this is required to see message from NetworkTables
import logging
logging.basicConfig(level=logging.DEBUG)

# IP address of the roboRio, which is hosting the NetworkTables server 
ip = '10.29.10.2'

# connect to the NetworkTable server and the camera stream
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable('limelight')

cap = cv2.VideoCapture('http://limelight.local:5800')

imgp = np.zeros((8, 2), dtype=int)

while True:
    ret, img = cap.read()

    tv = sd.getNumber('tv', 0)

    if (ret == True) and (tv == 1):
        tcornx = sd.getNumberArray('tcornx', [0, 0])
        tcorny = sd.getNumberArray('tcorny', [0, 0])
        
        if (len(tcornx) == 8) and (len(tcorny) == 8):
            for i in range (8):
                imgp[i][0] = int(tcornx[i])
                imgp[i][1] = int(tcorny[i])

            # blue, green, red, white
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (100, 100, 100)]
            for x in range (8):
                cv2.circle(img, (imgp[x][0], imgp[x][1]), 4, colors[x], -2)

            print(imgp)

            cv2.imshow('points',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# when everyting is done, release the capture
cap.release()
cv2.destroyAllWindows()