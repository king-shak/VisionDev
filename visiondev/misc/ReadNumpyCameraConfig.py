import numpy as np
import json

resolution = [640, 480]
framerate = 30
midpt = [resolution[0] / 2, resolution[1] / 2]
frameSize = resolution[0] * resolution[1]

with np.load('data/calibration/configs/camera_properties.npz') as npCamConfigFile:
    mtx, dist = [npCamConfigFile[i] for i in ('arr_0','arr_1')]

CameraConfig = open('data/calibration/configs/CameraConfig.cfg', 'w')

CameraConfig.write(json.dumps(resolution) + '\n')
CameraConfig.write(json.dumps(framerate) + '\n')
CameraConfig.write(json.dumps(mtx.tolist()) + '\n')
CameraConfig.write(json.dumps(dist.tolist()) + '\n')
CameraConfig.write(json.dumps(midpt) + '\n')
CameraConfig.write(json.dumps(frameSize))

CameraConfig.close()