from time import sleep
from picamera import PiCamera
import numpy as np
import cv2 as cv

# Values
awbGains = (0.0, 0.0)
shutterSpeed = 2
ISO = 800

# Ranges for the values
iso_range(1, 1600)
shutter_speed_range = (2, 20)
awb_range(0.0, 8.0)

# Initialize the camera, define the resolution and framerate
cam = PiCamera(resolution=(320, 240), framerate=120)

# Set the shutter speed to 2 ms (this can be changed in between frames)
cam.shutter_speed = shutterSpeed
# Set the iso to 800. 400-800 is reccomended for low light situations (this also can be changed between frames)
cam.iso = ISO

# Wait for the automatic gain control to settle
sleep(2)
# Now turn off the exposure mode, and print out the analog and digital gains
cam.exposure_mode = 'off'
print(cam.analog_gain)
print(cam.digital_gain)

cam.awb_mode = 'off'
cam.awb_gains = awbGains

# GUI setup
def redBalanceScaleCallback(val):
    global awbGains
    awbGains[0] = val / 100

def blueBalanceScaleCallback(val):
    global awbGains
    awbGains[1] = val / 100

def shutterSpeedScaleCallback(val):
    global shutterSpeed
    shutterSpeed = val

def isoScaleCallback(val):
    global ISO
    ISO = val

cv.namedWindow('CameraSetup')
cv.createTrackbar('Red Balance', 'CameraSetup', 0, 800, redBalanceScaleCallback)
cv.createTrackbar('Blue Balance', 'CameraSetup', 0, 800, blueBalanceScaleCallback)
cv.createTrackbar('Shutter Speed', 'CameraSetup', 2, 20, shutterSpeedScaleCallback)
cv.createTrackbar('ISO', 'CameraSetup', 0, 1600, isoScaleCallback)

while True:
    frame = np.empty((240 * 320 * 3,), dtype=np.uint8)
    cam.capture(frame, 'bgr')
    frame = frame.reshape((240, 320, 3))

    cv.imshow('CameraSetup', frame)

    # Set the shutter speed, iso and AWB gains for the next frame
    cam.shutter_speed = shutterSpeed
    cam.iso = ISO
    cam.awb_gains = awbGains