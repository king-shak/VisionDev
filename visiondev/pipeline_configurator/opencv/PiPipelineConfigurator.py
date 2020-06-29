import cv2 as cv
import numpy as np
import picamera

cam = PiCamera()

max_exposure = 800
max_balance = 8

resolution = (1280, 720)
fps = 60
red_balance = cam.awb_gains[0]
blue_balance = cam.awb_gains[1]
exposure = cam.iso

cam.resolution = resolution
cam.framerate = fps
cam.awb_gains = (red_balance, blue_balance)
cam.iso = exposure
rawCapture = PiRGBArray(cam, size=resolution)

high = 255
high_H = 180

low_H = 0
low_S = 0
low_V = 0
high_H = high_H
high_S = high
high_V = high

window_capture_name = 'Video Capture'
red_balance_name = 'Red Balance'
blue_balance_name = 'Blue Balance'
exposure_name = 'Exposure'

window_threshold_name = 'Threshold View'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

window_contour_name = 'Countour View'

def on_low_H_tresh_trackbar(val):
        global low_H
        global high_H
        low_H = min(high_H - 1, val)
        cv.setTrackbarPos(low_H_name, window_threshold_name, low_H)

def on_high_H_tresh_trackbar(val):
        global low_H
        global high_H
        low_H = max(val, low_H + 1)
        cv.setTrackbarPos(high_H_name, window_threshold_name, high_H)

def on_low_S_tresh_trackbar(val):
        global low_S
        global high_S
        low_S = min(high_S - 1, val)
        cv.setTrackbarPos(low_S_name, window_threshold_name, low_S)

def on_high_S_tresh_trackbar(val):
        global low_S
        global high_S
        low_S = max(val, low_S + 1)
        cv.setTrackbarPos(high_S_name, window_threshold_name, high_S)

def on_low_V_tresh_trackbar(val):
        global low_V
        global high_V
        low_V = min(high_V - 1, val)
        cv.setTrackbarPos(low_V_name, window_threshold_name, low_V)

def on_high_V_tresh_trackbar(val):
        global low_V
        global high_V
        low_V = max(val, low_V + 1)
        cv.setTrackbarPos(high_V_name, window_threshold_name, high_V)

def on_exposure_trackbar(val):
        global exposure
        exposure = val
        cv.setTrackbarPos(exposure_name, window_capture_name, exposure)

def on_red_balance_trackbar(val):
        global red_balance
        red_balance = val
        cv.setTrackbarPos(red_balance_name, window_capture_name, red_balance)

def on_blue_balance_trackbar(val):
        global blue_balance
        blue_balance = val
        cv.setTrackbarPos(blue_balance_name, window_capture_name, blue_balance)

cv.namedWindow(window_capture_name)
cv.createTrackbar(exposure_name, window_capture_name, exposure, max_exposure, on_exposure_trackbar)
cv.createTrackbar(red_balance_name, window_capture_name, red_balance, max_balance, on_red_balance_trackbar)
cv.createTrackbar(blue_balance_name, window_capture_name, blue_balance, max_balance, on_blue_balance_trackbar)

cv.namedWindow(window_threshold_name)
cv.createTrackbar(low_H_name, window_threshold_name, low_H, high_H, on_low_H_tresh_trackbar)
cv.createTrackbar(high_H_name, window_threshold_name, high_H, high_H, on_high_H_tresh_trackbar)
cv.createTrackbar(low_S_name, window_threshold_name, low_S, high, on_low_S_tresh_trackbar)
cv.createTrackbar(high_S_name, window_threshold_name, high_S, high, on_high_S_tresh_trackbar)
cv.createTrackbar(low_V_name, window_threshold_name, low_V, high, on_low_V_tresh_trackbar)
cv.createTrackbar(high_V_name, window_threshold_name, high_V, high, on_high_V_tresh_trackbar)

cv.namedWindow(window_contour_name)


for img in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = img.array
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    frame_contours = frame_threshold
    frame_temp = np.array(frame, copy=True)

    contours, hierarchy = cv.findContours(frame_contours, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    frame_contours = cv.drawContours(frame_temp, contours, -1, (0, 0, 255), 5)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame_contours, str(len(contours)), (10,500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow(window_capture_name, frame)
    cv.imshow(window_threshold_name, frame_threshold)
    cv.imshow(window_contour_name, frame_contours)
    
    rawCapture.truncate(0)

    cam.awb_gains = (red_balance, blue_balance)
    cam.iso = exposure

    key = cv.waitKey(30)
    if key == ord("q"):
        break