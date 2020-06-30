import cv2 as cv
import numpy as np
from time import sleep

low = 0
high = 255
high_H = 180

low_H = low
low_S = low
low_V = low
high_H = high_H
high_S = high
high_V = high

window_capture_name = 'Video Capture'
median_blur_button_name = 'Median Blur'
gaussian_blur_button_name = 'Gaussian Blur'
smooth_button_name = 'Image Smoothing'

window_threshold_name = 'Threshold View'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

window_contour_name = 'Countour View'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_threshold_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_threshold_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_threshold_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_threshold_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_threshold_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_threshold_name, high_V)

def setupCaptureWindow():
    cv.namedWindow(window_capture_name)

def setupHSVThresholdingWindow():
    cv.namedWindow(window_threshold_name)
    cv.createTrackbar(low_H_name, window_threshold_name , low_H, high_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_threshold_name , high_H, high_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_threshold_name , low_S, high, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_threshold_name , high_S, high, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_threshold_name , low_V, high, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_threshold_name , high_V, high, on_high_V_thresh_trackbar)

def setupContourWindow():
    cv.namedWindow(window_contour_name)

setupCaptureWindow()
setupHSVThresholdingWindow()
setupContourWindow()

frame = cv.imread('test/test_images/test.jpg')

while True:
    # Perform HSV thresholding
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))

    # Find contours within the image
    frame_contours = frame_threshold
    frame_temp = np.array(frame, copy=True)

    contours, hierarchy = cv.findContours(frame_contours, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    frame_contours = cv.drawContours(frame_temp, contours, -1, (0, 0, 255), 5)

    print(contours)

    # Label the number of contours found
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame_contours, str(len(contours)), (10,500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

    # Update the windows with the updated frame
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_threshold_name, frame_threshold)
    cv.imshow(window_contour_name, frame_contours)
    sleep(0.01)
    key = cv.waitKey(30)
    if key == ord("q"):
        break