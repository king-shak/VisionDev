# Yeah so this is basically just used to get thresholding values for static images...may want to clean it up for that
# Need to update the tkinter version so you can switch it to work with a static image

import cv2 as cv
import numpy as np

# with np.load('src/camera_properties.npz') as X:
#     mtx, dist = [X[i] for i in ('arr_0','arr_1')]

# objp = np.array([
#     [5.9363, 2.9128, 0],
#     [4, 2.4120, 0],
#     [-4, 2.4120, 0],
#     [-5.9363, 2.9128, 0],
#     [-7.3134, -2.4120, 0],
#     [-5.3771, -2.9128, 0],
#     [5.3771, -2.9128, 0],
#     [7.3134, -2.4120, 0]
# ], dtype=np.float32)

# NUM_OF_POINTS = len(objp)

# corners = np.zeros((NUM_OF_POINTS, 2), dtype=np.float32)

# axis = np.float32([[0,0,0], [5,0,0], [0,5,0], [0,0,5]]).reshape(-1,3)

# def draw(img, imgpts):
#     origin = tuple(imgpts[0].ravel())
#     img = cv.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 3)
#     img = cv.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 3)
#     img = cv.line(img, origin, tuple(imgpts[3].ravel()), (0,255,255), 3)
#     return img

cap = cv.VideoCapture(0)
frame_area = cap.get(3) * cap.get(4)

low = 0
high = 255
high_H = 180
high_noise = 40
high_aperture = 20
high_gaussian = 20
high_sigma = 20
high_contour_filtering = 100
contour_grouping_high = 4
# 0 for center, 1 for right, 2 for left, 3 for top, 4 for bottom
contour_sorting_mode_high = 4
# 0 for up, 1 for down
intersection_high = 1
aspect_ratio_high_val = 5

aperture_size = 5

gaussian_kernel_size = 5
sigma_x = 0
sigma_y = 0

low_H = low
low_S = low
low_V = low
high_H = high_H
high_S = high
high_V = high

low_target_area = low
low_target_fullness = low
low_aspect_ratio = low
high_target_area = high_contour_filtering
high_target_fullness = high_contour_filtering
high_aspect_ratio = aspect_ratio_high_val

num_of_contours_to_group = low
intersection = low
contour_sorting_mode = low

window_capture_name = 'Video Capture'

window_median_blur_name = 'Median Blur'
aperture_name = 'Linear Aperture Size'

window_gaussian_blur_name = 'Gaussian Blur'
gaussian_kernel_size_name = 'Gaussian Kernel Size'
sigma_x_name = 'Sigma X'
sigma_y_name = 'Sigma Y'

window_threshold_name = 'Threshold View'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

window_contour_name = 'Countour View'
low_target_area_name = 'Low Target Area Ratio'
low_target_fullness_name = 'Low Target Fullness Ratio'
low_aspect_ratio_name = 'Low Aspect Ratio'
high_target_area_name = 'High Target Area Ratio'
high_target_fullness_name = 'High Target Fullness Ratio'
high_aspect_ratio_name = 'High Aspect Ratio'
num_of_contours_to_group_name = 'Number of Contours to Group'
intersection_name = 'Intersection (0-1, up-down)'
contour_sorting_mode_name = 'Contour Sorting Mode (0-1-2-3-4, center-right-left-top-bottom)'

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

def on_aperture_size_trackbar(val):
    global aperture_size
    if (val % 2 == 0):
        aperture_size = val + 1
    elif (val < 0):
        aperture_size = 1
    cv.setTrackbarPos(aperture_name, window_median_blur_name, aperture_size)

def on_gaussian_kernel_size_trackbar(val):
    global gaussian_kernel_size
    if (val % 2 == 0):
        gaussian_kernel_size = val + 1
    elif (val < 0):
        gaussian_kernel_size = 1
    cv.setTrackbarPos(gaussian_kernel_size_name, window_gaussian_blur_name, gaussian_kernel_size)

def on_sigma_x_trackbar(val):
    global sigma_x
    sigma_x = val
    cv.setTrackbarPos(sigma_x_name, window_gaussian_blur_name, sigma_x)

def on_sigma_y_trackbar(val):
    global sigma_y
    sigma_y = val
    cv.setTrackbarPos(sigma_y_name, window_gaussian_blur_name, sigma_y)

def on_low_target_area_trackbar(val):
    global low_target_area
    low_target_area = val / 100
    cv.setTrackbarPos(low_target_area, window_contour_name, low_target_area)

def on_low_target_fullness_trackbar(val):
    global low_target_fullness
    low_target_fullness = val / 100
    cv.setTrackbarPos(low_target_fullness_name, window_contour_name, low_target_fullness)

def on_low_aspect_ratio_trackbar(val):
    global low_aspect_ratio
    low_aspect_ratio = val
    cv.setTrackbarPos(low_aspect_ratio_name, window_contour_name, low_aspect_ratio)

def on_high_target_area_trackbar(val):
    global high_target_area
    high_target_area = val / 100
    cv.setTrackbarPos(high_target_area_name, window_contour_name, high_target_area)

def on_high_target_fullness_trackbar(val):
    global high_target_fullness
    high_target_fullness = val / 100
    cv.setTrackbarPos(high_target_fullness, window_contour_name, high_target_fullness)

def on_high_aspect_ratio_trackbar(val):
    global high_aspect_ratio
    high_aspect_ratio = val
    cv.setTrackbarPos(high_aspect_ratio_name, window_contour_name, high_aspect_ratio)

def on_num_of_contours_to_group_trackbar(val):
    global num_of_contours_to_group
    num_of_contours_to_group = val
    cv.setTrackbarPos(num_of_contours_to_group_name, window_contour_name, num_of_contours_to_group)

def on_intersection_trackbar(val):
    global intersection
    intersection = val
    cv.setTrackbarPos(intersection_name, window_contour_name, intersection)

def get_intersection(intersection_val):
    if (intersection_val == 0):
        return 'up'
    elif (intersection_val == 1):
        return 'down'
    else:
        raise Exception('Invalid intersection value (the intersection value was not 0 or 1')

def on_contour_sorting_mode_trackbar(val):
    global contour_sorting_mode
    contour_sorting_mode = val
    cv.setTrackbarPos(contour_sorting_mode_name, window_contour_name, contour_sorting_mode)

def get_contour_sorting_mode(contour_sorting_mode_val):
    if (contour_sorting_mode == 0):
        return 'center'
    elif (contour_sorting_mode == 1):
        return 'right'
    elif (contour_sorting_mode == 2):
        return 'left'
    elif (contour_sorting_mode == 3):
        return 'top'
    elif (contour_sorting_mode == 4):
        return 'bottom'
    else:
        raise Exception('Invalid contour sorting mode value (the contour sorting mode value was not 0, 1, 2, 3 or 4')

def processContours(contours):
    processed_contours = []
    for contour in contours:
        cnt = Contour(contour)
        processed_contours.append(cnt)
    return processed_contours

# def filterContours(contours):
#     filtered_contours = []
#     for contour in contours:
#         if (contour.getBoundingRectangleArea / frame_area)

def setupCaptureWindow():
    cv.namedWindow(window_capture_name)

def setupMedianBlurWindow():
    cv.namedWindow(window_median_blur_name)
    cv.createTrackbar(aperture_name, window_median_blur_name, aperture_size, high_aperture, on_aperture_size_trackbar)

def setupGaussianBlurWindow():
    cv.namedWindow(window_gaussian_blur_name)
    cv.createTrackbar(gaussian_kernel_size_name, window_gaussian_blur_name, gaussian_kernel_size, high_gaussian, on_gaussian_kernel_size_trackbar)
    cv.createTrackbar(sigma_x_name, window_gaussian_blur_name, sigma_x, high_sigma, on_sigma_x_trackbar)
    cv.createTrackbar(sigma_y_name, window_gaussian_blur_name, sigma_y, high_sigma, on_sigma_y_trackbar)

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
    cv.createTrackbar(low_target_area_name, window_contour_name, low_target_area, high_contour_filtering, on_low_target_area_trackbar)
    cv.createTrackbar(high_target_area_name, window_contour_name, high_target_area, high_contour_filtering, on_high_target_area_trackbar)
    cv.createTrackbar(low_target_fullness_name, window_contour_name, low_target_fullness, high_contour_filtering, on_low_target_fullness_trackbar)
    cv.createTrackbar(high_target_fullness_name, window_contour_name, high_target_fullness, high_contour_filtering, on_high_target_fullness_trackbar)
    cv.createTrackbar(low_aspect_ratio_name, window_contour_name, low_aspect_ratio, aspect_ratio_high_val, on_low_aspect_ratio_trackbar)
    cv.createTrackbar(high_aspect_ratio_name, window_contour_name, high_aspect_ratio, aspect_ratio_high_val, on_high_aspect_ratio_trackbar)
    cv.createTrackbar(num_of_contours_to_group_name, window_contour_name, num_of_contours_to_group, contour_grouping_high, on_num_of_contours_to_group_trackbar)
    cv.createTrackbar(intersection_name, window_contour_name, intersection, intersection_high, on_intersection_trackbar)
    cv.createTrackbar(contour_sorting_mode_name, window_contour_name, contour_sorting_mode, contour_sorting_mode_high, on_contour_sorting_mode_trackbar)

setupCaptureWindow()
setupMedianBlurWindow()
setupGaussianBlurWindow()
setupHSVThresholdingWindow()
setupContourWindow()

# instantiate PnP pipeline here

while True:
    # ret, frame = cap.read()
    # frame = cv.imread('test\\TestImages\\OuterGoal\\BlueGoal-084in-Center.jpg')
    # frame = cv.imread('test\\TestImages\\OuterGoal\\BlueGoal-330in-ProtectedZone.jpg')
    frame = cv.imread('data/targets/2019Hatch/test1.jpg')
    ret = True
    if (ret):
        # Perform a median blur to remove salt and pepper noise
        frame_median = frame#cv.medianBlur(frame, aperture_size)

        # Perform a Gaussian blur to remove Gaussian noise
        #frame_gaussian = cv.GaussianBlur(frame_median, (gaussian_kernel_size, gaussian_kernel_size), sigma_x, sigma_y)

        # Perform HSV thresholding
        frame_hsv = cv.cvtColor(frame_median, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))

        # Find contours within the image
        frame_contours = frame_threshold
        frame_temp = np.array(frame, copy=True)

        contours, hierarchy = cv.findContours(frame_contours, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        frame_contours = cv.drawContours(frame_temp, contours, -1, (0, 0, 255), 5)

        # processed_contours = processContours(contours)
        # filtered_contours = filterContours(processed_contours)
        # sorted_contours = sortContours(filtered_contours, contour_sorting_mode)
        # if (num_of_contours_to_group != 0):
        #     target = groupContours(sorted_contours, num_of_contours_to_group, intersection)
        # else:
        #     target = sorted_contours[0]  

        # if (type(target) is Contour):
        #     translationVector, rotationMatrix = pipeline.getTranslation(target.getContourPoints)
        # else:
        #     translationVector, rotationMatrix = pipeline.getTranslation(target.getPoints)

        # # Label the number of contours found
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame_contours, 'Contours found: ' + str(len(contours)), (3,10), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
        # cv.putText(frame_contours, 'X:' + translationVector[0] + ' Y:' + translationVector[1] + ' Z:' + translationVector[2], (3, 22), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
        # cv.putText(frame_contours, VisionUtil.getAngleToTarget(rotationMatrix) + ' deg', (3, 34), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)

        # cv.putText(frame_contours, 'Resolution: 640x480', (205, 10), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
        # cv.putText(frame_contours, 'FPS: 123', (265, 22), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)

        # Update the windows with the updated frame
        cv.imshow(window_capture_name, frame)
        cv.imshow(window_median_blur_name, frame_median)
        # cv.imshow(window_gaussian_blur_name, frame_gaussian)
        cv.imshow("thresholding view", frame_threshold)
        cv.imshow(window_contour_name, frame_contours)
    key = cv.waitKey(30)
    if key == ord("q"):
        cap.release()
        break