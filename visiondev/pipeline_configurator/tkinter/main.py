from visiondev.pipeline_configurator.tkinter.PipelineConfigurator import Application
from common.util.camera import USBCamera
from common.vision.contour import *
from common.vision import pnp
from common.util.math import Vector3, Rotation3, RigidTransform3
import json
from math import atan2, cos, sin, sqrt, pi
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import time

PATH_TO_CONFIG_FILE = 'data/calibration/configs/CameraConfig.cfg'
PATH_TO_TARGET_FILE = 'data/misc/TargetModel.mdl'

cap = USBCamera(PATH_TO_CONFIG_FILE, 0)

targetModel = open(PATH_TO_TARGET_FILE, 'r')
objectPoints = np.array(json.loads(targetModel.readline())['points'], dtype=np.float32)

font = cv.FONT_HERSHEY_SIMPLEX

app = Application()

# Rotates and expands an image to avoid cropping
def rotateImage(img, angle):
    # Get the dimensions and center of the image
    height, width = img.shape[:2]
    imgCenter = (width / 2, height / 2)
    
    # Now get our ratation matrix
    rotationMatrix = cv.getRotationMatrix2D(imgCenter, angle, 1)

    # Take the absolute value of the cos and sin from the rotation matrix
    absoluteCos = abs(rotationMatrix[0, 0])
    absoluteSin = abs(rotationMatrix[0, 1])

    # Find the new width and height bounds
    widthBound = int(height * absoluteSin + width * absoluteCos)
    heightBound = int(height * absoluteCos + width * absoluteSin)

    # Subtract the old image center from the rotation matrix (essentially bringing it back to the origin) and add the new corrdinates
    rotationMatrix[0, 2] += widthBound / 2 - imgCenter[0]
    rotationMatrix[1, 2] += heightBound / 2 - imgCenter[1]

    # Finally rotate the image with our modified rotation matrix
    rotatedImg = cv.warpAffine(img, rotationMatrix, (widthBound, heightBound))
    return rotatedImg

def pipeline():
    ret, frame = cap.getFrame()
    cameraSetupFrame = app.getController().frames['CameraSetupFrame']
    hsvFrame = app.getController().frames['HSVFrame']
    contourFilteringFrame = app.getController().frames['ContourFilteringFrame']
    contourPairingFrame = app.getController().frames['ContourPairingFrame']
    poseEstimationFrame = app.getController().frames['PoseEstimationFrame']
    if (ret):
        angle = cameraSetupFrame.getImageRotation()
        if (angle != 0):
            frame = rotateImage(frame, angle)

        blurOption = cameraSetupFrame.getBlurOption()
        if (blurOption == 'median'):
            frame = cv.medianBlur(frame, cameraSetupFrame.getAperatureSize())
        elif (blurOption == 'gaussian'):
            frame = cv.GaussianBlur(frame, cameraSetupFrame.getGaussianKernelSize(), cameraSetupFrame.getSigmaX(), cameraSetupFrame.getSigmaY())

        # Convert to RGB so colors will appear correctly in the GUI
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cameraSetupFrame.updateCanvas(frame_rgb)

        start = time.time()
        #  Now we threshold
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_hsv, hsvFrame.getLowTresholdingValues(), hsvFrame.getHighTresholdingValues())

        # Check if we should perform a erosion
        if (hsvFrame.getEnableErosionVal()):
            erosionKernel = cv.getStructuringElement(hsvFrame.getShapeOfErosionKernel(), hsvFrame.getSizeOfErosionKernel(), anchor=hsvFrame.getErosionKernelAnchor())
            frame_threshold = cv.erode(frame_threshold, erosionKernel, anchor=hsvFrame.getErosionAnchor(), iterations=hsvFrame.getErosionIterations())

        # Check if we should perform a dilation
        if (hsvFrame.getEnableDilationVal()):
            dilationKernel = cv.getStructuringElement(hsvFrame.getShapeOfDilationKernel(), hsvFrame.getSizeOfDilationKernel(), anchor=hsvFrame.getDilationKernelAnchor())
            frame_threshold = cv.dilate(frame_threshold, dilationKernel, anchor=hsvFrame.getDilationAnchor(), iterations=hsvFrame.getDilationIterations())

        hsvFrame.updateCanvas(frame_threshold)

        # Find the contours in the image
        contours = findContours(frame_threshold)

        # Process the contours
        processedContours = processContours(contours, cap.getFrameMidpoint(), False, 0)

        # Filter the contours
        targetAreaRange = contourFilteringFrame.getTargetAreaRange()
        targetFullnessRange = contourFilteringFrame.getTargetFullnessRange()
        aspectRatioRange = contourFilteringFrame.getAspectRatioRange()
        filteredContours = filterContours(processedContours, targetAreaRange, targetFullnessRange, aspectRatioRange, cap.getFrameSize())

        target = filteredContours

        if filteredContours is not None:
            # Sort the contours
            sortingMode = contourFilteringFrame.getSortingMode()
            sortedContours = sortContours(filteredContours, sortingMode)

            target = [sortedContours[0]]

            # Check if we should pair the contours
            if (contourPairingFrame.getEnableContourPairing()):
                intersectionLocation = contourPairingFrame.getIntersectionLocation()

                targetAreaRange = contourPairingFrame.getTargetAreaRange()
                targetFullnessRange = contourPairingFrame.getTargetFullnessRange()
                aspectRatioRange = contourPairingFrame.getAspectRatioRange()

                # Sorting mode for the contours pairs
                sortingMode = contourPairingFrame.getSortingMode()

                target = pairContours(sortedContours, intersectionLocation, targetAreaRange, targetFullnessRange, aspectRatioRange, sortingMode, cap.getFrameMidpoint(), cap.getFrameSize())

        rigidTransform = None
        imgpts = None
        angleToTarget = None
        if ((poseEstimationFrame.getEnablePoseEstimation()) and (target is not None) and (len(objectPoints) == len(target[0].getVerticies()))):
            range = [1, len(target[0].getVerticies())]
            if (poseEstimationFrame.getEnablePoseEstimationRange()):
                range = poseEstimationFrame.getRange()
            rigidTransform, imgpts = pnp.getTranslation(cap.getCameraMatrix(), cap.getDistortionCoefficients(), objectPoints, target[0].getVerticies(), range)
            angleToTarget = pnp.getAngleToTarget(rigidTransform.rotation.rotationMatrix)

        end = time.time()
        print(end - start)

        resolution = cap.getResolution()
        cv.putText(frame, str(resolution[0]) + 'x' + str(resolution[1]) + ' @' + str(cap.getFramerate()) + ' fps', (3, 11), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(frame, 'Contours found: ' + str(len(contours)), (3, 23), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
        
        if target is not None:
            # Draw and label the remaining bits
            frame_contours = drawContours(frame, target)
            labelVerticies(frame_contours, target)
            drawBoundingBoxes(frame_contours, target)
            if (rigidTransform is not None):
                cv.putText(frame_contours, str(rigidTransform.translation), (3, 35), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
                cv.putText(frame_contours, str(rigidTransform.rotation), (3, 47), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
                cv.putText(frame_contours, str(np.degrees(angleToTarget)), (3, 59), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
                frame_contours = drawPnPAxes(frame_contours, imgpts)
            else:
                drawReferenceVector(frame_contours, target)
            contourFilteringFrame.updateCanvas(frame_contours)
            contourPairingFrame.updateCanvas(frame_contours)
            poseEstimationFrame.updateCanvas(frame_contours)           
        else:
            contourFilteringFrame.updateCanvas(frame)
            contourPairingFrame.updateCanvas(frame)
            poseEstimationFrame.updateCanvas(frame)

    app.after(10, pipeline)

app.after(10, pipeline)

app.mainloop()