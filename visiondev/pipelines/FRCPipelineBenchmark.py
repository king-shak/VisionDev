'''
    This implmentation of the pipeline is the most up-to-date and sophisticated one. It uses the new Common library.

    This implmentation of the pipeline can also be found in the tkinter version of the pipeline configurator. The one in the OpenCV version
    is significantly out of date.

    Here you can choose a sample image and run it through the pipeline as a benchmark to evaluate its performance.
'''

from common.vision.pnp import TargetModel
from common.vision.contour import *
from common.util.math import Vector3, Rotation3, RigidTransform3
from common.vision.pnp import getTranslation, getAngleToTarget
import json
from math import atan2, cos, sin, sqrt, pi
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import time
import sys
import glob

# 640x480
resolution = (640, 480)
frameSize = 307200
frameMidpoint = (320, 240)
framerate = 60

distortionCoefficients = []
cameraMatrix = []

targetModel = TargetModel('data/misc/TargetModel.mdl')

def pipeline(frame):
    pipelineStart = time.time()
    #  Now we threshold
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # # 2019 target thresholding values
    # minThresh = (57, 105, 154)
    # maxThresh = (72, 255, 255)

    # 2020 target thresholding values
    minThresh = (69, 225, 165)
    maxThresh = (87, 255, 255)

    frame_threshold = cv.inRange(frame_hsv, minThresh, maxThresh)

    performErosion = False
    shapeOfErosionKernel = cv.MORPH_RECT
    sizeOfErosionKernel = (3, 3)
    erosionKernelAnchor = (-1, -1)
    erosionAnchor = (-1, -1)
    erosionIterations = 1

    performDilation = False
    shapeOfDilationKernel = cv.MORPH_RECT
    sizeOfDilationKernel = (3, 3)
    dilationKernelAnchor = (-1, -1)
    dilationAnchor = (-1, -1)
    dilationIterations = 1
    # Check if we should perform a erosion
    if (performErosion):
        erosionKernel = cv.getStructuringElement(shapeOfErosionKernel, sizeOfErosionKernel, anchor=erosionKernelAnchor)
        frame_threshold = cv.erode(frame_threshold, erosionKernel, anchor=erosionAnchor, iterations=erosionIterations)

    # Check if we should perform a dilation
    if (performDilation):
        dilationKernel = cv.getStructuringElement(shapeOfDilationKernel, sizeOfDilationKernel, anchor=dilationKernelAnchor)
        frame_threshold = cv.dilate(frame_threshold, dilationKernel, anchor=dilationAnchor, iterations=dilationIterations)

    # Find the contours in the image
    contours = findContours(frame_threshold)

    # Process the contours
    useConvexHull = True
    numOfContourCorners = 0
    processedContours = processContours(contours, frameMidpoint, useConvexHull, numOfContourCorners)

    # Filter the contours
    contourAreaRange = (0.0, 1.0)
    contourFullnessRange = (0.0, 1.0)
    contourAspectRatioRange = (0.0, 10.0)
    filteredContours = filterContours(processedContours, contourAreaRange, contourFullnessRange, contourAspectRatioRange, frameSize)

    target = filteredContours

    if filteredContours is not None:
        # Sort the contours
        sortingMode = 'center'
        sortedContours = sortContours(filteredContours, sortingMode)

        target = [sortedContours[0]]

        # Check if we should pair the contours
        _pairContours = False
        intersectionLocation = 'above'
        if (_pairContours):
            intersectionLocation = intersectionLocation

            targetAreaRange = (0.0, 1.0)
            targetFullnessRange = (0.0, 1.0)
            targetAspectRatioRange = (0.0, 10.0)

            # Sorting mode for the contours pairs
            targetSortingMode = 'center'

            useConvexHull = True
            numOfPairCorners = 6
            target = pairContours(sortedContours, intersectionLocation, targetAreaRange, targetFullnessRange, targetAspectRatioRange, targetSortingMode, frameMidpoint, frameSize, useConvexHull, numOfPairCorners)

    rigidTransform = None
    imgpts = None
    angleToTarget = None
    performPoseEstimation = False
    if (performPoseEstimation and (target is not None) and (len(targetModel.objPts) == len(target[0].vertices))):
        range = [1, len(target[0].vertices)]
        rigidTransform, imgpts = getTranslation(cameraMatrix, distortionCoefficients, targetModel.objPts, target[0].vertices, range)
        angleToTarget = getAngleToTarget(rigidTransform.rotation.rotationMatrix)

    pipelineEnd = time.time()
    print(pipelineEnd - pipelineStart)

    cv.putText(frame, str(resolution[0]) + 'x' + str(resolution[1]) + '@' + str(framerate) + ' fps', (3, 11), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(frame, 'Contours found: ' + str(len(contours)), (3, 23), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
    
    if target is not None:
        # Draw and label the remaining bits
        frame = drawContours(frame, target)
        labelVerticies(frame, target)
        drawBoundingBoxes(frame, target)
        if (rigidTransform is not None):
            cv.putText(frame, str(rigidTransform.translation), (3, 35), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(frame, str(rigidTransform.rotation), (3, 47), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(frame, str(np.degrees(angleToTarget)), (3, 59), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
            frame = drawPnPAxes(frame, imgpts)
        else:
            drawReferenceVector(frame, target)
        cv.imshow('Pipeline Output', frame)          
    else:
        cv.imshow('Pipeline Output', frame)
    cv.waitKey(0)

# This bit does it on the 2019 Hatch target
# frame = cv.imread('data/targets/2019Hatch/test1.jpg')
# pipeline(frame)

# This bit will go through the sample images of the 2020 goal
pathToFrames = 'data/targets/2020OuterGoal/*.jpg'
frames = glob.glob(pathToFrames)

for frame in frames:
    frame = cv.imread(frame)
    pipeline(frame)
    time.sleep(0.01)