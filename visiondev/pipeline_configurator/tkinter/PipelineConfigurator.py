import tkinter as tk
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np

class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.container = tk.Frame(self)
        self.container.pack(anchor=tk.CENTER, fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.menuBar = tk.Menu(self)
        
        self.pipelineMenu = tk.Menu(self.menuBar, tearoff=0)

        self.pipelineMenu.add_command(label='Camera Setup', command=lambda: self.show_frame("CameraSetupFrame"))
        self.pipelineMenu.add_command(label='Thresholding', command=lambda: self.show_frame("HSVFrame"))
        self.pipelineMenu.add_command(label='Contour Filtering', command=lambda: self.show_frame("ContourFilteringFrame"))
        self.pipelineMenu.add_command(label='Contour Pairing', command=lambda: self.show_frame("ContourPairingFrame"))
        self.pipelineMenu.add_command(label='Pose Estimation', command=lambda: self.show_frame("PoseEstimationFrame"))

        self.menuBar.add_cascade(label='Pipeline', menu=self.pipelineMenu)

        tk.Tk.config(self, menu=self.menuBar)

        # self.cameraSetupFrame = CameraSetupFrame(parent=self, controller=self)
        # self.hsvFrame = HSVFrame(parent=self, controller=self)

        self.frames = {}
        for F in (CameraSetupFrame, HSVFrame, ContourFilteringFrame, ContourPairingFrame, PoseEstimationFrame):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("CameraSetupFrame")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
    
    def getController(self):
        return self

class CameraSetupFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.master = parent
        #self.pack()

        # Variables for median blur
        self.aperatureSize = 5

        # Variables for gaussian blur

        # These must be positive and odd
        # Note that these can be passed to the function as either a tuple or a OpenCV size instance
        self.gaussianBlurKernelWidth = 5
        self.gaussianBlurKernelHeight = 5

        # The kernels standard deviation in the respective (X & Y) directions
        # if Y is zero, it is set to X, if both are 0, they are computed from the gaussian kernel passed to the function
        self.sigmaX = 0
        self.sigmaY = 0

        self.frame_init()

    def frame_init(self):        
        # Do the initial packing
        #self.pack(fill=tk.BOTH, expand=True)

        # Setup the frame and canvas with the image
        self.imgFrame = tk.Frame(self)
        self.imgFrame.pack(fill=tk.X)

        self.canvas = tk.Canvas(self.imgFrame)
        self.canvas.pack()

        # Set the canvas initially to a blank image
        self.cv_img = cv.cvtColor(np.zeros((640, 480, 3), np.uint8), cv.COLOR_BGR2RGB)
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        # Setup the frame to select a pre-processing filter
        self.filtersFrame = tk.Frame(self)
        self.filtersFrame.pack(fill=tk.X)

        self.filtersFrameLabel = tk.Label(self.filtersFrame, text='Select a filter', width=9)
        self.filtersFrameLabel.pack()

        # Create the StringVar() instance tied to our radio buttons, set the default option to 'neither' (none of the filters)
        self.filter = tk.StringVar()
        self.filter.set('neither')

        # Setup the no blur frame and button
        self.noBlurFrame = tk.Frame(self.filtersFrame)
        self.noBlurFrame.pack(fill=tk.X)

        self.noBlurRadioButton = tk.Radiobutton(self.noBlurFrame, text='No Filter', variable=self.filter, value='neither')
        self.noBlurRadioButton.pack(side=tk.LEFT)
        
        # Setup the median blur frame, button, entry and scale
        self.medianBlurFrame = tk.Frame(self.filtersFrame)
        self.medianBlurFrame.pack(fill=tk.X)
        
        self.medianBlurRadioButton = tk.Radiobutton(self.medianBlurFrame, text='Median Blur', variable=self.filter, value='median')
        self.medianBlurRadioButton.pack(side=tk.LEFT)

        self.medianBlurEntryStringVar = tk.StringVar()
        self.medianBlurEntry = tk.Entry(self.medianBlurFrame, textvariable=self.medianBlurEntryStringVar)
        self.medianBlurEntry.bind('<Return>', self.medianBlurEntryCallback)
        self.medianBlurEntry.pack(side=tk.LEFT)

        self.medianBlurScale = tk.Scale(self.medianBlurFrame, orient=tk.HORIZONTAL, from_=1, to=21, command=self.medianBlurScaleCallback)
        self.medianBlurScale.pack(fill=tk.X, expand=True)

        self.setAperatureSize(self.aperatureSize)

        # Now set the gaussian blur frames, button, entries and scales
        self.gaussianBlurFrame = tk.Frame(self.filtersFrame)
        self.gaussianBlurFrame.pack(fill=tk.X)

        self.gaussianBlurRadioButton = tk.Radiobutton(self.gaussianBlurFrame, text='Gaussian Blur', variable=self.filter, value='gaussian')
        self.gaussianBlurRadioButton.pack(side=tk.LEFT)

        self.gaussianBlurKernelFrame = tk.Frame(self.gaussianBlurFrame)
        self.gaussianBlurKernelFrame.pack(fill=tk.X, side=tk.LEFT)

        # Setup the gaussian kernel width frame, entry and scale
        self.gaussianBlurKernelWidthFrame = tk.Frame(self.gaussianBlurKernelFrame)
        self.gaussianBlurKernelWidthFrame.pack(fill=tk.X)

        self.gaussianBlurKernelWidthLabel = tk.Label(self.gaussianBlurKernelWidthFrame, text='Kernel Width')
        self.gaussianBlurKernelWidthLabel.pack(side=tk.LEFT)

        self.gaussianBlurKernelWidthEntryStringVar = tk.StringVar()
        self.gaussianBlurKernelWidthEntry = tk.Entry(self.gaussianBlurKernelWidthFrame, textvariable=self.gaussianBlurKernelWidthEntryStringVar)
        self.gaussianBlurKernelWidthEntry.bind('<Return>', self.gaussianBlurKernelWidthEntryCallback)
        self.gaussianBlurKernelWidthEntry.pack(side=tk.LEFT)

        self.gaussianBlurKernelWidthScale = tk.Scale(self.gaussianBlurKernelWidthFrame, orient=tk.HORIZONTAL, from_=1, to=21, command=self.gaussianBlurKernelWidthScaleCallback)
        self.gaussianBlurKernelWidthScale.pack(fill=tk.X, expand=True)

        self.setGaussianBlurKernelWidth(self.gaussianBlurKernelWidth)

        # Setup the gaussian kernel height frame, entry and scale
        self.gaussianBlurKernelHeightFrame = tk.Frame(self.gaussianBlurKernelFrame)
        self.gaussianBlurKernelHeightFrame.pack(fill=tk.X)

        self.gaussianBlurKernelHeightLabel = tk.Label(self.gaussianBlurKernelHeightFrame, text='Kernel Height')
        self.gaussianBlurKernelHeightLabel.pack(side=tk.LEFT)

        self.gaussianBlurKernelHeightEntryStringVar = tk.StringVar()
        self.gaussianBlurKernelHeightEntry = tk.Entry(self.gaussianBlurKernelHeightFrame, textvariable=self.gaussianBlurKernelHeightEntryStringVar)
        self.gaussianBlurKernelHeightEntry.bind('<Return>', self.gaussianBlurKernelHeightEntryCallback)
        self.gaussianBlurKernelHeightEntry.pack(side=tk.LEFT)

        self.gaussianBlurKernelHeightScale = tk.Scale(self.gaussianBlurKernelHeightFrame, orient=tk.HORIZONTAL, from_=1, to=21, command=self.gaussianBlurKernelHeightScaleCallback)
        self.gaussianBlurKernelHeightScale.pack(fill=tk.X, expand=True)

        self.setGaussianBlurKernelHeight(self.gaussianBlurKernelHeight)

        # Setup the geometirc operations frame
        self.geometricOperationsFrame = tk.Frame(self)
        self.geometricOperationsFrame.pack(fill=tk.X)

        self.geometricOperationsFrameImageRotationLabel = tk.Label(self.geometricOperationsFrame, text='Image Rotation', width='11')
        self.geometricOperationsFrameImageRotationLabel.pack(side=tk.LEFT)

        self.rotation = tk.IntVar()
        self.rotation.set(0)

        self.zeroDegRotationRadioButton = tk.Radiobutton(self.geometricOperationsFrame, text='0°', variable=self.rotation, value=0)
        self.zeroDegRotationRadioButton.pack(side=tk.LEFT)

        self.ninetyDegRotationRadioButton = tk.Radiobutton(self.geometricOperationsFrame, text='90°', variable=self.rotation, value=90)
        self.ninetyDegRotationRadioButton.pack(side=tk.LEFT)

        self.oneEightyDegRotationRadioButton = tk.Radiobutton(self.geometricOperationsFrame, text='180°', variable=self.rotation, value=180)
        self.oneEightyDegRotationRadioButton.pack(side=tk.LEFT)

        self.twoSeventyDegRotationRadioButton = tk.Radiobutton(self.geometricOperationsFrame, text='270°', variable=self.rotation, value=270)
        self.twoSeventyDegRotationRadioButton.pack(side=tk.LEFT)
 
    # This ensures the number returned is odd and greater than 0
    def verifyNumber(self, past, current):
        if (current < 1):
            return 1
        if (current % 2 == 0):
            if (current > past):
                current = current + 1
            else:
                current = current - 1
        return current

    def setAperatureSize(self, val):
        self.aperatureSize = val
        self.medianBlurEntryStringVar.set(str(self.aperatureSize))
        self.medianBlurScale.set(self.aperatureSize)

    def setGaussianBlurKernelWidth(self, val):
        self.gaussianBlurKernelWidth = val
        self.gaussianBlurKernelWidthEntryStringVar.set(str(self.gaussianBlurKernelWidth))
        self.gaussianBlurKernelWidthScale.set(self.gaussianBlurKernelWidth)
    
    def setGaussianBlurKernelHeight(self, val):
        self.gaussianBlurKernelHeight = val
        self.gaussianBlurKernelHeightEntryStringVar.set(str(self.gaussianBlurKernelHeight))
        self.gaussianBlurKernelHeightScale.set(self.gaussianBlurKernelHeight)

    def medianBlurEntryCallback(self, event):
        print('median blur entry callback triggered!')
        try:
            self.aperatureSize = self.verifyNumber(self.aperatureSize, int(event.widget.get()))
            self.medianBlurEntryStringVar.set(str(self.aperatureSize))
            self.medianBlurScale.set(self.aperatureSize)
        except:
            print('invalid entry!')
        return True

    def gaussianBlurKernelWidthEntryCallback(self, event):
        print('gaussian blur kernel width entry callback triggered!')
        try:
            self.gaussianBlurKernelWidth = self.verifyNumber(self.gaussianBlurKernelWidth, int(event.widget.get()))
            self.gaussianBlurKernelWidthEntryStringVar.set(str(self.gaussianBlurKernelWidth))
            self.gaussianBlurKernelWidthScale.set(self.gaussianBlurKernelWidth)
        except:
            print('invalid entry!')
        return True

    def gaussianBlurKernelHeightEntryCallback(self, event):
        print('gaussian blur kernel height entry callback triggered!')
        try:
            self.gaussianBlurKernelHeight = self.verifyNumber(self.gaussianBlurKernelHeight, int(event.widget.get()))
            self.gaussianBlurKernelHeightEntryStringVar.set(str(self.gaussianBlurKernelHeight))
            self.gaussianBlurKernelHeightScale.set(self.gaussianBlurKernelHeight)
        except:
            print('invalid entry!')
        return True

    def medianBlurScaleCallback(self, value=None):
        print('median blur scale callback triggered!')
        self.aperatureSize = self.verifyNumber(self.aperatureSize, self.medianBlurScale.get())
        self.medianBlurScale.set(self.aperatureSize)
        self.medianBlurEntryStringVar.set(str(self.aperatureSize))

    def gaussianBlurKernelWidthScaleCallback(self, value=None):
        print('gaussian blur kernel width scale callback triggered!')
        self.gaussianBlurKernelWidth = self.verifyNumber(self.gaussianBlurKernelWidth, self.gaussianBlurKernelWidthScale.get())
        self.gaussianBlurKernelWidthScale.set(self.gaussianBlurKernelWidth)
        self.gaussianBlurKernelWidthEntryStringVar.set(str(self.gaussianBlurKernelWidth))

    def gaussianBlurKernelHeightScaleCallback(self, value=None):
        print('gaussian blur kernel height scale callback triggered!')
        self.gaussianBlurKernelHeight = self.verifyNumber(self.gaussianBlurKernelHeight, self.gaussianBlurKernelHeightScale.get())
        self.gaussianBlurKernelHeightScale.set(self.gaussianBlurKernelHeight)
        self.gaussianBlurKernelHeightEntryStringVar.set(str(self.gaussianBlurKernelHeight))
    
    def getBlurOption(self):
        return self.filter.get()

    def getImageRotation(self):
        return self.rotation.get()

    def getAperatureSize(self):
        return self.aperatureSize

    def getGaussianKernelWidth(self):
        return self.gaussianBlurKernelWidth

    def getGaussianKernelHeight(self):
        return self.gaussianBlurKernelHeight

    def getGaussianKernelSize(self):
        return (self.gaussianBlurKernelWidth, self.gaussianBlurKernelHeight)

    def getSigmaX(self):
        return self.sigmaX

    def getSigmaY(self):
        return self.sigmaY

    def updateCanvas(self, _img):
        self.cv_img = _img
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.imgHeight = self.img.height()
        self.imgWidth = self.img.width()
        self.canvas.config(width=self.imgWidth, height=self.imgHeight)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

class HSVFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.master = parent
        #self.pack()
       
        # Variables for HSV Thresholding
        self.lowHue = 0
        self.lowSaturation = 0
        self.lowValue = 0

        self.highHue = 180
        self.highSaturation = 255
        self.highValue = 255

        # Variables for erosion values
        self.erosionAnchorX = -1
        self.erosionAnchorY = -1

        self.erosionIterations = 1

        self.erosionKernelWidth = 2
        self.erosionKernelHeight = 2

        self.erosionKernelAnchorX = -1
        self.erosionKernelAnchorY = -1

        # Variables for dilation values
        self.dilationAnchorX = -1
        self.dilationAnchorY = -1

        self.dilationIterations = 1

        self.dilationKernelWidth = 2
        self.dilationKernelHeight = 2

        self.dilationKernelAnchorX = -1
        self.dilationKernelAnchorY = -1

        self.frame_init()
    
    def frame_init(self):
        # Do the initial packing
        #self.pack(fill=tk.BOTH, expand=True)

        # Setup the frame with the image
        self.imgFrame = tk.Frame(self)
        self.imgFrame.pack(side=tk.TOP)

        self.canvas = tk.Canvas(self.imgFrame)
        self.canvas.pack()

        # Set the canvas initially to a blank image
        self.cv_img = cv.cvtColor(np.zeros((640, 480, 3), np.uint8), cv.COLOR_BGR2RGB)
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        self.thresholdingFrame = tk.Frame(self)
        self.thresholdingFrame.pack(side=tk.LEFT)

        # Setup the hue frame
        self.hueFrame = tk.Frame(self.thresholdingFrame)
        self.hueFrame.pack(fill=tk.X)

        # Low hue
        self.lowHueFrame = tk.Frame(self.hueFrame)
        self.lowHueFrame.pack(fill=tk.X)

        self.lowHueLabel = tk.Label(self.lowHueFrame, text='Low Hue', width=6)
        self.lowHueLabel.pack(side=tk.LEFT, padx=5, pady=5)

        self.lowHueScaleFrame = tk.Frame(self.lowHueFrame)
        self.lowHueScaleFrame.pack(fill=tk.X)
        
        self.lowHueScaleEntryStringVar = tk.StringVar()
        self.lowHueScaleEntry = tk.Entry(self.lowHueScaleFrame, textvariable=self.lowHueScaleEntryStringVar)
        self.lowHueScaleEntry.bind('<Return>', self.lowHueScaleEntryCallback)
        self.lowHueScaleEntry.pack(side=tk.LEFT, padx=5, pady=5)

        self.lowHueScale = tk.Scale(self.lowHueScaleFrame, orient=tk.HORIZONTAL, from_=0, to=180, command=self.lowHueScaleCallback)
        self.lowHueScaleCallback()
        self.lowHueScale.pack(fill=tk.X, padx=5, expand=True)

        # High hue
        self.highHueFrame = tk.Frame(self.hueFrame)
        self.highHueFrame.pack(fill=tk.X)

        self.highHueLabel = tk.Label(self.highHueFrame, text='High Hue', width=7)
        self.highHueLabel.pack(side=tk.LEFT, padx=5, pady=5)

        self.highHueScaleFrame = tk.Frame(self.highHueFrame)
        self.highHueScaleFrame.pack(fill=tk.X)
        
        self.highHueScaleEntryStringVar = tk.StringVar()
        self.highHueScaleEntry = tk.Entry(self.highHueScaleFrame, textvariable=self.highHueScaleEntryStringVar)
        self.highHueScaleEntry.bind('<Return>', self.highHueScaleEntryCallback)
        self.highHueScaleEntry.pack(side=tk.LEFT, padx=5, pady=5)

        self.highHueScale = tk.Scale(self.highHueScaleFrame, orient=tk.HORIZONTAL, from_=0, to=180, command=self.highHueScaleCallback)
        self.highHueScale.set(180)
        self.highHueScale.pack(fill=tk.X, padx=5, expand=True)

        # Setup the saturation frame
        self.saturationFrame = tk.Frame(self.thresholdingFrame)
        self.saturationFrame.pack(fill=tk.X)

        # Low saturation
        self.lowSaturationFrame = tk.Frame(self.saturationFrame)
        self.lowSaturationFrame.pack(fill=tk.X)

        self.lowSaturationLabel = tk.Label(self.lowSaturationFrame, text='Low Saturation', width=11)
        self.lowSaturationLabel.pack(side=tk.LEFT, padx=5, pady=5)

        self.lowSaturationScaleFrame = tk.Frame(self.lowSaturationFrame)
        self.lowSaturationScaleFrame.pack(fill=tk.X)
        
        self.lowSaturationScaleEntryStringVar = tk.StringVar()
        self.lowSaturationScaleEntry = tk.Entry(self.lowSaturationScaleFrame, textvariable=self.lowSaturationScaleEntryStringVar)
        self.lowSaturationScaleEntry.bind('<Return>', self.lowSaturationScaleEntryCallback)
        self.lowSaturationScaleEntry.pack(side=tk.LEFT, padx=5, pady=5)

        self.lowSaturationScale = tk.Scale(self.lowSaturationScaleFrame, orient=tk.HORIZONTAL, from_=0, to=255, command=self.lowSaturationScaleCallback)
        self.lowSaturationScaleCallback()
        self.lowSaturationScale.pack(fill=tk.X, padx=5, expand=True)

        # High saturation
        self.highSaturationFrame = tk.Frame(self.saturationFrame)
        self.highSaturationFrame.pack(fill=tk.X)

        self.highSaturationLabel = tk.Label(self.highSaturationFrame, text='High Saturation', width=11)
        self.highSaturationLabel.pack(side=tk.LEFT, padx=5, pady=5)

        self.highSaturationScaleFrame = tk.Frame(self.highSaturationFrame)
        self.highSaturationScaleFrame.pack(fill=tk.X)
        
        self.highSaturationScaleEntryStringVar = tk.StringVar()
        self.highSaturationScaleEntry = tk.Entry(self.highSaturationScaleFrame, textvariable=self.highSaturationScaleEntryStringVar)
        self.highSaturationScaleEntry.bind('<Return>', self.highSaturationScaleEntryCallback)
        self.highSaturationScaleEntry.pack(side=tk.LEFT, padx=5, pady=5)

        self.highSaturationScale = tk.Scale(self.highSaturationScaleFrame, orient=tk.HORIZONTAL, from_=0, to=255, command=self.highSaturationScaleCallback)
        self.highSaturationScale.set(255)
        self.highSaturationScale.pack(fill=tk.X, padx=5, expand=True)

        # Setup the value frame
        self.valueFrame = tk.Frame(self.thresholdingFrame)
        self.valueFrame.pack(fill=tk.X)

        # Low value
        self.lowValueFrame = tk.Frame(self.valueFrame)
        self.lowValueFrame.pack(fill=tk.X)

        self.lowValueLabel = tk.Label(self.lowValueFrame, text='Low Value', width=8)
        self.lowValueLabel.pack(side=tk.LEFT, padx=5, pady=5)

        self.lowValueScaleFrame = tk.Frame(self.lowValueFrame)
        self.lowValueScaleFrame.pack(fill=tk.X)
        
        self.lowValueScaleEntryStringVar = tk.StringVar()
        self.lowValueScaleEntry = tk.Entry(self.lowValueScaleFrame, textvariable=self.lowValueScaleEntryStringVar)
        self.lowValueScaleEntry.bind('<Return>', self.lowValueScaleEntryCallback)
        self.lowValueScaleEntry.pack(side=tk.LEFT, padx=5, pady=5)

        self.lowValueScale = tk.Scale(self.lowValueScaleFrame, orient=tk.HORIZONTAL, from_=0, to=255, command=self.lowValueScaleCallback)
        self.lowValueScaleCallback()
        self.lowValueScale.pack(fill=tk.X, padx=5, expand=True)

        # High value
        self.highValueFrame = tk.Frame(self.valueFrame)
        self.highValueFrame.pack(fill=tk.X)

        self.highValueLabel = tk.Label(self.highValueFrame, text='High Value', width=8)
        self.highValueLabel.pack(side=tk.LEFT, padx=5, pady=5)

        self.highValueScaleFrame = tk.Frame(self.highValueFrame)
        self.highValueScaleFrame.pack(fill=tk.X)
        
        self.highValueScaleEntryStringVar = tk.StringVar()
        self.highValueScaleEntry = tk.Entry(self.highValueScaleFrame, textvariable=self.highValueScaleEntryStringVar)
        self.highValueScaleEntry.bind('<Return>', self.highValueScaleEntryCallback)
        self.highValueScaleEntry.pack(side=tk.LEFT, padx=5, pady=5)

        self.highValueScale = tk.Scale(self.highValueScaleFrame, orient=tk.HORIZONTAL, from_=0, to=255, command=self.highValueScaleCallback)
        self.highValueScale.set(255)
        self.highValueScale.pack(fill=tk.X, padx=5, expand=True)

        # Now, we set up the morphological operations frame
        # self.morphologicalOperationsFrame = tk.Frame(self)
        # self.morphologicalOperationsFrame.pack(side=tk.RIGHT)

        # self.morphologicalOperationsFrameLabel = tk.Label(self.morphologicalOperationsFrame, text='Morphological Operations')
        
        # Now we setup the erosion area
        self.erosionFrame = tk.Frame(self)
        self.erosionFrame.pack(side=tk.LEFT)

        self.mainErodeFrame = tk.Frame(self.erosionFrame)
        self.mainErodeFrame.pack(fill=tk.X)

        # This is the uppermost frame with the checkbox to enable/disable the erode functionality
        self.enableErosionFrame = tk.Frame(self.mainErodeFrame)
        self.enableErosionFrame.pack(fill=tk.X)

        self.enableErosionIntVar = tk.IntVar()
        self.enableErosionIntVar.set(0)
        self.enableErosionCheckButton = tk.Checkbutton(self.enableErosionFrame, text='Erode', variable=self.enableErosionIntVar, onvalue=1, offvalue=0)
        self.enableErosionCheckButton.pack(side=tk.LEFT)

        # This is the anchor frame

        # First, the x component
        self.erosionAnchorFrame = tk.Frame(self.mainErodeFrame)
        self.erosionAnchorFrame.pack(fill=tk.X)

        self.erosionAnchorXFrame = tk.Frame(self.erosionAnchorFrame)
        self.erosionAnchorXFrame.pack(fill=tk.X)

        self.erosionAnchorXLabel = tk.Label(self.erosionAnchorXFrame, text='Anchor X Position')
        self.erosionAnchorXLabel.pack(side=tk.LEFT)
        
        self.erosionAnchorXEntryStringVar = tk.StringVar()
        self.erosionAnchorXEntry = tk.Entry(self.erosionAnchorXFrame, textvariable=self.erosionAnchorXEntryStringVar)
        self.erosionAnchorXEntry.bind('<Return>', self.erosionAnchorXEntryCallback)
        self.erosionAnchorXEntry.pack(side=tk.LEFT)

        self.erosionAnchorXScale = tk.Scale(self.erosionAnchorXFrame, orient=tk.HORIZONTAL, from_=-1, to=9, command=self.erosionAnchorXScaleCallback)
        self.erosionAnchorXScale.pack(fill=tk.X, expand=True)

        self.setErosionAnchorX(-1)

        # Next, the y component
        self.erosionAnchorYFrame = tk.Frame(self.erosionAnchorFrame)
        self.erosionAnchorYFrame.pack(fill=tk.X)

        self.erosionAnchorYLabel = tk.Label(self.erosionAnchorYFrame, text='Anchor Y Position')
        self.erosionAnchorYLabel.pack(side=tk.LEFT)
        
        self.erosionAnchorYEntryStringVar = tk.StringVar()
        self.erosionAnchorYEntry = tk.Entry(self.erosionAnchorYFrame, textvariable=self.erosionAnchorYEntryStringVar)
        self.erosionAnchorYEntry.bind('<Return>', self.erosionAnchorYEntryCallback)
        self.erosionAnchorYEntry.pack(side=tk.LEFT)

        self.erosionAnchorYScale = tk.Scale(self.erosionAnchorYFrame, orient=tk.HORIZONTAL, from_=-1, to=9, command=self.erosionAnchorYScaleCallback)
        self.erosionAnchorYScale.pack(fill=tk.X, expand=True)

        self.setErosionAnchorY(-1)

        # Now we do the iterations scale
        self.erosionIterationsFrame = tk.Frame(self.mainErodeFrame)
        self.erosionIterationsFrame.pack(fill=tk.X)

        self.erosionIterationsLabel = tk.Label(self.erosionIterationsFrame, text='# of Iterations')
        self.erosionIterationsLabel.pack(side=tk.LEFT)

        self.erosionIterationsEntryStringVar = tk.StringVar()
        self.erosionIterationsEntry = tk.Entry(self.erosionIterationsFrame, textvariable=self.erosionIterationsEntryStringVar)
        self.erosionIterationsEntry.bind('<Return>', self.erosionIterationsEntryCallback)
        self.erosionIterationsEntry.pack(side=tk.LEFT)
        
        self.erosionIterationsScale = tk.Scale(self.erosionIterationsFrame, orient=tk.HORIZONTAL, from_=1, to=20, command=self.erosionIterationsScaleCallback)
        self.erosionIterationsScale.pack(fill=tk.X, expand=True)

        self.setErosionIterations(1)

        # Now we set up the erosion kernel frame
        self.erosionKernelFrame = tk.Frame(self.erosionFrame)
        self.erosionKernelFrame.pack(fill=tk.X)

        # First we set up the radiobuttons for the shape of the kernel
        self.erosionKernelShapeFrame = tk.Frame(self.erosionKernelFrame)
        self.erosionKernelShapeFrame.pack(fill=tk.X)

        self.erosionKernelShapeLabel = tk.Label(self.erosionKernelShapeFrame, text='Shape of Erosion Kernel')
        self.erosionKernelShapeLabel.pack(side=tk.LEFT)

        self.erosionKernelShapeStringVar = tk.StringVar()
        self.erosionKernelShapeStringVar.set('rectangle')

        self.rectangleKernelShapeRadioButtonErosion = tk.Radiobutton(self.erosionKernelShapeFrame, text='Rectangle', variable=self.erosionKernelShapeStringVar, value='rectangle')
        self.rectangleKernelShapeRadioButtonErosion.pack(side=tk.LEFT)
        
        self.ellipseKernelShapeRadioButtonErosion = tk.Radiobutton(self.erosionKernelShapeFrame, text='Ellipse', variable=self.erosionKernelShapeStringVar, value='ellipse')
        self.ellipseKernelShapeRadioButtonErosion.pack(side=tk.LEFT)
        
        self.crossKernelShapeRadioButtonErosion = tk.Radiobutton(self.erosionKernelShapeFrame, text='Cross', variable=self.erosionKernelShapeStringVar, value='cross')
        self.crossKernelShapeRadioButtonErosion.pack(side=tk.LEFT)

        # Setup the erosion kernel width frame, entry and scale
        self.erosionKernelWidthFrame = tk.Frame(self.erosionKernelFrame)
        self.erosionKernelWidthFrame.pack(fill=tk.X)

        self.erosionKernelWidthLabel = tk.Label(self.erosionKernelWidthFrame, text='Kernel Width')
        self.erosionKernelWidthLabel.pack(side=tk.LEFT)

        self.erosionKernelWidthEntryStringVar = tk.StringVar()
        self.erosionKernelWidthEntry = tk.Entry(self.erosionKernelWidthFrame, textvariable=self.erosionKernelWidthEntryStringVar)
        self.erosionKernelWidthEntry.bind('<Return>', self.erosionKernelWidthEntryCallback)
        self.erosionKernelWidthEntry.pack(side=tk.LEFT)

        self.erosionKernelWidthScale = tk.Scale(self.erosionKernelWidthFrame, orient=tk.HORIZONTAL, from_=1, to=9, command=self.erosionKernelWidthScaleCallback)
        self.erosionKernelWidthScale.pack(fill=tk.X, expand=True)

        self.setErosionKernelWidth(self.erosionKernelWidth)

        # Setup the erosion kernel height frame, entry and scale
        self.erosionKernelHeightFrame = tk.Frame(self.erosionKernelFrame)
        self.erosionKernelHeightFrame.pack(fill=tk.X)

        self.erosionKernelHeightLabel = tk.Label(self.erosionKernelHeightFrame, text='Kernel Height')
        self.erosionKernelHeightLabel.pack(side=tk.LEFT)

        self.erosionKernelHeightEntryStringVar = tk.StringVar()
        self.erosionKernelHeightEntry = tk.Entry(self.erosionKernelHeightFrame, textvariable=self.erosionKernelHeightEntryStringVar)
        self.erosionKernelHeightEntry.bind('<Return>', self.erosionKernelHeightEntryCallback)
        self.erosionKernelHeightEntry.pack(side=tk.LEFT)

        self.erosionKernelHeightScale = tk.Scale(self.erosionKernelHeightFrame, orient=tk.HORIZONTAL, from_=1, to=9, command=self.erosionKernelHeightScaleCallback)
        self.erosionKernelHeightScale.pack(fill=tk.X, expand=True)

        self.setErosionKernelHeight(self.erosionKernelHeight)

        # Now set up the erosion kernel anchor items

        # First, the x component
        self.erosionKernelAnchorFrame = tk.Frame(self.erosionKernelFrame)
        self.erosionKernelAnchorFrame.pack(fill=tk.X)

        self.erosionKernelAnchorXFrame = tk.Frame(self.erosionKernelAnchorFrame)
        self.erosionKernelAnchorXFrame.pack(fill=tk.X)

        self.erosionKernelAnchorXLabel = tk.Label(self.erosionKernelAnchorXFrame, text='Kernel Anchor X Position')
        self.erosionKernelAnchorXLabel.pack(side=tk.LEFT)
        
        self.erosionKernelAnchorXEntryStringVar = tk.StringVar()
        self.erosionKernelAnchorXEntry = tk.Entry(self.erosionKernelAnchorXFrame, textvariable=self.erosionKernelAnchorXEntryStringVar)
        self.erosionKernelAnchorXEntry.bind('<Return>', self.erosionKernelAnchorXEntryCallback)
        self.erosionKernelAnchorXEntry.pack(side=tk.LEFT)

        self.erosionKernelAnchorXScale = tk.Scale(self.erosionKernelAnchorXFrame, orient=tk.HORIZONTAL, from_=-1, to=2, command=self.erosionKernelAnchorXScaleCallback)
        self.erosionKernelAnchorXScale.pack(fill=tk.X, expand=True)

        self.setErosionKernelAnchorX(-1)

        # Next, the y component
        self.erosionKernelAnchorYFrame = tk.Frame(self.erosionKernelAnchorFrame)
        self.erosionKernelAnchorYFrame.pack(fill=tk.X)

        self.erosionKernelAnchorYLabel = tk.Label(self.erosionKernelAnchorYFrame, text='Kernel Anchor Y Position')
        self.erosionKernelAnchorYLabel.pack(side=tk.LEFT)
        
        self.erosionKernelAnchorYEntryStringVar = tk.StringVar()
        self.erosionKernelAnchorYEntry = tk.Entry(self.erosionKernelAnchorYFrame, textvariable=self.erosionKernelAnchorYEntryStringVar)
        self.erosionKernelAnchorYEntry.bind('<Return>', self.erosionKernelAnchorYEntryCallback)
        self.erosionKernelAnchorYEntry.pack(side=tk.LEFT)

        self.erosionKernelAnchorYScale = tk.Scale(self.erosionKernelAnchorYFrame, orient=tk.HORIZONTAL, from_=-1, to=2, command=self.erosionKernelAnchorYScaleCallback)
        self.erosionKernelAnchorYScale.pack(fill=tk.X, expand=True)

        self.setErosionKernelAnchorY(-1)

        # Now, we do everything for the we did for erosion for dilation

        # Now we setup the dilation area
        self.dilationFrame = tk.Frame(self)
        self.dilationFrame.pack(side=tk.LEFT)

        self.mainDilateFrame = tk.Frame(self.dilationFrame)
        self.mainDilateFrame.pack(fill=tk.X)

        # This is the uppermost frame with the checkbox to enable/disable the dilate functionality
        self.enableDilationFrame = tk.Frame(self.mainDilateFrame)
        self.enableDilationFrame.pack(fill=tk.X)

        self.enableDilationIntVar = tk.IntVar()
        self.enableDilationIntVar.set(0)
        self.enableDilationCheckButton = tk.Checkbutton(self.enableDilationFrame, text='Dilate', variable=self.enableDilationIntVar, onvalue=1, offvalue=0)
        self.enableDilationCheckButton.pack(side=tk.LEFT)

        # This is the anchor frame

        # First, the x component
        self.dilationAnchorFrame = tk.Frame(self.mainDilateFrame)
        self.dilationAnchorFrame.pack(fill=tk.X)

        self.dilationAnchorXFrame = tk.Frame(self.dilationAnchorFrame)
        self.dilationAnchorXFrame.pack(fill=tk.X)

        self.dilationAnchorXLabel = tk.Label(self.dilationAnchorXFrame, text='Anchor X Position')
        self.dilationAnchorXLabel.pack(side=tk.LEFT)
        
        self.dilationAnchorXEntryStringVar = tk.StringVar()
        self.dilationAnchorXEntry = tk.Entry(self.dilationAnchorXFrame, textvariable=self.dilationAnchorXEntryStringVar)
        self.dilationAnchorXEntry.bind('<Return>', self.dilationAnchorXEntryCallback)
        self.dilationAnchorXEntry.pack(side=tk.LEFT)

        self.dilationAnchorXScale = tk.Scale(self.dilationAnchorXFrame, orient=tk.HORIZONTAL, from_=-1, to=2, command=self.dilationAnchorXScaleCallback)
        self.dilationAnchorXScale.pack(fill=tk.X, expand=True)

        self.setDilationAnchorX(-1)

        # Next, the y component
        self.dilationAnchorYFrame = tk.Frame(self.dilationAnchorFrame)
        self.dilationAnchorYFrame.pack(fill=tk.X)

        self.dilationAnchorYLabel = tk.Label(self.dilationAnchorYFrame, text='Anchor Y Position')
        self.dilationAnchorYLabel.pack(side=tk.LEFT)
        
        self.dilationAnchorYEntryStringVar = tk.StringVar()
        self.dilationAnchorYEntry = tk.Entry(self.dilationAnchorYFrame, textvariable=self.dilationAnchorYEntryStringVar)
        self.dilationAnchorYEntry.bind('<Return>', self.dilationAnchorYEntryCallback)
        self.dilationAnchorYEntry.pack(side=tk.LEFT)

        self.dilationAnchorYScale = tk.Scale(self.dilationAnchorYFrame, orient=tk.HORIZONTAL, from_=-1, to=2, command=self.dilationAnchorYScaleCallback)
        self.dilationAnchorYScale.pack(fill=tk.X, expand=True)

        self.setDilationAnchorY(-1)

        # Now we do the iterations scale
        self.dilationIterationsFrame = tk.Frame(self.mainDilateFrame)
        self.dilationIterationsFrame.pack(fill=tk.X)

        self.dilationIterationsLabel = tk.Label(self.dilationIterationsFrame, text='# of Iterations')
        self.dilationIterationsLabel.pack(side=tk.LEFT)

        self.dilationIterationsEntryStringVar = tk.StringVar()
        self.dilationIterationsEntry = tk.Entry(self.dilationIterationsFrame, textvariable=self.dilationIterationsEntryStringVar)
        self.dilationIterationsEntry.bind('<Return>', self.dilationIterationsEntryCallback)
        self.dilationIterationsEntry.pack(side=tk.LEFT)
        
        self.dilationIterationsScale = tk.Scale(self.dilationIterationsFrame, orient=tk.HORIZONTAL, from_=1, to=20, command=self.dilationIterationsScaleCallback)
        self.dilationIterationsScale.pack(fill=tk.X, expand=True)

        self.setDilationIterations(1)

        # Now we set up the dilation kernel frame
        self.dilationKernelFrame = tk.Frame(self.dilationFrame)
        self.dilationKernelFrame.pack(fill=tk.X)

        # First we set up the radiobuttons for the shape of the kernel
        self.dilationKernelShapeFrame = tk.Frame(self.dilationKernelFrame)
        self.dilationKernelShapeFrame.pack(fill=tk.X)

        self.dilationKernelShapeLabel = tk.Label(self.dilationKernelShapeFrame, text='Shape of Dilation Kernel')
        self.dilationKernelShapeLabel.pack(side=tk.LEFT)

        self.dilationKernelShapeStringVar = tk.StringVar()
        self.dilationKernelShapeStringVar.set('rectangle')

        self.rectangleKernelShapeRadioButtonDilation = tk.Radiobutton(self.dilationKernelShapeFrame, text='Rectangle', variable=self.dilationKernelShapeStringVar, value='rectangle')
        self.rectangleKernelShapeRadioButtonDilation.pack(side=tk.LEFT)
        
        self.ellipseKernelShapeRadioButtonDilation = tk.Radiobutton(self.dilationKernelShapeFrame, text='Ellipse', variable=self.dilationKernelShapeStringVar, value='ellipse')
        self.ellipseKernelShapeRadioButtonDilation.pack(side=tk.LEFT)
        
        self.crossKernelShapeRadioButtonDilation = tk.Radiobutton(self.dilationKernelShapeFrame, text='Cross', variable=self.dilationKernelShapeStringVar, value='cross')
        self.crossKernelShapeRadioButtonDilation.pack(side=tk.LEFT)

        # Setup the dilation kernel width frame, entry and scale
        self.dilationKernelWidthFrame = tk.Frame(self.dilationKernelFrame)
        self.dilationKernelWidthFrame.pack(fill=tk.X)

        self.dilationKernelWidthLabel = tk.Label(self.dilationKernelWidthFrame, text='Kernel Width')
        self.dilationKernelWidthLabel.pack(side=tk.LEFT)

        self.dilationKernelWidthEntryStringVar = tk.StringVar()
        self.dilationKernelWidthEntry = tk.Entry(self.dilationKernelWidthFrame, textvariable=self.dilationKernelWidthEntryStringVar)
        self.dilationKernelWidthEntry.bind('<Return>', self.dilationKernelWidthEntryCallback)
        self.dilationKernelWidthEntry.pack(side=tk.LEFT)

        self.dilationKernelWidthScale = tk.Scale(self.dilationKernelWidthFrame, orient=tk.HORIZONTAL, from_=1, to=9, command=self.dilationKernelWidthScaleCallback)
        self.dilationKernelWidthScale.pack(fill=tk.X, expand=True)

        self.setDilationKernelWidth(self.dilationKernelWidth)

        # Setup the erodsion kernel height frame, entry and scale
        self.dilationKernelHeightFrame = tk.Frame(self.dilationKernelFrame)
        self.dilationKernelHeightFrame.pack(fill=tk.X)

        self.dilationKernelHeightLabel = tk.Label(self.dilationKernelHeightFrame, text='Kernel Height')
        self.dilationKernelHeightLabel.pack(side=tk.LEFT)

        self.dilationKernelHeightEntryStringVar = tk.StringVar()
        self.dilationKernelHeightEntry = tk.Entry(self.dilationKernelHeightFrame, textvariable=self.dilationKernelHeightEntryStringVar)
        self.dilationKernelHeightEntry.bind('<Return>', self.dilationKernelHeightEntryCallback)
        self.dilationKernelHeightEntry.pack(side=tk.LEFT)

        self.dilationKernelHeightScale = tk.Scale(self.dilationKernelHeightFrame, orient=tk.HORIZONTAL, from_=1, to=9, command=self.dilationKernelHeightScaleCallback)
        self.dilationKernelHeightScale.pack(fill=tk.X, expand=True)

        self.setDilationKernelHeight(self.dilationKernelHeight)

        # Now set up the dilation kernel anchor items

        # First, the x component
        self.dilationKernelAnchorFrame = tk.Frame(self.dilationKernelFrame)
        self.dilationKernelAnchorFrame.pack(fill=tk.X)

        self.dilationKernelAnchorXFrame = tk.Frame(self.dilationKernelAnchorFrame)
        self.dilationKernelAnchorXFrame.pack(fill=tk.X)

        self.dilationKernelAnchorXLabel = tk.Label(self.dilationKernelAnchorXFrame, text='Kernel Anchor X Position')
        self.dilationKernelAnchorXLabel.pack(side=tk.LEFT)
        
        self.dilationKernelAnchorXEntryStringVar = tk.StringVar()
        self.dilationKernelAnchorXEntry = tk.Entry(self.dilationKernelAnchorXFrame, textvariable=self.dilationKernelAnchorXEntryStringVar)
        self.dilationKernelAnchorXEntry.bind('<Return>', self.dilationKernelAnchorXEntryCallback)
        self.dilationKernelAnchorXEntry.pack(side=tk.LEFT)

        self.dilationKernelAnchorXScale = tk.Scale(self.dilationKernelAnchorXFrame, orient=tk.HORIZONTAL, from_=-1, to=2, command=self.dilationKernelAnchorXScaleCallback)
        self.dilationKernelAnchorXScale.pack(fill=tk.X, expand=True)

        self.setDilationKernelAnchorX(-1)

        # Next, the y component
        self.dilationKernelAnchorYFrame = tk.Frame(self.dilationKernelAnchorFrame)
        self.dilationKernelAnchorYFrame.pack(fill=tk.X)

        self.dilationKernelAnchorYLabel = tk.Label(self.dilationKernelAnchorYFrame, text='Kernel Anchor Y Position')
        self.dilationKernelAnchorYLabel.pack(side=tk.LEFT)
        
        self.dilationKernelAnchorYEntryStringVar = tk.StringVar()
        self.dilationKernelAnchorYEntry = tk.Entry(self.dilationKernelAnchorYFrame, textvariable=self.dilationKernelAnchorYEntryStringVar)
        self.dilationKernelAnchorYEntry.bind('<Return>', self.dilationKernelAnchorYEntryCallback)
        self.dilationKernelAnchorYEntry.pack(side=tk.LEFT)

        self.dilationKernelAnchorYScale = tk.Scale(self.dilationKernelAnchorYFrame, orient=tk.HORIZONTAL, from_=-1, to=2, command=self.dilationKernelAnchorYScaleCallback)
        self.dilationKernelAnchorYScale.pack(fill=tk.X, expand=True)

        self.setDilationKernelAnchorY(-1)
    
    def lowHueScaleEntryCallback(self, event):
        print('low hue entry callback triggered!')
        try:
            self.lowHue = int(event.widget.get())
            self.lowHueScale.set(self.lowHue)
        except:
            print("invalid entry!")
        return True

    def highHueScaleEntryCallback(self, event):
        print('high hue entry callback triggered!')
        try:
            self.highHue = int(event.widget.get())
            self.highHueScale.set(self.highHue)
        except:
            print("invalid entry!")
        return True

    def lowSaturationScaleEntryCallback(self, event):
        print('low saturation entry callback triggered!')
        try:
            self.lowSaturation = int(event.widget.get())
            self.lowSaturationScale.set(self.lowSaturation)
        except:
            print("invalid entry!")
        return True

    def highSaturationScaleEntryCallback(self, event):
        print('high saturation entry callback triggered!')
        try:
            self.highSaturation = int(event.widget.get())
            self.highSaturationScale.set(self.highSaturation)
        except:
            print("invalid entry!")
        return True

    def lowValueScaleEntryCallback(self, event):
        print('low value entry callback triggered!')
        try:
            self.lowValue = int(event.widget.get())
            self.lowValueScale.set(self.lowValue)
        except:
            print("invalid entry!")
        return True

    def highValueScaleEntryCallback(self, event):
        print('high value entry callback triggered!')
        try:
            self.highValue = int(event.widget.get())
            self.highValueScale.set(self.highValue)
        except:
            print("invalid entry!")
        return True

    def erosionAnchorXEntryCallback(self, event):
        print('erosion anchor x entry callback triggered!')
        try:
            self.erosionAnchorX = int(event.widget.get())
            self.erosionAnchorXScale.set(self.erosionAnchorX)
        except:
            print('invalid entry!')
        return True

    def erosionAnchorYEntryCallback(self, event):
        print('erosion anchor y entry callback triggered!')
        try:
            self.erosionAnchorY = int(event.widget.get())
            self.erosionAnchorYScale.set(self.erosionAnchorY)
        except:
            print('invalid entry!')
        return True

    def erosionIterationsEntryCallback(self, event):
        print('erosion iterations entry callback triggered!')
        try:
            self.erosionIterations = int(event.widget.get())
            self.erosionIterationsScale.set(self.erosionIterations)
        except:
            print('invalid entry!')
        return True

    def erosionKernelWidthEntryCallback(self, event):
        print('erosion kernel width entry callback triggered!')
        try:
            self.erosionKernelWidth = int(event.widget.get())
            self.erosionKernelWidthScale.set(self.erosionKernelWidth)
        except:
            print('invalid entry!')
        return True

    def erosionKernelHeightEntryCallback(self, event):
        print('erosion kernel height entry callback triggered!')
        try:
            self.erosionKernelHeight = int(event.widget.get())
            self.erosionKernelHeightScale.set(self.erosionKernelHeight)
        except:
            print('invalid entry!')
        return True

    def erosionKernelAnchorXEntryCallback(self, event):
        print('erosion kernel anchor x entry callback triggered!')
        try:
            self.erosionKernelAnchorX = int(event.widget.get())
            self.erosionKernelAnchorXScale.set(self.erosionKernelAnchorX)
        except:
            print('invalid entry!')
        return True

    def erosionKernelAnchorYEntryCallback(self, event):
        print('erosion kernel anchor y entry callback triggered!')
        try:
            self.erosionKernelAnchorY = int(event.widget.get())
            self.erosionKernelAnchorYScale.set(self.erosionKernelAnchorY)
        except:
            print('invalid entry!')
        return True

    def dilationAnchorXEntryCallback(self, event):
        print('dilation anchor x entry callback triggered!')
        try:
            self.dilationAnchorX = int(event.widget.get())
            self.dilationAnchorXScale.set(self.dilationAnchorX)
        except:
            print('invalid entry!')
        return True

    def dilationAnchorYEntryCallback(self, event):
        print('dilation anchor y entry callback triggered!')
        try:
            self.dilationAnchorY = int(event.widget.get())
            self.dilationAnchorYScale.set(self.dilationAnchorY)
        except:
            print('invalid entry!')
        return True

    def dilationIterationsEntryCallback(self, event):
        print('dilation iterations entry callback triggered!')
        try:
            self.dilationIterations = int(event.widget.get())
            self.dilationIterationsScale.set(self.dilationIterations)
        except:
            print('invalid entry!')
        return True

    def dilationKernelWidthEntryCallback(self, event):
        print('dilation kernel width entry callback triggered!')
        try:
            self.dilationKernelWidth = int(event.widget.get())
            self.dilationKernelWidthScale.set(self.dilationKernelWidth)
        except:
            print('invalid entry!')
        return True

    def dilationKernelHeightEntryCallback(self, event):
        print('dilation kernel height entry callback triggered!')
        try:
            self.dilationKernelHeight = int(event.widget.get())
            self.dilationKernelHeightScale.set(self.dilationKernelHeight)
        except:
            print('invalid entry!')
        return True

    def dilationKernelAnchorXEntryCallback(self, event):
        print('dilation kernel anchor x entry callback triggered!')
        try:
            self.dilationKernelAnchorX = int(event.widget.get())
            self.dilationKernelAnchorXScale.set(self.dilationKernelAnchorX)
        except:
            print('invalid entry!')
        return True

    def dilationKernelAnchorYEntryCallback(self, event):
        print('dilation kernel anchor y entry callback triggered!')
        try:
            self.dilationKernelAnchorY = int(event.widget.get())
            self.dilationKernelAnchorYScale.set(self.dilationKernelAnchorY)
        except:
            print('invalid entry!')
        return True

    def lowHueScaleCallback(self, value=None):
        print('low hue scale callback triggered!')
        self.lowHue = self.lowHueScale.get()
        self.lowHueScaleEntryStringVar.set(str(self.lowHue))

    def highHueScaleCallback(self, value=None):
        print('high hue scale callback triggered!')
        self.highHue = self.highHueScale.get()
        self.highHueScaleEntryStringVar.set(str(self.highHue))

    def lowSaturationScaleCallback(self, value=None):
        print('low saturation scale callback triggered!')
        self.lowSaturation = self.lowSaturationScale.get()
        self.lowSaturationScaleEntryStringVar.set(str(self.lowSaturation))

    def highSaturationScaleCallback(self, value=None):
        print('high saturation scale callback triggered!')
        self.highSaturation = self.highSaturationScale.get()
        self.highSaturationScaleEntryStringVar.set(str(self.highSaturation))

    def lowValueScaleCallback(self, value=None):
        print('low value scale callback triggered!')
        self.lowValue = self.lowValueScale.get()
        self.lowValueScaleEntryStringVar.set(str(self.lowValue))

    def highValueScaleCallback(self, value=None):
        print('high value scale callback triggered!')
        self.highValue = self.highValueScale.get()
        self.highValueScaleEntryStringVar.set(str(self.highValue))

    def erosionAnchorXScaleCallback(self, value=None):
        print('erosion anchor x scale callback triggered!')
        self.erosionAnchorX = self.erosionAnchorXScale.get()
        self.erosionAnchorXEntryStringVar.set(str(self.erosionAnchorX))

    def erosionAnchorYScaleCallback(self, value=None):
        print('erosion anchor y scale callback triggered!')
        self.erosionAnchorY = self.erosionAnchorYScale.get()
        self.erosionAnchorYEntryStringVar.set(str(self.erosionAnchorY))

    def erosionIterationsScaleCallback(self, value=None):
        print('erosion iterations scale callback triggered!')
        self.erosionIterations = self.erosionIterationsScale.get()
        self.erosionIterationsEntryStringVar.set(str(self.erosionIterations))

    def erosionKernelWidthScaleCallback(self, value=None):
        print('erosion kernel width scale callback triggered!')
        self.erosionKernelWidth = self.erosionKernelWidthScale.get()
        self.erosionKernelWidthEntryStringVar.set(str(self.erosionKernelWidth))

    def erosionKernelHeightScaleCallback(self, value=None):
        print('erosion kernel height scale callback triggered!')
        self.erosionKernelHeight = self.erosionKernelHeightScale.get()
        self.erosionKernelHeightEntryStringVar.set(str(self.erosionKernelHeight))

    def erosionKernelAnchorXScaleCallback(self, value=None):
        print('erosion kernel anchor x scale callback triggered!')
        self.erosionKernelAnchorX = self.erosionKernelAnchorXScale.get()
        self.erosionKernelAnchorXEntryStringVar.set(str(self.erosionKernelAnchorX))

    def erosionKernelAnchorYScaleCallback(self, value=None):
        print('erosion kernel anchor y scale callback triggered!')
        self.erosionKernelAnchorY = self.erosionKernelAnchorYScale.get()
        self.erosionKernelAnchorYEntryStringVar.set(str(self.erosionKernelAnchorY))

    def dilationAnchorXScaleCallback(self, value=None):
        print('dilation anchor x scale callback triggered!')
        self.dilationAnchorX = self.dilationAnchorXScale.get()
        self.dilationAnchorXEntryStringVar.set(str(self.dilationAnchorX))

    def dilationAnchorYScaleCallback(self, value=None):
        print('dilation anchor y scale callback triggered!')
        self.dilationAnchorY = self.dilationAnchorYScale.get()
        self.dilationAnchorYEntryStringVar.set(str(self.dilationAnchorY))

    def dilationIterationsScaleCallback(self, value=None):
        print('dilation iterations scale callback triggered!')
        self.dilationIterations = self.dilationIterationsScale.get()
        self.dilationIterationsEntryStringVar.set(str(self.dilationIterations))

    def dilationKernelWidthScaleCallback(self, value=None):
        print('dilation kernel width scale callback triggered!')
        self.dilationKernelWidth = self.dilationKernelWidthScale.get()
        self.dilationKernelWidthEntryStringVar.set(str(self.dilationKernelWidth))

    def dilationKernelHeightScaleCallback(self, value=None):
        print('dilation kernel height scale callback triggered!')
        self.dilationKernelHeight = self.dilationKernelHeightScale.get()
        self.dilationKernelHeightEntryStringVar.set(str(self.dilationKernelHeight))

    def dilationKernelAnchorXScaleCallback(self, value=None):
        print('dilation kernel anchor x scale callback triggered!')
        self.dilationKernelAnchorX = self.dilationKernelAnchorXScale.get()
        self.dilationKernelAnchorXEntryStringVar.set(str(self.dilationKernelAnchorX))

    def dilationKernelAnchorYScaleCallback(self, value=None):
        print('dilation kernel anchor y scale callback triggered!')
        self.dilationAnchorY = self.dilationAnchorYScale.get()
        self.dilationAnchorYEntryStringVar.set(str(self.dilationAnchorY))

    def setErosionAnchorX(self, val):
        self.erosionAnchorX = val
        self.erosionAnchorXEntryStringVar.set(str(self.erosionAnchorX))
        self.erosionAnchorXScale.set(self.erosionAnchorX)

    def setErosionAnchorY(self, val):
        self.erosionAnchorY = val
        self.erosionAnchorYEntryStringVar.set(str(self.erosionAnchorY))
        self.erosionAnchorYScale.set(self.erosionAnchorY)

    def setErosionIterations(self, val):
        self.erosionIterations = val
        self.erosionIterationsEntryStringVar.set(str(self.erosionIterations))
        self.erosionIterationsScale.set(self.erosionIterations)

    def setErosionKernelWidth(self, val):
        self.erosionKernelWidth = val
        self.erosionKernelWidthEntryStringVar.set(str(self.erosionKernelWidth))
        self.erosionKernelWidthScale.set(self.erosionKernelWidth)

    def setErosionKernelHeight(self, val):
        self.erosionKernelHeight = val
        self.erosionKernelHeightEntryStringVar.set(str(self.erosionKernelHeight))
        self.erosionKernelHeightScale.set(self.erosionKernelHeight)

    def setErosionKernelAnchorX(self, val):
        self.erosionKernelAnchorX = val
        self.erosionKernelAnchorXEntryStringVar.set(str(self.erosionKernelAnchorX))
        self.erosionKernelAnchorXScale.set(self.erosionKernelAnchorX)

    def setErosionKernelAnchorY(self, val):
        self.erosionKernelAnchorY = val
        self.erosionKernelAnchorYEntryStringVar.set(str(self.erosionKernelAnchorY))
        self.erosionKernelAnchorYScale.set(self.erosionKernelAnchorY)

    def setDilationAnchorX(self, val):
        self.dilationAnchorX = val
        self.dilationAnchorXEntryStringVar.set(str(self.dilationAnchorX))
        self.dilationAnchorXScale.set(self.dilationAnchorX)

    def setDilationAnchorY(self, val):
        self.dilationAnchorY = val
        self.dilationAnchorYEntryStringVar.set(str(self.dilationAnchorY))
        self.dilationAnchorYScale.set(self.dilationAnchorY)

    def setDilationIterations(self, val):
        self.dilationIterations = val
        self.dilationIterationsEntryStringVar.set(str(self.dilationIterations))
        self.dilationIterationsScale.set(self.dilationIterations)

    def setDilationKernelWidth(self, val):
        self.dilationKernelWidth = val
        self.dilationKernelWidthEntryStringVar.set(str(self.dilationKernelWidth))
        self.dilationKernelWidthScale.set(self.dilationKernelWidth)

    def setDilationKernelHeight(self, val):
        self.dilationKernelHeight = val
        self.dilationKernelHeightEntryStringVar.set(str(self.dilationKernelHeight))
        self.dilationKernelHeightScale.set(self.dilationKernelHeight)

    def setDilationKernelAnchorX(self, val):
        self.dilationKernelAnchorX = val
        self.dilationKernelAnchorXEntryStringVar.set(str(self.dilationKernelAnchorX))
        self.dilationKernelAnchorXScale.set(self.dilationKernelAnchorX)

    def setDilationKernelAnchorY(self, val):
        self.dilationKernelAnchorY = val
        self.dilationKernelAnchorYEntryStringVar.set(str(self.dilationKernelAnchorY))
        self.dilationKernelAnchorYScale.set(self.dilationKernelAnchorY)
    
    def getLowTresholdingValues(self):
        return (self.lowHue, self.lowSaturation, self.lowValue)

    def getHighTresholdingValues(self):
        return (self.highHue, self.highSaturation, self.highValue)

    def getEnableErosionVal(self):
        return self.enableErosionIntVar.get()

    def getErosionAnchor(self):
        return (self.erosionAnchorX, self.erosionAnchorY)

    def getErosionIterations(self):
        return self.erosionIterations

    def getShapeOfErosionKernel(self):
        shape = self.erosionKernelShapeStringVar.get()
        if (shape == 'rectangle'):
            return cv.MORPH_RECT
        elif (shape == 'ellipse'):
            return cv.MORPH_ELLIPSE
        elif (shape == 'cross'):
            return cv.MORPH_CROSS

    def getSizeOfErosionKernel(self):
        return (self.erosionKernelWidth, self.erosionKernelHeight)

    def getErosionKernelAnchor(self):
        return (self.erosionKernelAnchorX, self.erosionKernelAnchorY)

    def getEnableDilationVal(self):
        return self.enableDilationIntVar.get()

    def getDilationAnchor(self):
        return (self.dilationAnchorX, self.dilationAnchorY)

    def getDilationIterations(self):
        return self.dilationIterations

    def getShapeOfDilationKernel(self):
        shape = self.dilationKernelShapeStringVar.get()
        if (shape == 'rectangle'):
            return cv.MORPH_RECT
        elif (shape == 'ellipse'):
            return cv.MORPH_ELLIPSE
        elif (shape == 'cross'):
            return cv.MORPH_CROSS

    def getSizeOfDilationKernel(self):
        return (self.dilationKernelWidth, self.dilationKernelHeight)

    def getDilationKernelAnchor(self):
        return (self.dilationKernelAnchorX, self.dilationKernelAnchorY)

    def updateCanvas(self, _img):
        self.cv_img = _img
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.imgHeight = self.img.height()
        self.imgWidth = self.img.width()
        self.canvas.config(width=self.imgWidth, height=self.imgHeight)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

class ContourFilteringFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.master = parent

        # filtering vals
        self.lowTargetArea = 0.0
        self.lowTargetFullness = 0.0
        self.lowAspectRatio = 0.0

        self.highTargetArea = 1.0
        self.highTargetFullness = 1.0
        self.highAspectRatio = 10.0

        self.frame_init()

    def frame_init(self):
        # First, setup the frame with the image
        self.imgFrame = tk.Frame(self)
        self.imgFrame.pack(side=tk.TOP)

        self.canvas = tk.Canvas(self.imgFrame)
        self.canvas.pack()

        self.cv_img = cv.cvtColor(np.zeros((640, 480, 3), np.uint8), cv.COLOR_BGR2RGB)
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        # Now, setup the container with all the contours filters
        self.filtersFrame = tk.Frame(self)
        self.filtersFrame.pack(side=tk.LEFT)

        # Setup the target area frame
        self.targetAreaFrame = tk.Frame(self.filtersFrame)
        self.targetAreaFrame.pack(fill=tk.X)

        # Setup the low target area frame
        self.lowTargetAreaFrame = tk.Frame(self.targetAreaFrame)
        self.lowTargetAreaFrame.pack()

        self.lowTargetAreaLabel = tk.Label(self.lowTargetAreaFrame, text='Low Target Area')
        self.lowTargetAreaLabel.pack(side=tk.LEFT)

        self.lowTargetAreaScaleFrame = tk.Frame(self.lowTargetAreaFrame)
        self.lowTargetAreaScaleFrame.pack(fill=tk.X)

        self.lowTargetAreaEntryStringVar = tk.StringVar()
        self.lowTargetAreaEntry = tk.Entry(self.lowTargetAreaScaleFrame, textvariable=self.lowTargetAreaEntryStringVar)
        self.lowTargetAreaEntry.bind('<Return>', self.lowTargetAreaEntryCallback)
        self.lowTargetAreaEntry.pack(side=tk.LEFT)

        self.lowTargetAreaScale = tk.Scale(self.lowTargetAreaScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=1.0, resolution=0.01, command=self.lowTargetAreaScaleCallback)
        self.lowTargetAreaScale.pack(fill=tk.X)

        self.setLowTargetArea(self.lowTargetArea)

        # Now set up the high target area frame
        self.highTargetAreaFrame = tk.Frame(self.targetAreaFrame)
        self.highTargetAreaFrame.pack(fill=tk.X)

        self.highTargetAreaLabel = tk.Label(self.highTargetAreaFrame, text='High Target Area')
        self.highTargetAreaLabel.pack(side=tk.LEFT)

        self.highTargetAreaScaleFrame = tk.Frame(self.highTargetAreaFrame)
        self.highTargetAreaScaleFrame.pack(fill=tk.X)

        self.highTargetAreaEntryStringVar = tk.StringVar()
        self.highTargetAreaEntry = tk.Entry(self.highTargetAreaScaleFrame, textvariable=self.highTargetAreaEntryStringVar)
        self.highTargetAreaEntry.bind('<Return>', self.highTargetAreaEntryCallback)
        self.highTargetAreaEntry.pack(side=tk.LEFT)

        self.highTargetAreaScale = tk.Scale(self.highTargetAreaScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=1.0, resolution=0.01, command=self.highTargetAreaScaleCallback)
        self.highTargetAreaScale.pack(fill=tk.X)

        self.setHighTargetArea(self.highTargetArea)

        # Setup the target fullness frame
        self.targetFullnessFrame = tk.Frame(self.filtersFrame)
        self.targetFullnessFrame.pack()

        # Setup the low target fullness frame
        self.lowTargetFullnessFrame = tk.Frame(self.targetFullnessFrame)
        self.lowTargetFullnessFrame.pack()

        self.lowTargetFullnessLabel = tk.Label(self.lowTargetFullnessFrame, text='Low Target Fullness')
        self.lowTargetFullnessLabel.pack(side=tk.LEFT)

        self.lowTargetFullnessScaleFrame = tk.Frame(self.lowTargetFullnessFrame)
        self.lowTargetFullnessScaleFrame.pack(fill=tk.X)

        self.lowTargetFullnessEntryStringVar = tk.StringVar()
        self.lowTargetFullnessEntry = tk.Entry(self.lowTargetFullnessScaleFrame, textvariable=self.lowTargetFullnessEntryStringVar)
        self.lowTargetFullnessEntry.bind('<Return>', self.lowTargetFullnessEntryCallback)
        self.lowTargetFullnessEntry.pack(side=tk.LEFT)

        self.lowTargetFullnessScale = tk.Scale(self.lowTargetFullnessScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=1.0, resolution=0.01, command=self.lowTargetFullnessScaleCallback)
        self.lowTargetFullnessScale.pack(fill=tk.X)

        self.setLowTargetFullness(self.lowTargetFullness)

        # Setup the high target fullness frame
        self.highTargetFullnessFrame = tk.Frame(self.targetFullnessFrame)
        self.highTargetFullnessFrame.pack()

        self.highTargetFullnessLabel = tk.Label(self.highTargetFullnessFrame, text='High Target Fullness')
        self.highTargetFullnessLabel.pack(side=tk.LEFT)

        self.highTargetFullnessScaleFrame = tk.Frame(self.highTargetFullnessFrame)
        self.highTargetFullnessScaleFrame.pack(fill=tk.X)

        self.highTargetFullnessEntryStringVar = tk.StringVar()
        self.highTargetFullnessEntry = tk.Entry(self.highTargetFullnessScaleFrame, textvariable=self.highTargetFullnessEntryStringVar)
        self.highTargetFullnessEntry.bind('<Return>', self.highTargetFullnessEntryCallback)
        self.highTargetFullnessEntry.pack(side=tk.LEFT)

        self.highTargetFullnessScale = tk.Scale(self.highTargetFullnessScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=1.0, resolution=0.01, command=self.highTargetFullnessScaleCallback)
        self.highTargetFullnessScale.pack(fill=tk.X)

        self.setHighTargetFullness(self.highTargetFullness)

        # Setup the aspect ratio frame
        self.aspectRatioFrame = tk.Frame(self.filtersFrame)
        self.aspectRatioFrame.pack()

        # Setup the low aspect ratio frame
        self.lowAspectRatioFrame = tk.Frame(self.aspectRatioFrame)
        self.lowAspectRatioFrame.pack()

        self.lowAspectRatioLabel = tk.Label(self.lowAspectRatioFrame, text='Low Aspect Ratio')
        self.lowAspectRatioLabel.pack(side=tk.LEFT)

        self.lowAspectRatioScaleFrame = tk.Frame(self.lowAspectRatioFrame)
        self.lowAspectRatioScaleFrame.pack(fill=tk.X)

        self.lowAspectRatioEntryStringVar = tk.StringVar()
        self.lowAspectRatioEntry = tk.Entry(self.lowAspectRatioScaleFrame, textvariable=self.lowAspectRatioEntryStringVar)
        self.lowAspectRatioEntry.bind('<Return>', self.lowAspectRatioEntryCallback)
        self.lowAspectRatioEntry.pack(side=tk.LEFT)

        self.lowAspectRatioScale = tk.Scale(self.lowAspectRatioScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=10.0, resolution=0.01, command=self.lowAspectRatioScaleCallback)
        self.lowAspectRatioScale.pack(fill=tk.X)

        self.setLowAspectRatio(self.lowAspectRatio)

        # Setup the high aspect ratio frame
        self.highAspectRatioFrame = tk.Frame(self.aspectRatioFrame)
        self.highAspectRatioFrame.pack()

        self.highAspectRatioLabel = tk.Label(self.highAspectRatioFrame, text='High Aspect Ratio')
        self.highAspectRatioLabel.pack(side=tk.LEFT)

        self.highAspectRatioScaleFrame = tk.Frame(self.highAspectRatioFrame)
        self.highAspectRatioScaleFrame.pack(fill=tk.X)

        self.highAspectRatioEntryStringVar = tk.StringVar()
        self.highAspectRatioEntry = tk.Entry(self.highAspectRatioScaleFrame, textvariable=self.highAspectRatioEntryStringVar)
        self.highAspectRatioEntry.bind('<Return>', self.highAspectRatioEntryCallback)
        self.highAspectRatioEntry.pack(side=tk.LEFT)

        self.highAspectRatioScale = tk.Scale(self.highAspectRatioScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=10.0, resolution=0.01, command=self.highAspectRatioScaleCallback)
        self.highAspectRatioScale.pack(side=tk.LEFT)

        self.setHighAspectRatio(self.highAspectRatio)

        # Setup contour sorting/grouping frame
        self.contourSortingFrame = tk.Frame(self)
        self.contourSortingFrame.pack(side=tk.LEFT)

        # Setup the sorting mode frame
        self.sortingModeFrame = tk.Frame(self.contourSortingFrame)
        self.sortingModeFrame.pack()

        self.sortingModeLabel = tk.Label(self.sortingModeFrame, text='Sorting Mode')
        self.sortingModeLabel.pack(side=tk.TOP)

        self.sortingMode = tk.StringVar()
        self.sortingMode.set('center')

        self.leftRadioButton = tk.Radiobutton(self.sortingModeFrame, text='Left to Right', variable=self.sortingMode, value='left')
        self.leftRadioButton.pack(side=tk.BOTTOM)
        
        self.rightRadioButton = tk.Radiobutton(self.sortingModeFrame, text='Right to Left', variable=self.sortingMode, value='right')
        self.rightRadioButton.pack(side=tk.BOTTOM)
        
        self.centerRadioButton = tk.Radiobutton(self.sortingModeFrame, text='Center Outwards', variable=self.sortingMode, value='center')
        self.centerRadioButton.pack(side=tk.BOTTOM)
        
        self.topRadioButton = tk.Radiobutton(self.sortingModeFrame, text='Top to Bottom', variable=self.sortingMode, value='top')
        self.topRadioButton.pack(side=tk.BOTTOM)
        
        self.bottomRadioButton = tk.Radiobutton(self.sortingModeFrame, text='Bottom to Top', variable=self.sortingMode, value='bottom')
        self.bottomRadioButton.pack(side=tk.BOTTOM)

    def lowTargetAreaEntryCallback(self, event):
        print('low target area entry callback triggered!')
        try:
            self.lowTargetArea = float(event.widget.get())
            self.lowTargetAreaScale.set(self.lowTargetArea)
        except:
            print('invalid entry!')
        return True

    def lowTargetFullnessEntryCallback(self, event):
        print('low target fullness entry callback triggered!')
        try:
            self.lowTargetFullness = float(event.widget.get())
            self.lowTargetFullnessScale.set(self.lowTargetFullness)
        except:
            print('invalid entry!')
        return True

    def lowAspectRatioEntryCallback(self, event):
        print('low aspect ratio entry callback triggered!')
        try:
            self.lowAspectRatio = float(event.widget.get())
            self.lowAspectRatioScale.set(self.lowAspectRatio)
        except:
            print('invalid entry!')
        return True

    def highTargetAreaEntryCallback(self, event):
        print('high target area entry callback triggered!')
        try:
            self.highTargetArea = float(event.widget.get())
            self.highTargetAreaScale.set(self.highTargetArea)
        except:
            print('invalid entry!')
        return True

    def highTargetFullnessEntryCallback(self, event):
        print('high target fullness entry callback triggered!')
        try:
            self.highTargetFullness = float(event.widget.get())
            self.highTargetFullnessScale.set(self.highTargetFullness)
        except:
            print('invalid entry!')
        return True

    def highAspectRatioEntryCallback(self, event):
        print('high aspect ratio entry callback triggered!')
        try:
            self.highAspectRatio = float(event.widget.get())
            self.highAspectRatioScale.set(self.highAspectRatio)
        except:
            print('invalid entry!')
        return True

    def lowTargetAreaScaleCallback(self, value=None):
        print('low target area scale callback triggered!')
        self.lowTargetArea = self.lowTargetAreaScale.get()
        self.lowTargetAreaEntryStringVar.set(str(self.lowTargetArea))

    def lowTargetFullnessScaleCallback(self, value=None):
        print('low target fullness scale callback triggered!')
        self.lowTargetFullness = self.lowTargetFullnessScale.get()
        self.lowTargetFullnessEntryStringVar.set(str(self.lowTargetFullness))

    def lowAspectRatioScaleCallback(self, value=None):
        print('low aspect ratio scale callback triggered!')
        self.lowAspectRatio = self.lowAspectRatioScale.get()
        self.lowAspectRatioEntryStringVar.set(str(self.lowAspectRatio))

    def highTargetAreaScaleCallback(self, value=None):
        print('high target area scale callback triggered!')
        self.highTargetArea = self.highTargetAreaScale.get()
        self.highTargetAreaEntryStringVar.set(str(self.highTargetArea))

    def highTargetFullnessScaleCallback(self, value=None):
        print('high target fullness scale callback triggered!')
        self.highTargetFullness = self.highTargetFullnessScale.get()
        self.highTargetFullnessEntryStringVar.set(str(self.highTargetFullness))

    def highAspectRatioScaleCallback(self, value=None):
        print('high aspect ratio scale callback triggered!')
        self.highAspectRatio = self.highAspectRatioScale.get()
        self.highAspectRatioEntryStringVar.set(str(self.highAspectRatio))

    def setLowTargetArea(self, val):
        self.lowTargetArea = val
        self.lowTargetAreaEntryStringVar.set(str(self.lowTargetArea))
        self.lowTargetAreaScale.set(self.lowTargetArea)

    def setLowTargetFullness(self, val):
        self.lowTargetFullness = val
        self.lowTargetFullnessEntryStringVar.set(str(self.lowTargetFullness))
        self.lowTargetFullnessScale.set(self.lowTargetFullness)

    def setLowAspectRatio(self, val):
        self.lowAspectRatio = val
        self.lowAspectRatioEntryStringVar.set(str(self.lowAspectRatio))
        self.lowAspectRatioScale.set(self.lowAspectRatio)

    def setHighTargetArea(self, val):
        self.highTargetArea = val
        self.highTargetAreaEntryStringVar.set(str(self.highTargetArea))
        self.highTargetAreaScale.set(self.highTargetArea)
    
    def setHighTargetFullness(self, val):
        self.highTargetFullness = val
        self.highTargetFullnessEntryStringVar.set(str(self.highTargetFullness))
        self.highTargetFullnessScale.set(self.highTargetFullness)

    def setHighAspectRatio(self, val):
        self.highAspectRatio = val
        self.highAspectRatioEntryStringVar.set(str(self.highAspectRatio))
        self.highAspectRatioScale.set(self.highAspectRatio)

    def updateCanvas(self, _img):
        self.cv_img = _img
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.imgHeight = self.img.height()
        self.imgWidth = self.img.width()
        self.canvas.config(width=self.imgWidth, height=self.imgHeight)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

    def getTargetAreaRange(self):
        return (self.lowTargetArea, self.highTargetArea)
    
    def getTargetFullnessRange(self):
        return (self.lowTargetFullness, self.highTargetFullness)

    def getAspectRatioRange(self):
        return (self.lowAspectRatio, self.highAspectRatio)

    def getSortingMode(self):
        return self.sortingMode.get()

class ContourPairingFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.master = parent

        # filtering vals
        self.lowTargetArea = 0.0
        self.lowTargetFullness = 0.0
        self.lowAspectRatio = 0.0

        self.highTargetArea = 1.0
        self.highTargetFullness = 1.0
        self.highAspectRatio = 10.0

        self.frame_init()

    def frame_init(self):
        # First, setup the frame with the image
        self.imgFrame = tk.Frame(self)
        self.imgFrame.pack(side=tk.TOP)

        self.canvas = tk.Canvas(self.imgFrame)
        self.canvas.pack()

        self.cv_img = cv.cvtColor(np.zeros((640, 480, 3), np.uint8), cv.COLOR_BGR2RGB)
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        # Setup contour grouping frame
        self.contourPairingFrame = tk.Frame(self)
        self.contourPairingFrame.pack(side=tk.LEFT)

        # First, setup the checkbox which'll ask whther or not we want to pair contours
        self.enableContourPairingFrame = tk.Frame(self.contourPairingFrame)
        self.enableContourPairingFrame.pack(fill=tk.X)

        self.enableContourPairingIntVar = tk.IntVar()
        self.enableContourPairingIntVar.set(0)
        self.enableContourPairingCheckButton = tk.Checkbutton(self.enableContourPairingFrame, text='Pair Contours', variable=self.enableContourPairingIntVar, onvalue=1, offvalue=0)
        self.enableContourPairingCheckButton.pack(side=tk.LEFT)

        # Now we setup the frame to choose where the contours will intersect
        self.intersectionFrame = tk.Frame(self.contourPairingFrame)
        self.intersectionFrame.pack()

        self.intersectionLabel = tk.Label(self.intersectionFrame, text='Contour Intersection Location')
        self.intersectionLabel.pack(side=tk.LEFT)

        self.intersectionStringVar = tk.StringVar()
        self.intersectionStringVar.set('neither')

        self.neitherRadioButton = tk.Radiobutton(self.intersectionFrame, text='Neither', variable=self.intersectionStringVar, value='neither')
        self.neitherRadioButton.pack(side=tk.LEFT)

        self.intersectAboveRadioButton = tk.Radiobutton(self.intersectionFrame, text='Above', variable=self.intersectionStringVar, value='above')
        self.intersectAboveRadioButton.pack(side=tk.LEFT)
        
        self.intersectBelowRadioButton = tk.Radiobutton(self.intersectionFrame, text='Below', variable=self.intersectionStringVar, value='below')
        self.intersectBelowRadioButton.pack(side=tk.LEFT)

        self.intersectRightRadioButton = tk.Radiobutton(self.intersectionFrame, text='Right', variable=self.intersectionStringVar, value='right')
        self.intersectRightRadioButton.pack(side=tk.LEFT)

        self.intersectLeftRadioButton = tk.Radiobutton(self.intersectionFrame, text='Left', variable=self.intersectionStringVar, value='left')
        self.intersectLeftRadioButton.pack(side=tk.LEFT)

        # Setup the sorting mode frame for the target
        self.targetSortingModeFrame = tk.Frame(self.contourPairingFrame)
        self.targetSortingModeFrame.pack()

        self.targetSortingModeLabel = tk.Label(self.targetSortingModeFrame, text='Sorting Mode')
        self.targetSortingModeLabel.pack(side=tk.TOP)

        self.targetSortingMode = tk.StringVar()
        self.targetSortingMode.set('center')

        self.targetLeftRadioButton = tk.Radiobutton(self.targetSortingModeFrame, text='Left to Right', variable=self.targetSortingMode, value='left')
        self.targetLeftRadioButton.pack(side=tk.BOTTOM)
        
        self.targetRightRadioButton = tk.Radiobutton(self.targetSortingModeFrame, text='Right to Left', variable=self.targetSortingMode, value='right')
        self.targetRightRadioButton.pack(side=tk.BOTTOM)
        
        self.targetCenterRadioButton = tk.Radiobutton(self.targetSortingModeFrame, text='Center Outwards', variable=self.targetSortingMode, value='center')
        self.targetCenterRadioButton.pack(side=tk.BOTTOM)
        
        self.targetTopRadioButton = tk.Radiobutton(self.targetSortingModeFrame, text='Top to Bottom', variable=self.targetSortingMode, value='top')
        self.targetTopRadioButton.pack(side=tk.BOTTOM)
        
        self.targetBottomRadioButton = tk.Radiobutton(self.targetSortingModeFrame, text='Bottom to Top', variable=self.targetSortingMode, value='bottom')
        self.targetBottomRadioButton.pack(side=tk.BOTTOM)

        # Now we setup the filtering frame
        self.filtersFrame = tk.Frame(self)
        self.filtersFrame.pack(side=tk.LEFT)

        # Setup the target area frame
        self.targetAreaFrame = tk.Frame(self.filtersFrame)
        self.targetAreaFrame.pack(fill=tk.X)

        # Setup the low target area frame
        self.lowTargetAreaFrame = tk.Frame(self.targetAreaFrame)
        self.lowTargetAreaFrame.pack()

        self.lowTargetAreaLabel = tk.Label(self.lowTargetAreaFrame, text='Low Target Area')
        self.lowTargetAreaLabel.pack(side=tk.LEFT)

        self.lowTargetAreaScaleFrame = tk.Frame(self.lowTargetAreaFrame)
        self.lowTargetAreaScaleFrame.pack(fill=tk.X)

        self.lowTargetAreaEntryStringVar = tk.StringVar()
        self.lowTargetAreaEntry = tk.Entry(self.lowTargetAreaScaleFrame, textvariable=self.lowTargetAreaEntryStringVar)
        self.lowTargetAreaEntry.bind('<Return>', self.lowTargetAreaEntryCallback)
        self.lowTargetAreaEntry.pack(side=tk.LEFT)

        self.lowTargetAreaScale = tk.Scale(self.lowTargetAreaScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=1.0, resolution=0.01, command=self.lowTargetAreaScaleCallback)
        self.lowTargetAreaScale.pack(fill=tk.X)

        self.setLowTargetArea(self.lowTargetArea)

        # Now set up the high target area frame
        self.highTargetAreaFrame = tk.Frame(self.targetAreaFrame)
        self.highTargetAreaFrame.pack(fill=tk.X)

        self.highTargetAreaLabel = tk.Label(self.highTargetAreaFrame, text='High Target Area')
        self.highTargetAreaLabel.pack(side=tk.LEFT)

        self.highTargetAreaScaleFrame = tk.Frame(self.highTargetAreaFrame)
        self.highTargetAreaScaleFrame.pack(fill=tk.X)

        self.highTargetAreaEntryStringVar = tk.StringVar()
        self.highTargetAreaEntry = tk.Entry(self.highTargetAreaScaleFrame, textvariable=self.highTargetAreaEntryStringVar)
        self.highTargetAreaEntry.bind('<Return>', self.highTargetAreaEntryCallback)
        self.highTargetAreaEntry.pack(side=tk.LEFT)

        self.highTargetAreaScale = tk.Scale(self.highTargetAreaScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=1.0, resolution=0.01, command=self.highTargetAreaScaleCallback)
        self.highTargetAreaScale.pack(fill=tk.X)

        self.setHighTargetArea(self.highTargetArea)

        # Setup the target fullness frame
        self.targetFullnessFrame = tk.Frame(self.filtersFrame)
        self.targetFullnessFrame.pack()

        # Setup the low target fullness frame
        self.lowTargetFullnessFrame = tk.Frame(self.targetFullnessFrame)
        self.lowTargetFullnessFrame.pack()

        self.lowTargetFullnessLabel = tk.Label(self.lowTargetFullnessFrame, text='Low Target Fullness')
        self.lowTargetFullnessLabel.pack(side=tk.LEFT)

        self.lowTargetFullnessScaleFrame = tk.Frame(self.lowTargetFullnessFrame)
        self.lowTargetFullnessScaleFrame.pack(fill=tk.X)

        self.lowTargetFullnessEntryStringVar = tk.StringVar()
        self.lowTargetFullnessEntry = tk.Entry(self.lowTargetFullnessScaleFrame, textvariable=self.lowTargetFullnessEntryStringVar)
        self.lowTargetFullnessEntry.bind('<Return>', self.lowTargetFullnessEntryCallback)
        self.lowTargetFullnessEntry.pack(side=tk.LEFT)

        self.lowTargetFullnessScale = tk.Scale(self.lowTargetFullnessScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=1.0, resolution=0.01, command=self.lowTargetFullnessScaleCallback)
        self.lowTargetFullnessScale.pack(fill=tk.X)

        self.setLowTargetFullness(self.lowTargetFullness)

        # Setup the high target fullness frame
        self.highTargetFullnessFrame = tk.Frame(self.targetFullnessFrame)
        self.highTargetFullnessFrame.pack()

        self.highTargetFullnessLabel = tk.Label(self.highTargetFullnessFrame, text='High Target Fullness')
        self.highTargetFullnessLabel.pack(side=tk.LEFT)

        self.highTargetFullnessScaleFrame = tk.Frame(self.highTargetFullnessFrame)
        self.highTargetFullnessScaleFrame.pack(fill=tk.X)

        self.highTargetFullnessEntryStringVar = tk.StringVar()
        self.highTargetFullnessEntry = tk.Entry(self.highTargetFullnessScaleFrame, textvariable=self.highTargetFullnessEntryStringVar)
        self.highTargetFullnessEntry.bind('<Return>', self.highTargetFullnessEntryCallback)
        self.highTargetFullnessEntry.pack(side=tk.LEFT)

        self.highTargetFullnessScale = tk.Scale(self.highTargetFullnessScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=1.0, resolution=0.01, command=self.highTargetFullnessScaleCallback)
        self.highTargetFullnessScale.pack(fill=tk.X)

        self.setHighTargetFullness(self.highTargetFullness)

        # Setup the aspect ratio frame
        self.aspectRatioFrame = tk.Frame(self.filtersFrame)
        self.aspectRatioFrame.pack()

        # Setup the low aspect ratio frame
        self.lowAspectRatioFrame = tk.Frame(self.aspectRatioFrame)
        self.lowAspectRatioFrame.pack()

        self.lowAspectRatioLabel = tk.Label(self.lowAspectRatioFrame, text='Low Aspect Ratio')
        self.lowAspectRatioLabel.pack(side=tk.LEFT)

        self.lowAspectRatioScaleFrame = tk.Frame(self.lowAspectRatioFrame)
        self.lowAspectRatioScaleFrame.pack(fill=tk.X)

        self.lowAspectRatioEntryStringVar = tk.StringVar()
        self.lowAspectRatioEntry = tk.Entry(self.lowAspectRatioScaleFrame, textvariable=self.lowAspectRatioEntryStringVar)
        self.lowAspectRatioEntry.bind('<Return>', self.lowAspectRatioEntryCallback)
        self.lowAspectRatioEntry.pack(side=tk.LEFT)

        self.lowAspectRatioScale = tk.Scale(self.lowAspectRatioScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=10.0, resolution=0.01, command=self.lowAspectRatioScaleCallback)
        self.lowAspectRatioScale.pack(fill=tk.X)

        self.setLowAspectRatio(self.lowAspectRatio)

        # Setup the high aspect ratio frame
        self.highAspectRatioFrame = tk.Frame(self.aspectRatioFrame)
        self.highAspectRatioFrame.pack()

        self.highAspectRatioLabel = tk.Label(self.highAspectRatioFrame, text='High Aspect Ratio')
        self.highAspectRatioLabel.pack(side=tk.LEFT)

        self.highAspectRatioScaleFrame = tk.Frame(self.highAspectRatioFrame)
        self.highAspectRatioScaleFrame.pack(fill=tk.X)

        self.highAspectRatioEntryStringVar = tk.StringVar()
        self.highAspectRatioEntry = tk.Entry(self.highAspectRatioScaleFrame, textvariable=self.highAspectRatioEntryStringVar)
        self.highAspectRatioEntry.bind('<Return>', self.highAspectRatioEntryCallback)
        self.highAspectRatioEntry.pack(side=tk.LEFT)

        self.highAspectRatioScale = tk.Scale(self.highAspectRatioScaleFrame, orient=tk.HORIZONTAL, from_=0.0, to=10.0, resolution=0.01, command=self.highAspectRatioScaleCallback)
        self.highAspectRatioScale.pack(side=tk.LEFT)

        self.setHighAspectRatio(self.highAspectRatio)

    def lowTargetAreaEntryCallback(self, event):
        print('low target area entry callback triggered!')
        try:
            self.lowTargetArea = float(event.widget.get())
            self.lowTargetAreaScale.set(self.lowTargetArea)
        except:
            print('invalid entry!')
        return True

    def lowTargetFullnessEntryCallback(self, event):
        print('low target fullness entry callback triggered!')
        try:
            self.lowTargetFullness = float(event.widget.get())
            self.lowTargetFullnessScale.set(self.lowTargetFullness)
        except:
            print('invalid entry!')
        return True

    def lowAspectRatioEntryCallback(self, event):
        print('low aspect ratio entry callback triggered!')
        try:
            self.lowAspectRatio = float(event.widget.get())
            self.lowAspectRatioScale.set(self.lowAspectRatio)
        except:
            print('invalid entry!')
        return True

    def highTargetAreaEntryCallback(self, event):
        print('high target area entry callback triggered!')
        try:
            self.highTargetArea = float(event.widget.get())
            self.highTargetAreaScale.set(self.highTargetArea)
        except:
            print('invalid entry!')
        return True

    def highTargetFullnessEntryCallback(self, event):
        print('high target fullness entry callback triggered!')
        try:
            self.highTargetFullness = float(event.widget.get())
            self.highTargetFullnessScale.set(self.highTargetFullness)
        except:
            print('invalid entry!')
        return True

    def highAspectRatioEntryCallback(self, event):
        print('high aspect ratio entry callback triggered!')
        try:
            self.highAspectRatio = float(event.widget.get())
            self.highAspectRatioScale.set(self.highAspectRatio)
        except:
            print('invalid entry!')
        return True

    def lowTargetAreaScaleCallback(self, value=None):
        print('low target area scale callback triggered!')
        self.lowTargetArea = self.lowTargetAreaScale.get()
        self.lowTargetAreaEntryStringVar.set(str(self.lowTargetArea))

    def lowTargetFullnessScaleCallback(self, value=None):
        print('low target fullness scale callback triggered!')
        self.lowTargetFullness = self.lowTargetFullnessScale.get()
        self.lowTargetFullnessEntryStringVar.set(str(self.lowTargetFullness))

    def lowAspectRatioScaleCallback(self, value=None):
        print('low aspect ratio scale callback triggered!')
        self.lowAspectRatio = self.lowAspectRatioScale.get()
        self.lowAspectRatioEntryStringVar.set(str(self.lowAspectRatio))

    def highTargetAreaScaleCallback(self, value=None):
        print('high target area scale callback triggered!')
        self.highTargetArea = self.highTargetAreaScale.get()
        self.highTargetAreaEntryStringVar.set(str(self.highTargetArea))

    def highTargetFullnessScaleCallback(self, value=None):
        print('high target fullness scale callback triggered!')
        self.highTargetFullness = self.highTargetFullnessScale.get()
        self.highTargetFullnessEntryStringVar.set(str(self.highTargetFullness))

    def highAspectRatioScaleCallback(self, value=None):
        print('high aspect ratio scale callback triggered!')
        self.highAspectRatio = self.highAspectRatioScale.get()
        self.highAspectRatioEntryStringVar.set(str(self.highAspectRatio))

    def setLowTargetArea(self, val):
        self.lowTargetArea = val
        self.lowTargetAreaEntryStringVar.set(str(self.lowTargetArea))
        self.lowTargetAreaScale.set(self.lowTargetArea)

    def setLowTargetFullness(self, val):
        self.lowTargetFullness = val
        self.lowTargetFullnessEntryStringVar.set(str(self.lowTargetFullness))
        self.lowTargetFullnessScale.set(self.lowTargetFullness)

    def setLowAspectRatio(self, val):
        self.lowAspectRatio = val
        self.lowAspectRatioEntryStringVar.set(str(self.lowAspectRatio))
        self.lowAspectRatioScale.set(self.lowAspectRatio)

    def setHighTargetArea(self, val):
        self.highTargetArea = val
        self.highTargetAreaEntryStringVar.set(str(self.highTargetArea))
        self.highTargetAreaScale.set(self.highTargetArea)
    
    def setHighTargetFullness(self, val):
        self.highTargetFullness = val
        self.highTargetFullnessEntryStringVar.set(str(self.highTargetFullness))
        self.highTargetFullnessScale.set(self.highTargetFullness)

    def setHighAspectRatio(self, val):
        self.highAspectRatio = val
        self.highAspectRatioEntryStringVar.set(str(self.highAspectRatio))
        self.highAspectRatioScale.set(self.highAspectRatio)

    def updateCanvas(self, _img):
        self.cv_img = _img
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.imgHeight = self.img.height()
        self.imgWidth = self.img.width()
        self.canvas.config(width=self.imgWidth, height=self.imgHeight)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

    def getEnableContourPairing(self):
        return self.enableContourPairingIntVar.get()
    
    def getIntersectionLocation(self):
        return self.intersectionStringVar.get()

    def getSortingMode(self):
        return self.targetSortingMode.get()

    def getTargetAreaRange(self):
        return (self.lowTargetArea, self.highTargetArea)
    
    def getTargetFullnessRange(self):
        return (self.lowTargetFullness, self.highTargetFullness)

    def getAspectRatioRange(self):
        return (self.lowAspectRatio, self.highAspectRatio)

class PoseEstimationFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.master = parent

        self.range = [0, 0]

        self.frame_init()

    def frame_init(self):
        # First, setup the frame with the image
        self.imgFrame = tk.Frame(self)
        self.imgFrame.pack(side=tk.TOP)

        self.canvas = tk.Canvas(self.imgFrame)
        self.canvas.pack()

        self.cv_img = cv.cvtColor(np.zeros((640, 480, 3), np.uint8), cv.COLOR_BGR2RGB)
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        # Now set up the container frame
        self.poseEstimationSetupFrame = tk.Frame(self)
        self.poseEstimationSetupFrame.pack(side=tk.LEFT)

        # First, setup the checkbox which asks whether or not we want to perform pose estimation or not
        self.enablePoseEstimationFrame = tk.Frame(self.poseEstimationSetupFrame)
        self.enablePoseEstimationFrame.pack(fill=tk.X)

        self.enablePoseEstimationIntVar = tk.IntVar()
        self.enablePoseEstimationIntVar.set(0)
        self.enablePoseEstimationCheckButton = tk.Checkbutton(self.enablePoseEstimationFrame, text='Perform Pose Estimation', variable=self.enablePoseEstimationIntVar, onvalue=1, offvalue=0)
        self.enablePoseEstimationCheckButton.pack(side=tk.LEFT)

        # Setup the frame where we can specify the range of points to use
        self.rangeFrame = tk.Frame(self.poseEstimationSetupFrame)
        self.rangeFrame.pack(side=tk.LEFT)

        # First, setup the checkbox which asks whether or not we want to perform pose estimation or not
        self.enablePoseEstimationRangeFrame = tk.Frame(self.rangeFrame)
        self.enablePoseEstimationRangeFrame.pack(fill=tk.X)

        self.enablePoseEstimationRangeIntVar = tk.IntVar()
        self.enablePoseEstimationRangeIntVar.set(0)
        self.enablePoseEstimationRangeCheckButton = tk.Checkbutton(self.enablePoseEstimationRangeFrame, text='Range', variable=self.enablePoseEstimationRangeIntVar, onvalue=1, offvalue=0)
        self.enablePoseEstimationRangeCheckButton.pack(side=tk.LEFT)

        # Setup the frame to specify the range
        self.rangeSpecifierFrame = tk.Frame(self.poseEstimationSetupFrame)
        self.rangeSpecifierFrame.pack(fill=tk.X)

        # From property
        self.fromPropertyFrame = tk.Frame(self.rangeSpecifierFrame)
        self.fromPropertyFrame.pack()

        self.fromPropertyLabel = tk.Label(self.fromPropertyFrame, text='From')
        self.fromPropertyLabel.pack(side=tk.LEFT)

        self.fromPropertyEntryStringVar = tk.StringVar()
        self.fromPropertyEntry = tk.Entry(self.fromPropertyFrame, textvariable=self.fromPropertyEntryStringVar)
        self.fromPropertyEntry.bind('<Return>', self.fromPropertyEntryCallback)
        self.fromPropertyEntry.pack(side=tk.RIGHT)

        self.setFromProperty(0)

        # To property
        self.toPropertyFrame = tk.Frame(self.rangeSpecifierFrame)
        self.toPropertyFrame.pack()

        self.toPropertyLabel = tk.Label(self.toPropertyFrame, text='To')
        self.toPropertyLabel.pack(side=tk.LEFT)

        self.toPropertyEntryStringVar = tk.StringVar()
        self.toPropertyEntry = tk.Entry(self.toPropertyFrame, textvariable=self.toPropertyEntryStringVar)
        self.toPropertyEntry.bind('<Return>', self.toPropertyEntryCallback)
        self.toPropertyEntry.pack(side=tk.RIGHT)

        self.setToProperty(0)

    def fromPropertyEntryCallback(self, event):
        print('from property entry callback triggered!')
        try:
            self.range[0] = int(event.widget.get())
        except:
            print('invalid entry!')
        return True

    def toPropertyEntryCallback(self, event):
        print('to property entry callback triggered!')
        try:
            self.range[1] = int(event.widget.get())
        except:
            print('invalid entry!')
        return True

    def setFromProperty(self, val):
        self.range[0] = val
        self.fromPropertyEntryStringVar.set(str(self.range[0]))

    def setToProperty(self, val):
        self.range[1] = val
        self.toPropertyEntryStringVar.set(str(self.range[1]))

    def updateCanvas(self, _img):
        self.cv_img = _img
        self.img = ImageTk.PhotoImage(image = Image.fromarray(self.cv_img))
        self.imgHeight = self.img.height()
        self.imgWidth = self.img.width()
        self.canvas.config(width=self.imgWidth, height=self.imgHeight)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

    def getEnablePoseEstimation(self):
        return self.enablePoseEstimationIntVar.get()

    def getEnablePoseEstimationRange(self):
        return self.enablePoseEstimationRangeIntVar.get()

    def getRange(self):
        return self.range

if __name__ == "__main__":
    app = Application()
    app.mainloop()