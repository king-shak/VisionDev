import io
import time
import threading
import picamera
import cv2 as cv
import numpy as np
import queue

exp = None
event = threading.Event()
num = 0
numM = 0

class Pipeline(threading.Thread):
    def __init__(self, owner):
        super(Pipeline, self).__init__()
        # Reference back to the owner for getting frames
        self.owner = owner
        # Start the thread
        self.start()

    def run(self):
        # This method runs in a separate thread
        while True:
            frame = self.owner.getFrame()
            # If it is equal to none, the encoder hasn't processed it's first frame yet
            if (frame != None):
                print('got a frame')
                global exp
                exp = frame
                # Just see if we can do anything with it
                frame_threshold = cv.inRange(frame, (32, 71, 56), (153, 201, 198))
                # Save the frame for later analysis
                cv.imwrite('experiment.jpg', frame)
                cv.imwrite('experiment_threshold.jpg', frame_threshold)
                event.set()
                break

class ProcessOutput(object):
    def __init__(self):
        # Lock for accessing the frame property
        self.lock = threading.Lock()
        # The most recent frame
        self.frame = None
        # Create an instance of the pipeline
        self.pipeline = Pipeline(self)
        # Buffer for storing raw image data
        self.buffer = io.BytesIO()

    def write(self, data):
        # Check if we have a complete frame
        # start = time.time()
        if (len(data) + self.buffer.getbuffer().nbytes >= 640 * 480 * 3):
            # end = time.time()
            global num
            num+=1
            # print('Check frame')
            # print((end - start) * 1000)
            # start = time.time()
            # Get the number of bytes needed to complete the frame
            numOfBytesLeft = (640 * 480 * 3) - self.buffer.getbuffer().nbytes

            # Grab the remaining bytes
            remainingBytes = data[:numOfBytesLeft]
            # end = time.time()
            # print('get remaining bytes')
            # print((end - start) * 1000)

            # Write it to the buffer, seek to the beginning of the buffer
            # start = time.time()
            self.buffer.write(remainingBytes)
            self.buffer.seek(0, 0)
            # end = time.time()
            # print('write and seek to beginning')
            # print((end - start) * 1000)

            # Read the bytes from the buffer and parse it into a numpy array
            # start = time.time()
            img = np.frombuffer(self.buffer.read(640 * 480 * 3), dtype=np.uint8)
            # Call reshape to put the pixels in the correct order
            img = img.reshape((480, 640, 3))
            # end = time.time()
            # print('process data to frame')
            # print((end - start) * 1000)

            # Acquire the lock to update the frame
            # start = time.time()
            with self.lock:
                self.frame = img
            # end = time.time()
            # print('update class')
            # print((end - start) * 1000)

            # Clear the buffer, and write any remaining bytes (these are part of the next frame)
            # start = time.time()
            self.buffer = io.BytesIO()
            self.buffer.write(data[numOfBytesLeft:])
            # end = time.time()
            # print('clear buffer and write remaining bytes')
            # print((end - start) * 1000)
            # print('\n\n')
        else:
            global numM
            numM+=1
            self.buffer.write(data)

    def getFrame(self):
        with self.lock:
            return self.frame

    #def flush(self):
    #    self.pipeline.join()

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 90
    time.sleep(2)
    event.clear()
    output = ProcessOutput()
    camera.start_recording(output, format='bgr')
    # print('started recording!')
    # Just give us one frame to see if we can do that much correctly
    camera.wait_recording(1)
    camera.stop_recording()

    print(num)
    print(numM)

    # Wait for the encoder to finish
    event.wait()

    control = np.empty((480 * 640 * 3,), dtype=np.uint8)
    camera.capture(control, 'bgr')
    control = control.reshape((480, 640, 3))
    cv.inRange(control, (32, 71, 56), (153, 201, 198))
    cv.imwrite('control.jpg', control)
    cv.imwrite('control_threshold.jpg')

    # Compare the control and the experiment
    badPixels = 0
    for row in range(480):
        for column in range(640):
            for byte in range(3):
                if (control[row][column][byte] != exp[row][column][byte]):
                    badPixels+=1

    print(badPixels)