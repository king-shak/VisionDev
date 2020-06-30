import io
import time
import threading
import picamera
import cv2 as cv
import numpy as np
import queue

resolution = (640, 480)
fps = 30

class Pipeline(threading.Thread):
    def __init__(self, owner):
        super(Pipeline, self).__init__()
        self.owner = owner
        self.terminated = False
        self.event = threading.Event()
        self.buffer = io.BytesIO()
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            if self.event.wait(1):
                try:
                    self.buffer.seek(0, 0)
                    frame = np.frombuffer(self.buffer.read(resolution[0] * resolution[1] * 3), dtype=np.uint8)
                    frame = frame.reshape((resolution[1], resolution[0], 3))
                    # continue with the pipeline...
                    cv.imwrite('stream.jpg', frame)
                finally:
                    # Clear the buffer
                    self.buffer.seek(0, 0)
                    self.buffer.truncate(0)
                    self.event.clear()

class ProcessOutput(object):
    def __init__(self):
        # Create an instance of the pipeline
        self.pipeline = Pipeline(self)
        self.numFramesCaptured = 0

    def write(self, data):
        if not self.pipeline.event.is_set():
            self.pipeline.buffer.write(data)
            self.pipeline.event.set()
            self.numFramesCaptured+=1
        # else we drop the frame

    def flush(self):
        self.pipeline.terminated = True
        self.pipeline.join()

with picamera.PiCamera(resolution=resolution, framerate=fps) as cam:
    # Let the camera warm up
    cam.start_preview()
    time.sleep(2)
    output = ProcessOutput()
    cam.start_recording(output, format='bgr')
    cam.wait_recording(1)
    cam.stop_recording()
    cam.stop_preview()
    # This should be (at least close to) the fps we specified
    print(output.numFramesCaptured)