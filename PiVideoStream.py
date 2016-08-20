#Copied from http://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

#import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import cv2
##import RPi.GPIO as gp
##
##camNum = 1

class PiVideoStream:
    def __init__(self, resolution=(640, 480), framerate=32):
        #initialize the camera and stream
##        gp.setwarnings(False)
##        gp.setmode(gp.BOARD)
##
##        gp.setup(7, gp.OUT)
##        gp.setup(11, gp.OUT)
##        gp.setup(12, gp.OUT)
##
##        gp.output(7, False)
##        gp.output(11, False)
##        gp.output(12, True)
##        camNum = 1

        self.frame = None
        self.stopped = False
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)

        #initialize the frame and the variable used to indicate
        #if the thread should be stopped
        

##    def toggle_cam(self):
##        global camNum
##        gp.setmode(gp.BOARD)
##        if camNum == 1:
##            camNum = 2
##            gp.output(7, True)
##            gp.output(11, False)
##            gp.output(12, True)        
##
##        elif camNum == 2:
##            camNum = 1
##            gp.output(7, False)
##            gp.output(11, False)
##            gp.output(12, True)
    
    def start(self):
        #start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        #keep looping infinitely until the thread is stopped
        for f in self.stream:
            #grab the frame from the stream and clear the stream in
            #preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
            #self.toggle_cam()

            #if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        #return the frame most recently 
        #self.toggle_cam()
        return self.frame

    def stop(self):
        #indicate that the thread should be stopped
        self.stopped = True
