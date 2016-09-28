import cv2
import numpy as np
from multiprocessing import Process
from multiprocessing import Pipe
from multiprocessing import Event
from picamera import PiCamera
from picamera.array import PiRGBArray
import RPi.GPIO as gp

# gp.setwarnings(False)
# gp.setmode(gp.BOARD)
#
# gp.setup(7, gp.OUT)
# gp.setup(11, gp.OUT)
# gp.setup(12, gp.OUT)
#
# gp.output(7, False)
# gp.output(11, False)
# gp.output(12, True)
# camNum = 3

# def toggle_cam():
#     global camNum
#     gp.setmode(gp.BOARD)
#     if camNum == 1:
#         camNum = 3
#         gp.output(7, False)
#         gp.output(11, True)
#         gp.output(12, False)
#
#     elif camNum == 3:
#         camNum = 1
#         gp.output(7, False)
#         gp.output(11, False)
#         gp.output(12, True)

def captureImages(sendPipe, event=None):
    # The code for capturing images

    def toggle_cam():
        global camNum
        gp.setmode(gp.BOARD)
        if camNum == 1:
            camNum = 3
            gp.output(7, False)
            gp.output(11, True)
            gp.output(12, False)

        elif camNum == 3:
            camNum = 1
            gp.output(7, False)
            gp.output(11, False)
            gp.output(12, True)

    gp.setwarnings(False)
    gp.setmode(gp.BOARD)

    gp.setup(7, gp.OUT)
    gp.setup(11, gp.OUT)
    gp.setup(12, gp.OUT)

    gp.output(7, False)
    gp.output(11, False)
    gp.output(12, True)
    camNum = 3

    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera)

def rectifyImages(receivePipe, sendPipe, calibParams, event=None):
    # The code for rectifying captured images

    # Load in the calibration results, and compute the rectification maps

    distcoeffs = np.zeros(5) # Just ignore distortion

    outR1, outR2, outP1, outP2, outQ, junk1, junk2 = cv2.stereoRectify(cameraMatrix1=kLeft, distCoeffs1=distcoeffs,
                                                                       cameraMatrix2=kRight,
                                                                       distCoeffs2=distcoeffs, imageSize=(640, 480),
                                                                       R=rectifiedR,
                                                                       T=(opticalCenters[0] - opticalCenters[1]))

    map1L, map2L = cv2.initUndistortRectifyMap(cameraMatrix=kAverage, distCoeffs=distcoeffs, R=outR1,
                                               newCameraMatrix=outP1, size=(640, 480), m1type=cv2.CV_16SC2)

    map1R, map2R = cv2.initUndistortRectifyMap(cameraMatrix=kAverage, distCoeffs=distcoeffs, R=outR2,
                                               newCameraMatrix=outP2, size=(640, 480), m1type=cv2.CV_16SC2)

    leftImage = np.zeros((640, 480), dtype=np.uint8)
    rightImage = np.zeros((640, 480), dtype=np.uint8)

    workWithLeftImage = True

    while not event.is_set():
        if workWithLeftImage:
            leftImage = receivePipe.recv()
            # rectify!
            leftImage = cv2.remap(src=leftImage, map1=map1L, map2=map2L, interpolation=cv2.INTER_LINEAR)
            sendPipe.send(leftImage)
            workWithLeftImage = False
        else:
            rightImage = receivePipe.recv()
            # rectify!
            rightImage = cv2.remap(src=rightImage, map1=map1R, map2=map2R, interpolation=cv2.INTER_LINEAR)
            sendPipe.send(rightImage)
            workWithLeftImage = True

def stereoMatchImages(receivePipe, sendPipe, event=None):
    # The code for stereo matching the images

    leftImage = np.zeros((640, 480), dtype=np.uint8)
    rightImage = np.zeros((640, 480), dtype=np.uint8)

    receiveLeftImage = True

    while not event.is_set():
        if receiveLeftImage:
            leftImage = receivePipe.recv()
            receiveLeftImage = False
        else:
            rightImage = receivePipe.recv()
            receiveLeftImage = True

        # Do stereo matching on the images, and pass the results through to the decision process
        disparityMap = np.zeros((640, 480), dtype=np.uint8)
        sendPipe.send(disparityMap)

def makeDecision(receivePipe, event=None):
    while not event.is_set():
        image = receivePipe.recv()
        image += 5
        biggestX, biggestY = np.unravel_index(np.argmax(image), image.shape)
        np.average(image[(biggestX-2):(biggestX+2), (biggestY-2):(biggestY+2)])

    # Do any termination stuff required


# Load/specify calibration parameters

calibrationParameters = 0

# Create processes, one for each task
# Create also pipes for communication between processes
# 1.  Capture images from the cameras
# 2.  Rectify images
# 3.  Stereo match images
# 4.  Process disparity map and make decision

event = Event()

rectifyReceiverPipe, captureSenderPipe = Pipe(False)
stereoMatchReceiverPipe, rectifySenderPipe = Pipe(False)
decideReceiverPipe, stereoMatchSenderPipe = Pipe(False)

captureProcess = Process(target=captureImages, args=captureSenderPipe, kwargs={'event': event})
rectifyProcess = Process(target=rectifyImages, args=(rectifyReceiverPipe, rectifySenderPipe, calibrationParameters), kwargs={'event': event})
stereoMatchProcess = Process(target=stereoMatchImages, args=(stereoMatchReceiverPipe, stereoMatchSenderPipe), kwargs={'event': event})
decideProcess = Process(target=makeDecision, args=decideReceiverPipe, kwargs={'event': event})

#  Set processes running, and wait for user cancel signal
#  Gracefully terminate the processes, and then exit the program

captureProcess.start()
rectifyProcess.start()
stereoMatchProcess.start()
decideProcess.start()

input("Press Enter to continue...")

event.set()

captureProcess.join()
rectifyProcess.join()
stereoMatchProcess.join()
decideProcess.join()