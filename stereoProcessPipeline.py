import cv2
import numpy as np
from multiprocessing import Process
from multiprocessing import Pipe
from multiprocessing import Event

def dummy(blah, event=None):
    print("Hello World!")

def captureImages(sendPipe, event=None):
    # The code for capturing images
    print("I can has cheezburger")

def rectifyImages(receivePipe, sendPipe, event=None):
    # The code for rectifying captured images

    leftImage = np.zeros((640, 480), dtype=np.uint8)
    rightImage = np.zeros((640, 480), dtype=np.uint8)

    workWithLeftImage = True

    while not event.is_set():
        if workWithLeftImage:
            leftImage = receivePipe.recv()
            #rectify!
            sendPipe.send(leftImage)
            workWithLeftImage = False
        else:
            rightImage = receivePipe.recv()
            #rectify!
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

captureProcess = Process(target=captureImages(), args=captureSenderPipe, kwargs={'event': event})
rectifyProcess = Process(target=rectifyImages(), args=(rectifyReceiverPipe, rectifySenderPipe), kwargs={'event': event})
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