# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

import RPi.GPIO as gp

picNum = 0
leftImage = True

gp.setwarnings(False)
gp.setmode(gp.BOARD)

gp.setup(7, gp.OUT)
gp.setup(11, gp.OUT)
gp.setup(12, gp.OUT)

gp.output(7, False)
gp.output(11, False)
gp.output(12, True)
camNum = 1

def toggle_cam():
    global camNum
    gp.setmode(gp.BOARD)
    if camNum == 1:
        camNum = 2
        gp.output(7, True)
        gp.output(11, False)
        gp.output(12, True)        

    elif camNum == 2:
        camNum = 1
        gp.output(7, False)
        gp.output(11, False)
        gp.output(12, True)

def filenames():
    global leftImage
    global picNum

    if leftImage:
        returnString = "leftPic%02d.jpg" % picNum
        leftImage = False
        return returnString

    else:
        returnString = "rightPic%02d.jpg" % picNum
        leftImage = True
        picNum += 1
        return returnString
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
#rawCapture = PiRGBArray(camera, size=(800, 600))
rawCapture = PiRGBArray(camera)
#rawCapture = np.empty((480, 640, 3), dtype=np.uint8)
 
# allow the camera to warmup
time.sleep(2)

kernel = np.ones((5,5), np.float32)/25
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array

	toggle_cam()

	blurred = cv2.filter2D(image, -1, kernel)
 
	# show the frame
	cv2.imshow("Frame", blurred)
	#nextFileName = filenames()
	#cv2.imwrite(nextFileName, image)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)	
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
camera.close()
rawCapture.close()
