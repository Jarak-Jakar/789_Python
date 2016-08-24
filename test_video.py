# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import termios
import sys
import select

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
camNum = 3

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
 
# allow the camera to warmup
time.sleep(4)

old_settings = termios.tcgetattr(sys.stdin)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	toggle_cam()	
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
 
	# show the frame
	#cv2.imshow("Frame", image)
	nextFileName = filenames()
	cv2.imwrite(nextFileName, image)
	#key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)

	#toggle_cam()	
 
##	# if the `q` key was pressed, break from the loop
##	if key == ord("q"):
##		break

	if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            q = sys.stdin.read(1)
            break

termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
cv2.destroyAllWindows()
camera.close()
rawCapture.close()
