# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from PiVideoStream import PiVideoStream
import RPi.GPIO as gp

picNum = 0
leftImage = True
camNum = 1

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

vs = PiVideoStream().start()
time.sleep(0.5)
while True:
    
    frame = vs.read()
    toggle_cam()
    cv2.imshow("Frame", frame)
    nextFileName = filenames()
    cv2.imwrite(nextFileName, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
