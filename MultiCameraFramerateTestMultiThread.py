#Single camera framerate test
from picamera import PiCamera
from io import BytesIO
from threading import Thread
from PiVideoStreamMultiCam import PiVideoStreamMultiCam
import picamera.array
import time
import cv2
import numpy as np
##import RPi.GPIO as gp
##
##gp.setwarnings(False)
##gp.setmode(gp.BOARD)
##
##gp.setup(7, gp.OUT)
##gp.setup(11, gp.OUT)
##gp.setup(12, gp.OUT)
##
##gp.output(7, False)
##gp.output(11, False)
##gp.output(12, True)
##camNum = 1
picNum = 0
leftImage = True

##def toggle_cam():
##	global camNum
##	global leftImage
##	gp.setmode(gp.BOARD)
##	if camNum == 1:
##		camNum = 3
##		gp.output(7, False)
##		gp.output(11, True)
##		gp.output(12, False)
##		leftImage = True
##
##	elif camNum == 3:
##		camNum = 1
##		gp.output(7, False)
##		gp.output(11, False)
##		gp.output(12, True)
##		leftImage = False

##def threadTog():
##        while True:
##                toggle_cam()
##                time.sleep(.15)

def filenames():
    global leftImage
    global picNum

    if leftImage:
        returnString = "leftCamModePicFast%02d.jpg" % picNum
        leftImage = False
        return returnString

    else:
        returnString = "rightCamModePicFast%02d.jpg" % picNum
        leftImage = True
        picNum += 1
        return returnString

##def returnImage():
##        global image
##        global camera
##        global output
##        with picamera.array.PiRGBArray(camera) as output:
##                for image in camera.capture_continuous(output, format='bgr', use_video_port=True):
##                        toggle_cam()
##                        image = output.array

##with PiCamera() as camera:
##    camera.resolution = (640, 480)
##    camera.framerate = 32
##    frameCount = 1
##    numImages = 100
##    time.sleep(2) #give the camera a chance to warm up
##
####    #camera.start_preview()
####    camera.start_recording('clocktest.h264')
####    camera.wait_recording(10)
####    camera.stop_recording()
####    #camera.stop_preview()
##    #t = Thread(target=threadTog)
##    #t.daemon = True
##    #t.start()
##
##    beforeTime = time.perf_counter()
####    camera.capture_sequence([
####        'image%02d.jpg' % i
####        for i in range(100)
####        ], use_video_port=True)
##    
####    my_stream = BytesIO()
####    for image in camera.capture_continuous(my_stream, format='jpeg', use_video_port=True):
####        toggle_cam()
####        my_stream.truncate()
####        frameCount = frameCount + 1
####        my_stream.seek(0)
####        if frameCount > numImages:
####            break
##
####    with picamera.array.PiRGBArray(camera) as output:
####        for image in camera.capture_continuous(output, format='bgr', use_video_port=True):
####            toggle_cam()
######            if leftImage:
######                    leftStream = output.array
######            else:
######                    rightStream = output.array
####            nextFileName = filenames()
####            cv2.imwrite(nextFileName, output.array)
####            #image = cv2.Canny(output.array, 100, 200)
####            #cv2.imshow("Capture", output.array)
####            #key = cv2.waitKey(1) & 0xFF
####            frameCount = frameCount + 1
####            #blur = cv2.GaussianBlur(output.array, (5,5), 0)
####            output.truncate(0)
####            if frameCount > numImages:
####                break
##
####        nextFilename = filenames()
####        cv2.imwrite(nextFilename, leftStream)
####        nextFilename = filenames()
####        cv2.imwrite(nextFilename, rightStream)
##
####    image = np.empty((480, 640, 3), dtype=np.uint8)
####    for output in camera.capture_continuous(image, format='bgr', use_video_port=True):
####        toggle_cam()
####        cv2.imshow("Capture", image)
####        key = cv2.waitKey(1) & 0xFF
####        frameCount = frameCount + 1
####        if frameCount > numImages:
####            break
##
####    camera.capture_sequence([
####            'Images/image%02d.jpg' % i
####            for i in range(numImages)
####            ])
##  
##    afterTime = time.perf_counter()
##
##    elapsedTime = afterTime - beforeTime
##
##    #my_stream.close()
##
##    cv2.destroyAllWindows()
##
##    print("recorded %d image pairs in %f seconds, for a total of %f fps" % ((numImages/2), elapsedTime, ((numImages/2) / elapsedTime)))
##    print("average time per image pair = %fs" % (elapsedTime / (numImages / 2)))
##

vs = PiVideoStreamMultiCam(resolution=(640,480), framerate=32).start()
numImages = 100
frameCount = 1
time.sleep(2)

beforeTime = time.perf_counter()

while True:
        nextFilename = filenames()
        cv2.imwrite(nextFilename, vs.read())
        frameCount = frameCount + 1

        if frameCount > numImages:
                break

afterTime = time.perf_counter()

elapsedTime = afterTime - beforeTime

print("recorded %d image pairs in %f seconds, for a total of %f fps" % ((numImages/2), elapsedTime, ((numImages/2) / elapsedTime)))
print("average time per image pair = %fs" % (elapsedTime / (numImages / 2)))

vs.stop()
