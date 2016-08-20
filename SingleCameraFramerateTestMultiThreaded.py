# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from PiVideoStream import PiVideoStream
import RPi.GPIO as gp
import numpy as np

frameCount = 1
picNum = 0
##leftImage = True
##camNum = 1
##
gp.setwarnings(False)
gp.setmode(gp.BOARD)

gp.setup(7, gp.OUT)
gp.setup(11, gp.OUT)
gp.setup(12, gp.OUT)

gp.output(7, False)
gp.output(11, False)
gp.output(12, True)
##camNum = 1
##
##def toggle_cam():
##    global camNum
##    gp.setmode(gp.BOARD)
##    if camNum == 1:
##        camNum = 2
##        gp.output(7, True)
##        gp.output(11, False)
##        gp.output(12, True)        
##
##    elif camNum == 2:
##        camNum = 1
##        gp.output(7, False)
##        gp.output(11, False)
##        gp.output(12, True)
##
##def filenames():
##    global leftImage
##    global picNum
##
##    if leftImage:
##        returnString = "leftPic%02d.jpg" % picNum
##        leftImage = False
##        return returnString
##
##    else:
##        returnString = "rightPic%02d.jpg" % picNum
##        leftImage = True
##        picNum += 1
##        return returnString

def filenames():
    global picNum
    picNum = picNum + 1
    return "camModeTestFast%02d.jpg" % picNum

numImages = 50
vs = PiVideoStream(resolution=(640, 480), framerate=48).start()
#kernel = np.ones((5,5), np.float32)/25
time.sleep(2)
beforeTime = time.perf_counter()
while True:
    
    frame = vs.read()
    #frame = cv2.filter2D(frame, -1, kernel)
    #frame = cv2.GaussianBlur(frame, (5,5), 0)
    #frame = cv2.bilateralFilter(frame, 9, 75, 75)
    #toggle_cam()
    frameCount = frameCount + 1
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.boxFilter(frame, -1, (5,5))
    #frame = cv2.boxFilter(frame, -1, (5,5))
    #frame = cv2.Canny(frame, 100, 200)
    #cv2.imshow("Frame", frame)
    nextFileName = filenames()
    cv2.imwrite(nextFileName, frame)
    #key = cv2.waitKey(1) & 0xFF

    #if key == ord("q"):
     #   break

    if frameCount > numImages:
         break

afterTime = time.perf_counter()
elapsedTime = afterTime - beforeTime

cv2.destroyAllWindows()
vs.stop()

print("recorded %d images in %f seconds, for a total of %f fps" % (numImages, elapsedTime, (numImages / elapsedTime)))

###Single camera framerate test
##from picamera import PiCamera
##from io import BytesIO
##import picamera.array
##import time
##import cv2
##
##
##
##with PiCamera() as camera:
##    camera.resolution = (1280, 960)
##    camera.framerate = 32
##    time.sleep(2) #give the camera a chance to warm up
##
####    #camera.start_preview()
####    camera.start_recording('clocktest.h264')
####    camera.wait_recording(10)
####    camera.stop_recording()
####    #camera.stop_preview()
##
##    beforeTime = time.perf_counter()
####    camera.capture_sequence([
####        'image%02d.jpg' % i
####        for i in range(100)
####        ], use_video_port=True)
##    frameCount = 1
##    numImages = 1000
##    #my_stream = BytesIO()
####    for image in camera.capture_continuous(my_stream, format='jpeg', use_video_port=True):
####        my_stream.truncate()
####        frameCount = frameCount + 1
####        my_stream.seek(0)
####        if frameCount > numImages:
####            break
##
##    with picamera.array.PiRGBArray(camera) as output:
##        for image in camera.capture_continuous(output, format='bgr', use_video_port=True):
##            cv2.imshow("Capture", output.array)
##            key = cv2.waitKey(1) & 0xFF
##            frameCount = frameCount + 1
##            output.truncate(0)
##            if frameCount > numImages:
##                break
##    
##    afterTime = time.perf_counter()
##
##    elapsedTime = afterTime - beforeTime
##
##    #my_stream.close()
##
##    cv2.destroyAllWindows()
##
##    print("recorded %d images in %f seconds, for a total of %f fps" % (numImages, elapsedTime, (numImages / elapsedTime)))
