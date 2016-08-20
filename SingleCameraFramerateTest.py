#Single camera framerate test
from picamera import PiCamera
from io import BytesIO
import picamera.array
import time
import cv2
import numpy as np
import RPi.GPIO as gp

gp.setwarnings(False)
gp.setmode(gp.BOARD)

gp.setup(7, gp.OUT)
gp.setup(11, gp.OUT)
gp.setup(12, gp.OUT)

gp.output(7, False)
gp.output(11, False)
gp.output(12, True)


with PiCamera() as camera:
    camera.resolution = (1280, 960)
    camera.framerate = 32
    time.sleep(2) #give the camera a chance to warm up

##    #camera.start_preview()
##    camera.start_recording('clocktest.h264')
##    camera.wait_recording(10)
##    camera.stop_recording()
##    #camera.stop_preview()

    beforeTime = time.perf_counter()
##    camera.capture_sequence([
##        'image%02d.jpg' % i
##        for i in range(100)
##        ], use_video_port=True)
    frameCount = 1
    numImages = 100
##    my_stream = BytesIO()
##    for image in camera.capture_continuous(my_stream, format='jpeg', use_video_port=True):
##        my_stream.truncate()
##        frameCount = frameCount + 1
##        my_stream.seek(0)
##        if frameCount > numImages:
##            break

    with picamera.array.PiRGBArray(camera) as output:
        for image in camera.capture_continuous(output, format='bgr', use_video_port=True):
            cv2.imshow("Capture", output.array)
            key = cv2.waitKey(1) & 0xFF
            frameCount = frameCount + 1
            output.truncate(0)
            if frameCount > numImages:
                break

##    image = np.empty((480, 640, 3), dtype=np.uint8)
##    for output in camera.capture_continuous(image, format='bgr', use_video_port=True):
##        cv2.imshow("Capture", image)
##        key = cv2.waitKey(1) & 0xFF
##        frameCount = frameCount + 1
##        if frameCount > numImages:
##            break
  
    afterTime = time.perf_counter()

    elapsedTime = afterTime - beforeTime

    #my_stream.close()

    cv2.destroyAllWindows()

    print("recorded %d images in %f seconds, for a total of %f fps" % (numImages, elapsedTime, (numImages / elapsedTime)))
