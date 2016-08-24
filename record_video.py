import picamera
import RPi.GPIO as gp
import termios
import sys
import select

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

camera = picamera.PiCamera()
camera.resolution = (640, 480)
old_settings = termios.tcgetattr(sys.stdin)
camera.start_recording('my_video.h264')
#camera.wait_recording(10)
while True:
    toggle_cam()
    camera.wait_recording(0.3)
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            q = sys.stdin.read(1)
            break
    
camera.stop_recording()
