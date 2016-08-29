import io
import socket
import time
import picamera
import struct

import RPi.GPIO as gp

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

# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
client_socket = socket.socket()
client_socket.connect(('james-3570R-370R-470R-450R-510R-4450RV', 8000))

# Make a file-like object out of the connection
connection = client_socket.makefile('wb')
try:
    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        camera.framerate = 24
        #rawCapture = PiRGBArray(camera, size=(640, 480))
        # Start a preview and let the camera warm up for 2 seconds
        #camera.start_preview()
        time.sleep(2)
        start = time.time()
        stream = io.BytesIO()
        for frame in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
            #image = frame.array
            

            #stream image over network
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()

            stream.seek(0)
            connection.write(stream.read())
            #connection.write(image)

            if time.time() - start > 5:
                break

            stream.seek(0)
            stream.truncate()
            
            toggle_cam()

        connection.write(struct.pack('<L', 0))
            
finally:
    connection.close()
    client_socket.close()
