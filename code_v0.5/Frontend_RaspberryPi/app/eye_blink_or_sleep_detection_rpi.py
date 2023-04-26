import base64
import json                    
import numpy as np
import requests



# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

from config import *


headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
api = 'http://192.168.0.106:8000/video_feed_eye_blink_or_sleep_detection'


font = cv2.FONT_HERSHEY_SIMPLEX

def eye_blink_or_sleep_detection_rpi_to_server():
    '''
    get frame using picamera and send them to the server for processing
    '''
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    try:
        # set camera resolution
        camera.resolution = RESOLUTION
        #set frame rate
        camera.framerate = 5
        rawCapture = PiRGBArray(camera, size=RESOLUTION)
        # allow the camera to warmup
        time.sleep(2)#0.1
        count=0
        cons_frame_count=0
        blink_count=0
        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image - this array
            # will be 3D, representing the width, height, and # of channels
            count+=1
            
            image = frame.array
            
            
            if count%1==0:
                
                payload = json.dumps({"image": image.tolist()})
                #send api request
                response = requests.post(api, data=payload, headers=headers)
                try:
                    data = response.json()
                    
                    if data['output']==1:
                        cons_frame_count+=1
                    else:
                        if cons_frame_count>=1:
                            cons_frame_count=0
                            blink_count+=1
                        
                            
                    
                    image = cv2.resize(image,(640,480))
                    
                    if cons_frame_count>=5:
                        output = "The Person is Sleeping"
                        image = cv2.putText(image, output, (30,30), font, 
                           0.8, (75,45,227), 2, cv2.LINE_AA)
                    else:
                        output = "Blink: "+str(blink_count)
                        image = cv2.putText(image, output, (30,30), font, 
                           0.8, (255,0,76), 2, cv2.LINE_AA)
                    
                    
                    print(data)                
                except:
                    print(response.text)
            
            rawCapture.truncate(0)
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    finally:
        camera.close() 
            


  



