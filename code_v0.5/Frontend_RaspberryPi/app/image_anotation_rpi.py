import requests
import io                    
import base64
import cv2
import json
import numpy as np
from PIL import Image
#headers = {'Content-type': 'image/jpeg'}#'image/jpeg''application/json'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
api = 'http://192.168.0.106:8000/image_annote_emotion_gaze_and_sleep'

def post_image(image_file_name):
    """ post image and return the response """
    
    #read the image
    img = cv2.imread(image_file_name)#'image720x720.jpg'
    #send image to the sever for processing (emotion, sleep detedction)
    response = requests.post(api, data=json.dumps({"image": img.tolist()}), headers=headers)
    
    yield (b'--frame\r\n'
            b'Content-Type: image/jpg\r\n\r\n' + response.content  + b'\r\n\r\n')