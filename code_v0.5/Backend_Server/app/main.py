from flask import Flask, render_template, request, abort
from werkzeug.middleware.profiler import ProfilerMiddleware
import numpy as np

from eye_blink_sleep_detection import eye_processor
from emotion import emotion_detection
from emotion_and_eye_blink import emotion_and_eye_processor
from image_annotation import facial_expression_processor

app = Flask(__name__,template_folder='templates')
#app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=('eye_blink_sleep_detection.py',), profile_dir='app/profiling') #restrictions=[20])

@app.get('/')
async def index():
    return render_template('home.html')

@app.post('/video_feed_eye_blink_or_sleep_detection')
def video_feed_eye_blink():
    """
    count eye blink
    """        
    # print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']

    # PIL image object to numpy array
    img_arr = np.asarray(im_b64, dtype=np.uint8)      
    #print('img shape', img_arr.shape)

    # process your img_arr here   
    value = eye_processor(img_arr)
    print("Value: ",value)
    result_dict = {'output': value}
    return result_dict

@app.post("/video_feed_emotion")
async def emotion():         
    # print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']

    # PIL image object to numpy array
    img_arr = np.asarray(im_b64, dtype=np.uint8)      
    print('img shape', img_arr.shape)

    # process your img_arr here   
    value = emotion_detection(img_arr)

    result_dict = {'output': value}
    return result_dict

@app.post("/video_feed_emotion_eye_blink")
async def emotion_and_eye_blink():         
    # print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']

    # PIL image object to numpy array
    img_arr = np.asarray(im_b64, dtype=np.uint8)      
    print('img shape', img_arr.shape)

    # process your img_arr here   
    value = emotion_and_eye_processor(img_arr)

    result_dict = {'output': value}
    return result_dict

@app.post("/image_annote_emotion_gaze_and_sleep")
async def image_emotion_and_eye_blink():  
          
    #print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']

    # PIL image object to numpy array
    img_arr = np.asarray(im_b64, dtype=np.uint8)      
    print('img shape', img_arr.shape)
    
    # process your img_arr here   
    value = facial_expression_processor(img_arr)
    return value
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000, debug=True)
