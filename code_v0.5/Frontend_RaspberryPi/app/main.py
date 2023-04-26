from flask import Flask, render_template, Response, request
from werkzeug.utils import secure_filename

from eye_blink_or_sleep_detection_rpi import eye_blink_or_sleep_detection_rpi_to_server
from emotion_rpi import emotion_rpi_to_server
from emotion_and_eye_blink_rpi import emotion_and_eye_blink_rpi_to_server
import image_anotation_rpi

app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/eye_blink_sleep_detection_rpi')
def video_feed_sleep_detection():
    """
    eye blink or sleep detection
    """
    return Response(eye_blink_or_sleep_detection_rpi_to_server(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_rpi')
def video_feed_emotion():
    """
    detect emotion
    """
    
    return Response(emotion_rpi_to_server(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_and_eye_blink_rpi')
def video_feed_emotion_and_eyeblink():
    """
    detect emotion and eye blink
    """
    
    return Response(emotion_and_eye_blink_rpi_to_server(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_image')
def upload_file():
    '''
    upload image(.jpg, .jpeg, .png)
    '''
    return render_template('upload.html')

@app.route('/image_annotation_rpi', methods = ['GET', 'POST'])
def image_annotation():
   '''
   send the uploaded image to the server for further processing
   '''
   if request.method == 'POST':
      f = request.files['file']
      
      file_name="app/upload/"+secure_filename(f.filename)
      
      f.save(file_name)
      #send the image to th eserver for processing
      return Response(image_anotation_rpi.post_image(str(file_name)),mimetype='multipart/x-mixed-replace; boundary=frame')
                    


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000, debug=True)