from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2


from config import *
from fer import FER

emotion_detector = FER(mtcnn=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('app/assets/data/shape_predictor_68_face_landmarks.dat')

(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

def eye_aspect_ratio(eye):
    """Computes the euclidean distances between eye landmarks.
    
    Parameters
    ----------
    eye : list
        A list containing eyes landmarks.

    Returns
    -------
    ear : float
        The eye aspect ratio (EAR). 
    """

    # Euclidean distance between vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Euclidean distance between horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Compute EAR
    ear = (A + B) / (2.0 * C)

    return ear


def emotion_and_eye_processor(frame):
    """Main function.
    
    Parameters
    ----------
    args : dict
        A dictionary containing the arguments from the parser.
    """
    global COUNTER, TOTAL, ASLEEP

    #convert color bgr to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect facial points
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        
        #calculate eye aspect ration
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0


        EYE_AR_CONSEC_FRAMES = 1
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

    #detect emotion
    emotion, score = emotion_detector.top_emotion(frame)
    
    if emotion == None or score == None:
        emotion = "no strong emotion"
        score = 0.5
    
    return "Emotion: "+emotion+"  Blink: "+str(TOTAL)
        
