from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2


from config import *

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


def eye_processor(frame):
    """Main function.
    
    Parameters
    ----------
    args : dict
        A dictionary containing the arguments from the parser.
    """
    global COUNTER, TOTAL, ASLEEP

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        if ear < EYE_AR_THRESH:
            return 1
        else:
            return 0