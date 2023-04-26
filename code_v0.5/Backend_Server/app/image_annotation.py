from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

from fer import FER

from eye_blink_sleep_detection import eye_aspect_ratio
from config import *

font = cv2.FONT_HERSHEY_SIMPLEX

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('app/assets/data/shape_predictor_68_face_landmarks.dat')

(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# mark detector to detect landmarks.
mark_detector = MarkDetector()

# emotion detector
emotion_detector = FER(mtcnn=True)

# Introduce a pose estimator to solve pose.
pose_estimator = PoseEstimator(img_size=(720,720))

def facial_expression_processor(frame):
    """
    detect gaze + emotion + eye blink + sleep detection together
    return frame with gaze + emotion + eye blink + sleep detection together
    """
    #resize frame
    frame = cv2.resize(frame, (720,720))#(480,640)

    #detect emotion
    emotion, score = emotion_detector.top_emotion(frame)
    
    if emotion == None or score == None:
        emotion = "no strong emotion"
        score = 0.5
    #convert color bgr to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect facial points
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for _, (i, j) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            for (x, y) in shape[i:j]:
                cv2.circle(frame, (x, y), 1, POINT_COLOR, -1)

        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        
        #calculate eye aspect ration
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, EYES_COLOR, 1)
        cv2.drawContours(frame, [right_eye_hull], -1, EYES_COLOR, 1)
        
        if ear < EYE_AR_THRESH:
            ASLEEP = True
        else:
            ASLEEP = False
       
        if ASLEEP:
            cv2.putText(
                frame, 
                f'The person is sleeping!',
                (10, 30), font,
                0.7, RED, 2
            )
        else:
            cv2.putText(
                frame, 
                f'The person is Paying Attention',
                (10, 30), font,
                0.7, TEXT_COLOR, 2
            )


    
    # Step 1: Get a face from current frame.
    facebox = mark_detector.extract_cnn_facebox(frame)

    # Any face found?
    if facebox is not None:

        # Step 2: Detect landmarks. Crop and feed the face area into the
        # mark detector.
        x1, y1, x2, y2 = facebox
        face_img = frame[y1: y2, x1: x2]

        # Run the detection.
        #tm.start()
        marks = mark_detector.detect_marks(face_img)
        #tm.stop()

        # Convert the locations from local face area to the global image.
        marks *= (x2 - x1)
        marks[:, 0] += x1
        marks[:, 1] += y1

        # Try pose estimation with 68 points.
        pose = pose_estimator.solve_pose_by_68_points(marks)

        # All done. The best way to show the result would be drawing the
        # pose on the frame in realtime.

        # Do you want to see the pose annotation?
        pose_estimator.draw_annotation_box(
            frame, pose[0], pose[1], color=(0, 255, 0))

    # write emotion and score to the frame
    cv2.putText(frame, f"{emotion}, {score}", (10,60), font, 1, TEXT_COLOR, 2, cv2.LINE_AA)
    
    _, img_encode = cv2.imencode('.jpg', frame)
    
    return img_encode.tobytes() 
    