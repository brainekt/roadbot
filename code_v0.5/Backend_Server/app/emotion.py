from fer import FER

detector = FER(mtcnn=True)


def emotion_detection(frame):
    #detect emotion
    emotion, score = detector.top_emotion(frame)
    
    if emotion == None or score == None:
        emotion = "no strong emotion"
        score = 0.5
    
    return emotion
