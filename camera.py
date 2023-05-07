import cv2
import av
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 
import webbrowser
model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
try:
     emotion=np.load("emotion.npy")[0]
except:
      emotion=""
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frm = self.video.read()
        #start
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst=[]
        global em

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1,-1)

            pred = label[np.argmax(model.predict(lst))]
            em=pred

            # print(pred)
            cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)
            np.save("emotion.npy",np.array([pred]))

        ret, jpeg = cv2.imencode('.jpg', frm)
        return jpeg.tobytes(),em
