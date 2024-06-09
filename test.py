import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

while True:
    try:
        relPosLandmarks = []
        xCoordLandmarks = []
        yCoordLandmarks = []

        success, imgFromCam = camera.read()

        H, W, C = imgFromCam.shape

        img_rgb = cv2.cvtColor(imgFromCam, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    imgFromCam,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            landmarks=[]
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    xCoordLandmarks.append(x)
                    yCoordLandmarks.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    landmarks.append(x - min(xCoordLandmarks))
                    landmarks.append(y - min(yCoordLandmarks))

            x1 = int(min(xCoordLandmarks) * W) - 10
            y1 = int(min(yCoordLandmarks) * H) - 10

            x2 = int(max(xCoordLandmarks) * W) - 10
            y2 = int(max(yCoordLandmarks) * H) - 10
            prediction=""
            prediction = model.predict([np.asarray(landmarks)])
            if(prediction==None):
                predicted_character="Not Found"
            else:
                predicted_character = labels_dict[int(prediction[0])]
            cv2.rectangle(imgFromCam, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(imgFromCam, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
    except:
        pass
    cv2.imshow('frame', imgFromCam)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break


camera.release()
cv2.destroyAllWindows()