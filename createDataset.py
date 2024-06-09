import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

rootDirectory ='train.py'

data = []
labels = []
for label in os.listdir(rootDirectory):
    for img_path in os.listdir(os.path.join(rootDirectory, label)):
        img = cv2.imread(os.path.join(rootDirectory, label, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            relPosLandmarks=[]
            xCoordLandmarks=[]
            yCoordLandmarks=[]
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    xCoordLandmarks.append(x)
                    yCoordLandmarks.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    relPosLandmarks.append(x - min(xCoordLandmarks))
                    relPosLandmarks.append(y - min(yCoordLandmarks))

            data.append(relPosLandmarks)
            labels.append(label)
print(len(data))
print(type(data))
f = open('dataset.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()