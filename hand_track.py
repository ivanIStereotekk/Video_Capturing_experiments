import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(1) # Захват камеры
mpHands = mp.solutions.hands
hands = mpHands.Hands() # Модель рекогнайзера - руки (ctr+Hands) описание параметров
mpDraw = mp.solutions.drawing_utils # Рисовалка точек
while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # переобразовали ГБР--РГБ
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks: # Если hand_process(result) имеет метки
        print(results.multi_hand_landmarks)
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
    cv2.imshow("CAMERA CAPTURE",img)
    cv2.waitKey(1)



#https://www.youtube.com/watch?v=01sAkU_NvOY (15:29)
