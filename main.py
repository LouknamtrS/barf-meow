import cv2
import mediapipe as mp
import numpy as np
import time
import pickle

#โหลดโมเดล
with open('mlp_model.pkl', 'rb') as f:
    svm = pickle.load(f)

#Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.flip(frame, 1)
    #แปลงภาพเป็น RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    #andmark
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #ดึงข้อมูล landmark
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                data = np.array(landmarks)
                y_pred = svm.predict(data.reshape(1, -1))
                print("Predicted:", y_pred)

                cv2.putText(frame, str(y_pred[0]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
            else:
                print("ไม่พบ landmark ครบ 21 จุด")

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
