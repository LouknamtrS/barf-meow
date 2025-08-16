import cv2
import mediapipe as mp
import numpy as np
import time
import pickle

#โหลดโมเดล
with open('svm_model_v3.pkl', 'rb') as f:
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

current_state = "none"
last_stable_state = "none"
stable_count = 0
STABILITY_THRESHOLD = 5  #ต้องเจอ gesture ซ้ำ 5 frame ถึงจะเปลี่ยน

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                data = np.array(landmarks)
                y_pred = svm.predict(data.reshape(1, -1))
                predicted_gesture = str(y_pred[0])

                # --- State Machine ---
                if predicted_gesture == "transition":
                    # ไม่อัปเดต last_stable_state
                    stable_count = 0
                else:
                    if predicted_gesture == current_state:
                        stable_count += 1
                    else:
                        current_state = predicted_gesture
                        stable_count = 1

                    if stable_count >= STABILITY_THRESHOLD:
                        last_stable_state = current_state

                #  last_stable_state แสดงผล
                display_gesture = last_stable_state
                cv2.putText(frame, display_gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

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
