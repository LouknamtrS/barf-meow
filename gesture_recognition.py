from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import mediapipe as mp
import numpy as np
import base64
import time
import pickle
from PIL import Image
import io
import json

app = FastAPI()

# Load model
with open('svm_model_v5.pkl', 'rb') as f:
    svm = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# State tracking
current_state = "none"
last_stable_state = "none"
stable_count = 0
STABILITY_THRESHOLD = 3

frame_count = 0
last_fps_update = time.time()
current_fps = 0

def calculate_fps():
    global frame_count, last_fps_update, current_fps
    frame_count += 1
    now = time.time()
    if now - last_fps_update >= 1.0:
        current_fps = frame_count / (now - last_fps_update)
        frame_count = 0
        last_fps_update = now
    return current_fps

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global current_state, last_stable_state, stable_count
    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                # แปลง bytes เป็นภาพ
                image = Image.open(io.BytesIO(data)).convert('RGB')
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                frame = cv2.flip(frame, 1)

                # Mediapipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                gesture = "none"
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        if len(landmarks) == 63:
                            data_np = np.array(landmarks)
                            y_pred = svm.predict(data_np.reshape(1, -1))
                            predicted_gesture = str(y_pred[0])

                            # State machine
                            if predicted_gesture == current_state:
                                stable_count += 1
                            else:
                                current_state = predicted_gesture
                                stable_count = 1
                            if stable_count >= STABILITY_THRESHOLD:
                                last_stable_state = current_state
                            gesture = last_stable_state

                fps = calculate_fps()
                response = {"gesture": gesture, "fps": fps}
                await websocket.send_text(json.dumps(response))

            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        print("Client disconnected")
