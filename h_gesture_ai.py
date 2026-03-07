from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import mediapipe as mp
import numpy as np
import base64
import time
import pickle
import json
import os
import sys
import uvicorn
import asyncio
import socket

app = FastAPI()

# =========================
# Load SVM model
# =========================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

model_path = resource_path("svm_model_v6.pkl")

with open(model_path, "rb") as f:
    svm = pickle.load(f)

# =========================
# Mediapipe setup
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(0)

# =========================
# Gesture state
# =========================
current_state = "none"
last_stable_state = "none"
stable_count = 0
STABILITY_THRESHOLD = 3

# =========================
# FPS
# =========================
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


# =========================
# WebSocket
# =========================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    print("Unity connected")

    global current_state, last_stable_state, stable_count

    try:
        while True:
            try:

                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)

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

                            if predicted_gesture == current_state:
                                stable_count += 1
                            else:
                                current_state = predicted_gesture
                                stable_count = 1

                            if stable_count >= STABILITY_THRESHOLD:
                                last_stable_state = current_state

                            gesture = last_stable_state

                _, buffer = cv2.imencode(".jpg", frame)
                jpg_as_text = base64.b64encode(buffer).decode("utf-8")

                fps = calculate_fps()

                response = {
                    "gesture": gesture,
                    "fps": fps,
                    "frame": jpg_as_text
                }

                await websocket.send_text(json.dumps(response))

                await asyncio.sleep(0.01)

            except Exception as e:
                print("Loop error:", e)

    except WebSocketDisconnect:
        print("Unity disconnected")

        try:
            cap.release()
        except:
            pass

        cv2.destroyAllWindows()

        os._exit(0)



# =========================
# Run server
# =========================

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0



if __name__ == "__main__":
    if port_in_use(8000):
        print("Server already running")
        sys.exit()

    print("Starting Gesture AI Server...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )