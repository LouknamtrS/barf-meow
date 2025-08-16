from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
from PIL import Image
import io

app = FastAPI()

# Configure CORS to allow requests from Unity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to be more restrictive for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
with open('svm_model_v3.pkl', 'rb') as f:
    svm = pickle.load(f)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Changed to True since we're processing single images
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# State tracking variables
current_state = "none"
last_stable_state = "none"
stable_count = 0
STABILITY_THRESHOLD = 5

@app.post("/predict")
async def predict_gesture(file: UploadFile = File(...)):
    global current_state, last_stable_state, stable_count
    
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Convert to numpy array and OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)  # Mirror the image like in your original code
        
        # Process with Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    data = np.array(landmarks)
                    y_pred = svm.predict(data.reshape(1, -1))
                    predicted_gesture = str(y_pred[0])

                    # State Machine
                    if predicted_gesture == current_state:
                        stable_count += 1
                    else:
                        current_state = predicted_gesture
                        stable_count = 1

                    if stable_count >= STABILITY_THRESHOLD:
                        last_stable_state = current_state

                    return {"gesture": last_stable_state}
                else:
                    return {"gesture": "none", "message": "Incomplete hand landmarks"}
        else:
            return {"gesture": "none", "message": "No hand detected"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)