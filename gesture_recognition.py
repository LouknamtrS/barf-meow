import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64 
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# Create Flask app
app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST"], allow_headers=["Content-Type"])

# Load model - matching main.py implementation
try:
    with open('mlp_model.pkl', 'rb') as f:
        mlp = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    mlp = None

# Initialize MediaPipe Hands - matching main.py configuration
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to false as in main.py
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Store last processing time for FPS calculation
last_process_time = time.time()
current_fps = 0

def process_image(image):
    frame = cv2.flip(image, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    # Process landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # วาด landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # ดึงข้อมูล landmark
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if len(landmarks) == 63:
                return landmarks
            else:
                print("Incomplete landmarks detected")
    
    return None

@app.route('/predict', methods=['POST'])
def predict():
    global last_process_time, current_fps
    
    try:
        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - last_process_time
        current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        last_process_time = current_time
        
        # Get image data from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process image to get landmarks using main.py approach
        landmarks = process_image(image)
        
        if landmarks is None:
            return jsonify({
                'gesture': 'none',
                'confidence': 0.0,
                'error': 'No hand detected'
            })
        
        if mlp is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Make prediction using the same approach as main.py
        data = np.array(landmarks)
        y_pred = mlp.predict(data.reshape(1, -1))
        predicted_gesture = str(y_pred[0])
        
        print(f"Predicted: {y_pred}, FPS: {current_fps:.2f}")
        
        # Return prediction as JSON with FPS
        return jsonify({
            'gesture': predicted_gesture,
            'confidence': 1.0,
            'fps': round(current_fps, 2)
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

# Add a simple status endpoint
@app.route('/', methods=['GET'])
def status():
    return jsonify({
        'status': 'Hand gesture recognition service is running',
        'fps': round(current_fps, 2)
    })

if __name__ == '__main__':
    print("Starting gesture recognition server on http://localhost:5555")
    app.run(host='0.0.0.0', port=5555, debug=False)
