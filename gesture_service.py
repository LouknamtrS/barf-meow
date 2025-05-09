import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
from flask import Flask, request, jsonify

# Create Flask app
app = Flask(__name__)

# Load model
try:
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def process_image(image):
    """Process image and extract hand landmarks"""
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(rgb_image, 1)
    
    # Process the image with MediaPipe
    results = hands.process(frame)
    
    # Extract landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if len(landmarks) == 63:  # 21 landmarks with x,y,z coordinates
                return landmarks
    
    return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        
        # Process image to get landmarks
        landmarks = process_image(image)
        
        if landmarks is None:
            return jsonify({
                'gesture': 'none',
                'confidence': 0.0,
                'error': 'No hand detected'
            })
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Make prediction
        prediction = model.predict(np.array(landmarks).reshape(1, -1))
        predicted_gesture = str(prediction[0])
        
        # For confidence, we could use model probabilities if available
        # For now, just use a placeholder value
        confidence = 1.0
        
        print(f"Predicted gesture: {predicted_gesture}")
        
        # Return prediction as JSON
        return jsonify({
            'gesture': predicted_gesture,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

# Add a simple status endpoint
@app.route('/', methods=['GET'])
def status():
    return jsonify({'status': 'Gesture recognition service is running'})

if __name__ == '__main__':
    print("Starting gesture recognition server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
