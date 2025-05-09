import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 

def image_processed(file_path):
    hand_img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_rgb)
    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        landmarks = []
        for lm in data.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    except:
        return np.zeros(63).tolist()

def make_csv():
    mypath = 'DATASET'
    with open('dataset.csv', 'w') as file:
        headers = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
        file.write(','.join(headers) + '\n')

        for each_folder in os.listdir(mypath):
            if each_folder.startswith('._'):
                continue

            for each_file in os.listdir(os.path.join(mypath, each_folder)):
                if each_file.startswith('._'):
                    continue

                file_path = os.path.join(mypath, each_folder, each_file)
                label = each_folder
                data = image_processed(file_path)

                file.write(','.join([str(i) for i in data]) + f',{label}\n')

    print('Data Created!')

if __name__ == "__main__":
    make_csv()
