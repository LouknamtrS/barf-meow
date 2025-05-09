import numpy as np
import cv2 as cv
from pathlib import Path

def get_image():
    Class = 'yes'
    save_path = Path(f'DATASET/{Class}')
    save_path.mkdir(parents=True, exist_ok=True)

    existing_files = list(save_path.glob('*.png'))
    if existing_files:
        existing_nums = [int(f.stem) for f in existing_files if f.stem.isdigit()]
        start_num = max(existing_nums) + 5
    else:
        start_num = 5

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    i = start_num    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv.flip(frame, 1)

        if i % 5 == 0:
            cv.imwrite(str(save_path / f'{i}.png'), frame)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q') or i >= start_num + 500:
            break

        i += 1

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
   get_image()
