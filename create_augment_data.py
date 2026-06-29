import cv2
import numpy as np
from pathlib import Path
import random

def augment_image(image, class_name):
    """ฟังก์ชันสำหรับทำ Data Augmentation: Contrast/Brightness, Blur แบบสุ่ม"""
    augmented_images = []

    #Random Contrast & Brightness
    alpha = random.uniform(0.8, 1.5)  #contrast: 0.8–1.5
    beta = random.randint(-30, 30)    #brightness: -30 ถึง +30
    contrast_bright = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmented_images.append(contrast_bright)

    #Random Gaussian Blur
    #เลือก kernel 3,5,7
    ksize = random.choice([3,5,7])
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    augmented_images.append(blurred)

    #Random Horizontal Flip
    if class_name not in ["rh_left", "rh_right", "lh_left", "lh_right"]:
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)

    return augmented_images

def data_augmentation(dataset_path):
    """ทำ Data Augmentation สำหรับทุกคลาส และเซฟรวมกับ original"""
    dataset_path = Path(dataset_path)

    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            print(f"Processing class: {class_folder.name}")

            for image_file in class_folder.iterdir():
                if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:

                    if "_aug" in image_file.stem:
                        continue

                    image = cv2.imread(str(image_file))

                    if image is None:
                        print(f"Cannot read {image_file}")
                        continue

                    #Generate augmented images
                    augmented_images = augment_image(image, class_folder.name)
                    for idx, aug_img in enumerate(augmented_images):
                        aug_filename = f"{image_file.stem}_aug{idx}{image_file.suffix}"
                        cv2.imwrite(str(class_folder / aug_filename), aug_img)

    print("Data Augmentation Completed!")

if __name__ == "__main__":
    dataset_path = "DATASET5_augmented"
    data_augmentation(dataset_path)
