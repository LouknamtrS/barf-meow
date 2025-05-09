import cv2
import os
import numpy as np
from pathlib import Path

def augment_image(image):
    """ฟังก์ชันสำหรับทำ Data Augmentation"""
    augmented_images = []

    #Rotate
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)  # Rotate 15 degrees
    rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    augmented_images.append(rotated)

    return augmented_images

def data_augmentation(dataset_path, excluded_classes):
    """ทำ Data Augmentation โดยยกเว้นบางคลาส"""
    dataset_path = Path(dataset_path)
    augmented_path = dataset_path / "augmented"
    augmented_path.mkdir(parents=True, exist_ok=True)

    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir() and class_folder.name not in excluded_classes:
            print(f"Processing class: {class_folder.name}")
            class_augmented_path = augmented_path / class_folder.name
            class_augmented_path.mkdir(parents=True, exist_ok=True)

            for image_file in class_folder.iterdir():
                if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    image = cv2.imread(str(image_file))

                    # Save original image
                    cv2.imwrite(str(class_augmented_path / image_file.name), image)

                    # Generate augmented images
                    augmented_images = augment_image(image)
                    for idx, aug_img in enumerate(augmented_images):
                        aug_filename = f"{image_file.stem}_aug{idx}{image_file.suffix}"
                        cv2.imwrite(str(class_augmented_path / aug_filename), aug_img)

    print("✅ Data Augmentation Completed!")

if __name__ == "__main__":
    dataset_path = "DATASET"  # Path to your dataset
    excluded_classes = ["left", "right", "back"]  # Classes to exclude
    data_augmentation(dataset_path, excluded_classes)