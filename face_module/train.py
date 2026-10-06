# face_module/train.py
import cv2
import os
import numpy as np
from PIL import Image

def train_model(uploads_path="uploads", model_path="models/lbph_model.yml"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []

    for user_folder in os.listdir(uploads_path):
        folder_path = os.path.join(uploads_path, user_folder)
        if not os.path.isdir(folder_path):
            continue

        label = int(user_folder) if user_folder.isdigit() else hash(user_folder) % 10000

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert('L')  # grayscale
            img_array = np.array(img, 'uint8')
            faces.append(img_array)
            labels.append(label)

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(model_path)
        print(f"[INFO] Model trained and saved → {model_path}")
        return True
    return False