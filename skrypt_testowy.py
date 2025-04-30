import numpy as np
import cv2
import os
import random
from keras.models import load_model

data_dir = "Train"
model = load_model('gtsrb_model.h5')

class_labels = [f"Znak {i}" for i in range(43)]

def load_and_prepare_image(img_path, img_size=(32, 32)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać: {img_path}")
    img_resized = cv2.resize(img, img_size)
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img, img_expanded

# Losuj 5 unikalnych klas
selected_classes = random.sample(range(43), 5)

for class_id in selected_classes:
    subclass_id = random.randint(0, 5)
    variant_id = random.randint(0, 29)

    # Zbuduj nazwę pliku
    filename = f"{class_id:05d}_{subclass_id:05d}_{variant_id:05d}.png"
    full_path = os.path.join(data_dir, str(class_id), filename)

    try:
        img_orig, img_input = load_and_prepare_image(full_path)

        pred = model.predict(img_input)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)

        is_correct = (pred_class == class_id)
        label_text = f"Klasa: {class_labels[class_id]} "
        color = (0, 255, 0) if is_correct else (0, 0, 255)

        label_text2 = f"Przewidywana: {class_labels[pred_class]} ({confidence * 100:.1f}%)"
        color = (0, 255, 0) if is_correct else (0, 0, 255)

        # Wyświetl
        img_display = cv2.resize(img_orig, (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img_display, label_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(img_display, label_text2, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imshow("Predykcja", img_display)
        cv2.waitKey(0)

    except Exception as e:
        print(f"[!] Błąd dla pliku: {filename} → {e}")

cv2.destroyAllWindows()
