import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


data_dir = "Train"
IMG_HEIGHT, IMG_WIDTH = 32, 32
NUM_CLASSES = 43  # Klasy 0–42

images = []
labels = []

for class_id in range(NUM_CLASSES):
    class_path = os.path.join(data_dir, str(class_id))

    if not os.path.isdir(class_path):
        print(f"Pominięto brakujący folder: {class_path}")
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        if img_name.endswith(('.ppm', '.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Nie można wczytać obrazu: {img_path}")
                continue

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(class_id)

X = np.array(images, dtype='float32') / 255.0  # Normalizacja
y = to_categorical(np.array(labels), num_classes=NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Załadowano {X.shape[0]} obrazków.")
print(f"Trening: {X_train.shape[0]} | Walidacja: {X_val.shape[0]}")

# Budowa modelu CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Zapis modelu
model.save('gtsrb_model.h5')
print("💾 Model zapisany jako gtsrb_model.h5")
