"""
Try Simple CNN Model
====================
Since simple dense model showed learning (55%), try a simple CNN
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

print("="*70)
print("TRYING SIMPLE CNN MODEL")
print("="*70)
print("Simple dense model reached 55% - learning is possible!")
print("Trying a simple CNN to see if it can do better")
print("="*70)
print()

# Load data
original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'

def load_images(data_dir, class_name, label, max_samples=None):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(label)
        except:
            continue
    
    return images, labels

print("Loading images...")
rg_images, rg_labels = load_images(original_dir, 'RG', 1, max_samples=400)
nrg_images, nrg_labels = load_images(original_dir, 'NRG', 0, max_samples=400)

X = np.array(rg_images + nrg_images)
y = np.array(rg_labels + nrg_labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Loaded: {len(X_train)} train, {len(X_val)} val")
print()

# Build simple CNN
print("Building simple CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Model parameters: {model.count_params():,}")
print()

# Test initial
initial_pred = model.predict(X_val, verbose=0)
initial_acc = np.mean((initial_pred > 0.5).astype(int).flatten() == y_val)
print(f"Initial accuracy: {initial_acc*100:.2f}%")
print()

# Train
print("Training for 30 epochs...")
print("="*70)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Results
final_acc = max(history.history['val_accuracy'])
best_train_acc = max(history.history['accuracy'])

print()
print("="*70)
print("RESULTS")
print("="*70)
print(f"Initial accuracy: {initial_acc*100:.2f}%")
print(f"Best training accuracy: {best_train_acc*100:.2f}%")
print(f"Best validation accuracy: {final_acc*100:.2f}%")
print(f"Improvement: {(final_acc - initial_acc)*100:.2f}%")
print()

if final_acc > 0.70:
    print("[SUCCESS] Simple CNN works well!")
    print("We should use this approach instead of EfficientNet.")
elif final_acc > 0.60:
    print("[GOOD] Simple CNN shows good learning!")
    print("This is a viable approach - can improve with more training.")
elif final_acc > 0.55:
    print("[OK] Simple CNN shows some learning.")
    print("Better than simple dense model - CNN helps.")
else:
    print("[WARNING] Simple CNN not much better.")
    print("Need to investigate further.")

print("="*70)

