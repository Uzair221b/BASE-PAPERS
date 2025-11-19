"""
Test Simple Model - Can Learning Happen?
=========================================
Use a very simple model to test if learning is possible
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from datetime import datetime

print("="*70)
print("TESTING WITH SIMPLE MODEL")
print("="*70)
print("If this simple model can learn, then the problem is with the complex model")
print("If it can't learn, then there's a fundamental data/label issue")
print("="*70)
print()

# Load data
original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'

def load_images(data_dir, class_name, label, max_samples=200):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_samples]
    
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
rg_images, rg_labels = load_images(original_dir, 'RG', 1, max_samples=200)
nrg_images, nrg_labels = load_images(original_dir, 'NRG', 0, max_samples=200)

X = np.array(rg_images + nrg_images)
y = np.array(rg_labels + nrg_labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Loaded: {len(X_train)} train, {len(X_val)} val")
print(f"Labels: {np.sum(y_train==1)} RG, {np.sum(y_train==0)} NRG (train)")
print(f"Labels: {np.sum(y_val==1)} RG, {np.sum(y_val==0)} NRG (val)")
print()

# Build VERY SIMPLE model - just flatten and dense layers
print("Building SIMPLE model (no CNN, just dense layers)...")
model = models.Sequential([
    layers.Flatten(input_shape=(224, 224, 3)),  # Flatten image
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
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

# Train for 20 epochs
print("Training for 20 epochs...")
print("="*70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
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

if final_acc > 0.60:
    print("[SUCCESS] Simple model CAN learn!")
    print("The problem is with the EfficientNet model or training setup.")
    print("We should fix the complex model architecture.")
elif final_acc > 0.55:
    print("[PARTIAL] Simple model shows some learning.")
    print("The data might be difficult, but learning is possible.")
    print("Need to adjust the complex model.")
elif final_acc > 0.50:
    print("[WARNING] Simple model barely learns.")
    print("The data might be very difficult to distinguish.")
    print("Need to investigate data quality or use different approach.")
else:
    print("[FAILED] Simple model CANNOT learn.")
    print("This suggests a fundamental problem:")
    print("  - Labels might be wrong")
    print("  - Images might not be distinguishable")
    print("  - Data might be corrupted")
    print("Need to investigate the dataset itself.")

print("="*70)

