"""
Quick Test - Check if Training Will Work
=========================================
Test with a small sample to see if accuracy improves
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("="*70)
print("QUICK TEST - WILL TRAINING WORK?")
print("="*70)
print()
print("Testing with small sample to see if accuracy improves...")
print()

# Load model
model_path = 'models/best_model_20251110_193527.h5'
model = keras.models.load_model(model_path)

# Freeze base for Phase 1
for layer in model.layers:
    if 'efficientnet' in layer.name.lower():
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Load small sample
train_dir = 'processed_datasets/eyepacs_train'

def load_sample(data_dir, class_name, label, max_samples=200):
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

print("Loading small sample (200 images per class)...")
rg_images, rg_labels = load_sample(train_dir, 'RG', 1, max_samples=200)
nrg_images, nrg_labels = load_sample(train_dir, 'NRG', 0, max_samples=200)

X = np.array(rg_images + nrg_images)
y = np.array(rg_labels + nrg_labels)

print(f"Loaded: {len(X)} images ({len(rg_images)} RG, {len(nrg_images)} NRG)")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")
print()

# Test initial accuracy
print("Testing initial accuracy...")
initial_pred = model.predict(X_val[:50], verbose=0)
if initial_pred.shape[1] == 1:
    initial_pred_classes = (initial_pred > 0.5).astype(int).flatten()
    initial_acc = np.mean(initial_pred_classes == y_val[:50])
else:
    initial_pred_classes = np.argmax(initial_pred, axis=1)
    initial_acc = np.mean(initial_pred_classes == y_val[:50])

print(f"Initial validation accuracy: {initial_acc*100:.2f}%")
print()

# Train for 3 epochs
print("Training for 3 epochs (quick test)...")
print("="*70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=3,
    batch_size=16,
    verbose=1
)

# Check final accuracy
final_acc = max(history.history['val_accuracy'])
print()
print("="*70)
print("TEST RESULTS")
print("="*70)
print(f"Initial accuracy: {initial_acc*100:.2f}%")
print(f"Final accuracy: {final_acc*100:.2f}%")

if final_acc > initial_acc:
    improvement = (final_acc - initial_acc) * 100
    print(f"Improvement: +{improvement:.2f}%")
    print()
    print("[OK] Training is working! Accuracy is improving!")
    print("The full training should work correctly.")
elif final_acc == 0.5:
    print()
    print("[WARNING] Accuracy stuck at 50% - Model not learning!")
    print("There's a problem with the training setup.")
    print("Need to investigate before running full training.")
else:
    print()
    print("[INFO] Accuracy changed, but need to monitor more.")

print("="*70)

