"""
Rebuild Model Fresh - Fix the 50% Accuracy Problem
===================================================
The saved model is broken (always predicts 0.528). Rebuild from scratch.
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
from datetime import datetime

print("="*70)
print("REBUILDING MODEL FROM SCRATCH - FIXING 50% ACCURACY")
print("="*70)
print()
print("Problem: Saved model always predicts 0.528 (broken)")
print("Solution: Rebuild model from scratch with correct architecture")
print()
print("="*70)
print()

# Step 1: Build fresh model
print("Step 1: Building fresh model...")
input_size = (224, 224, 3)

# Load pre-trained EfficientNetB4
base = EfficientNetB4(
    weights='imagenet',
    include_top=False,
    input_shape=input_size
)

# Freeze base for Phase 1
base.trainable = False

# Build model
inputs = keras.Input(shape=input_size)

# Data augmentation
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.025)(x)
x = layers.RandomZoom(0.035)(x)
x = layers.Rescaling(1./255)(x)

# Base model
x = base(x, training=False)

# Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)

# Output
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"[OK] Model built!")
print(f"   Parameters: {model.count_params():,}")
print(f"   Trainable: {sum([1 for l in model.layers if l.trainable])}/{len(model.layers)}")
print()

# Step 2: Load data
print("Step 2: Loading data...")
train_dir = 'processed_datasets/eyepacs_train'

def load_images(data_dir, class_name, label):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"   Loading {len(image_files)} images from {class_name}...")
    
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

rg_images, rg_labels = load_images(train_dir, 'RG', 1)
nrg_images, nrg_labels = load_images(train_dir, 'NRG', 0)

X = np.array(rg_images + nrg_images)
y = np.array(rg_labels + nrg_labels)

print(f"[OK] Loaded {len(X)} images")
print(f"   RG: {np.sum(y==1)}, NRG: {np.sum(y==0)}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
print()

# Step 3: Test model works
print("Step 3: Testing model predictions...")
test_pred = model.predict(X_val[:10], verbose=0)
print(f"   Predictions: {test_pred.flatten()[:5]}")
print(f"   Range: [{test_pred.min():.3f}, {test_pred.max():.3f}]")

if test_pred.min() == test_pred.max():
    print("   [ERROR] All predictions are the same - model is broken!")
    exit(1)
else:
    print("   [OK] Model produces different predictions")
print()

# Step 4: Setup callbacks
print("Step 4: Setting up training...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('models', exist_ok=True)

callbacks = [
    ModelCheckpoint(
        f'models/best_model_fresh_{timestamp}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Step 5: Train
print("="*70)
print("STARTING FRESH TRAINING")
print("="*70)
print("Phase 1: Frozen base (50 epochs)")
print("Batch size: 16")
print("="*70)
print()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Results
best_acc = max(history.history['val_accuracy'])
print()
print("="*70)
print("PHASE 1 COMPLETE!")
print("="*70)
print(f"Best validation accuracy: {best_acc*100:.2f}%")

if best_acc > 0.50:
    print(f"[OK] Model is learning! Accuracy: {best_acc*100:.2f}%")
else:
    print(f"[WARNING] Accuracy still low: {best_acc*100:.2f}%")

print(f"Model saved: models/best_model_fresh_{timestamp}.h5")
print("="*70)

