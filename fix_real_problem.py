"""
Fix the Real Problem
====================
The issue: Model not learning because of data/preprocessing mismatch
Solution: Use original images OR fix preprocessing pipeline
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
from datetime import datetime

print("="*70)
print("FIXING THE REAL PROBLEM")
print("="*70)
print()
print("Problem: Model not learning (predictions stuck at ~0.5)")
print("Root cause: Need to investigate data/preprocessing")
print()
print("="*70)
print()

# Check what data we have
print("Checking available data...")
processed_dir = 'processed_datasets/eyepacs_train'
original_dir = 'EYEPACS(AIROGS)'

if os.path.exists(processed_dir):
    print(f"[OK] Processed images: {processed_dir}")
    rg_processed = os.path.join(processed_dir, 'RG')
    if os.path.exists(rg_processed):
        processed_count = len([f for f in os.listdir(rg_processed) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   Processed RG images: {processed_count}")

if os.path.exists(original_dir):
    print(f"[OK] Original images: {original_dir}")
else:
    print(f"[INFO] Original images folder not found")

print()

# The REAL fix: Use simpler approach - load processed images correctly
# AND make sure model can actually learn

print("Solution: Train with proper data loading and verify learning")
print()

def load_images_correct(data_dir, class_name, label):
    """Load images correctly - they're already preprocessed"""
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)
        try:
            # Load as RGB
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if img.shape[:2] != (224, 224):
                    img = cv2.resize(img, (224, 224))
                
                # Normalize to 0-1 (images are saved as 0-255)
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(label)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue
    
    return images, labels

# Load small sample for testing
print("Loading small test sample...")
rg_images, rg_labels = load_images_correct(processed_dir, 'RG', 1)
nrg_images, nrg_labels = load_images_correct(processed_dir, 'NRG', 0)

if len(rg_images) == 0 or len(nrg_images) == 0:
    print("[ERROR] No images loaded!")
    exit(1)

# Use smaller sample for quick test
X = np.array((rg_images[:200] + nrg_images[:200]))
y = np.array((rg_labels[:200] + nrg_labels[:200]))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test sample: {len(X_train)} train, {len(X_val)} val")
print(f"Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print()

# Build model WITHOUT Rescaling layer (data already normalized)
print("Building model (without Rescaling - data already normalized)...")
base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
# NO Rescaling - data is already 0-1
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.025)(x)
x = layers.RandomZoom(0.035)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Test initial
initial_pred = model.predict(X_val, verbose=0)
initial_acc = np.mean((initial_pred > 0.5).astype(int).flatten() == y_val)
print(f"Initial accuracy: {initial_acc*100:.2f}%")

# Train for 10 epochs
print("\nTraining for 10 epochs...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16,
    verbose=1
)

final_acc = max(history.history['val_accuracy'])
print(f"\nFinal accuracy: {final_acc*100:.2f}%")
print(f"Improvement: {(final_acc - initial_acc)*100:.2f}%")

if final_acc > 0.55:
    print("\n[SUCCESS] Model is learning! Accuracy improved!")
    print("Now we can train on full dataset.")
else:
    print("\n[FAILED] Model still not learning properly.")
    print("Need to investigate further - might need different approach.")

