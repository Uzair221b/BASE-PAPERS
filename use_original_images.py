"""
Use Original Images - Not Preprocessed
=======================================
The preprocessed images might have lost important features
Try using original raw images with minimal preprocessing
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
print("USING ORIGINAL RAW IMAGES")
print("="*70)
print("Preprocessed images might have lost important features")
print("Trying original images with minimal preprocessing")
print("="*70)
print()

# Use original images
original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'

def load_original_images(data_dir, class_name, label, max_samples=None):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"   Loading {len(image_files)} images from {class_name}...")
    
    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)
        try:
            # Load as RGB
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to 224x224 (minimal preprocessing)
                img = cv2.resize(img, (224, 224))
                
                # Normalize to 0-1
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(label)
        except Exception as e:
            continue
    
    return images, labels

# Load small test sample
print("Loading original images (small test sample)...")
rg_images, rg_labels = load_original_images(original_dir, 'RG', 1, max_samples=200)
nrg_images, nrg_labels = load_original_images(original_dir, 'NRG', 0, max_samples=200)

if len(rg_images) == 0 or len(nrg_images) == 0:
    print("[ERROR] No images loaded!")
    exit(1)

X = np.array(rg_images + nrg_images)
y = np.array(rg_labels + nrg_labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTest sample: {len(X_train)} train, {len(X_val)} val")
print(f"Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print()

# Build model
print("Building model...")
base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
# Use Rescaling since original images are 0-255
x = layers.Rescaling(1./255)(inputs)
x = layers.RandomFlip("horizontal")(x)
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

if final_acc > 0.60:
    print("\n[SUCCESS] Model is learning with original images!")
    print("The preprocessed images were the problem.")
    print("We should use original images for training.")
elif final_acc > 0.55:
    print("\n[PARTIAL SUCCESS] Some improvement, but still low.")
    print("Might need more epochs or different approach.")
else:
    print("\n[FAILED] Still not learning properly.")
    print("There might be a deeper issue with the data or labels.")

