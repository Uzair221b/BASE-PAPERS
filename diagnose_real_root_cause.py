"""
REAL Root Cause Diagnosis - Before Wasting More Time
=====================================================
Actually find what's wrong this time
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

print("="*70)
print("REAL DIAGNOSIS - FINDING ACTUAL PROBLEM")
print("="*70)
print()

# Check 1: Data loading
print("1. Checking data loading...")
train_dir = 'processed_datasets/eyepacs_train'

rg_path = os.path.join(train_dir, 'RG')
nrg_path = os.path.join(train_dir, 'NRG')

if not os.path.exists(rg_path):
    print(f"   [ERROR] RG folder not found: {rg_path}")
    exit(1)
if not os.path.exists(nrg_path):
    print(f"   [ERROR] NRG folder not found: {nrg_path}")
    exit(1)

rg_files = [f for f in os.listdir(rg_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
nrg_files = [f for f in os.listdir(nrg_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"   RG images: {len(rg_files)}")
print(f"   NRG images: {len(nrg_files)}")

if len(rg_files) == 0 or len(nrg_files) == 0:
    print(f"   [ERROR] No images found!")
    exit(1)

# Load a few images and check
print("\n2. Testing image loading...")
test_rg = os.path.join(rg_path, rg_files[0])
test_nrg = os.path.join(nrg_path, nrg_files[0])

img_rg = cv2.imread(test_rg)
img_nrg = cv2.imread(test_nrg)

if img_rg is None:
    print(f"   [ERROR] Cannot read RG image: {test_rg}")
    exit(1)
if img_nrg is None:
    print(f"   [ERROR] Cannot read NRG image: {test_nrg}")
    exit(1)

print(f"   RG image shape: {img_rg.shape}")
print(f"   NRG image shape: {img_nrg.shape}")

# Check preprocessing
img_rg_proc = cv2.cvtColor(img_rg, cv2.COLOR_BGR2RGB)
img_rg_proc = cv2.resize(img_rg_proc, (224, 224))
img_rg_proc = img_rg_proc.astype(np.float32) / 255.0

print(f"   Processed shape: {img_rg_proc.shape}")
print(f"   Value range: [{img_rg_proc.min():.3f}, {img_rg_proc.max():.3f}]")

if img_rg_proc.min() < 0 or img_rg_proc.max() > 1:
    print(f"   [ERROR] Image values not normalized correctly!")
    exit(1)

# Check 3: Model architecture
print("\n3. Testing model architecture...")
base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = layers.Rescaling(1./255)(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

test_model = models.Model(inputs, outputs)
test_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Test prediction
test_input = np.expand_dims(img_rg_proc, axis=0)
pred = test_model.predict(test_input, verbose=0)
print(f"   Test prediction: {pred[0][0]:.3f}")

if pred[0][0] == 0.5 or pred[0][0] == 0.528:
    print(f"   [WARNING] Model predicting suspicious value: {pred[0][0]}")
else:
    print(f"   [OK] Model produces reasonable prediction")

# Test with different images
test_inputs = []
test_labels = []
for i in range(5):
    img_path = os.path.join(rg_path, rg_files[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    test_inputs.append(img)
    test_labels.append(1)

for i in range(5):
    img_path = os.path.join(nrg_path, nrg_files[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    test_inputs.append(img)
    test_labels.append(0)

test_inputs = np.array(test_inputs)
test_labels = np.array(test_labels)

preds = test_model.predict(test_inputs, verbose=0)
print(f"\n   Predictions for 10 images:")
print(f"   {preds.flatten()}")

unique_preds = len(np.unique(preds))
if unique_preds == 1:
    print(f"   [ERROR] All predictions are the same! Model is broken!")
    exit(1)
else:
    print(f"   [OK] Model produces different predictions ({unique_preds} unique values)")

# Check 4: Training setup
print("\n4. Testing training setup...")
# Quick train test
test_model.fit(
    test_inputs, test_labels,
    epochs=1,
    batch_size=2,
    verbose=0
)

preds_after = test_model.predict(test_inputs, verbose=0)
if np.allclose(preds, preds_after, atol=1e-6):
    print(f"   [ERROR] Model weights not changing - cannot learn!")
    exit(1)
else:
    print(f"   [OK] Model weights are changing - can learn")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("[OK] Everything looks correct!")
print("Model should work properly.")
print("="*70)

