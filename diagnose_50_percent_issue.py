"""
Diagnose 50% Accuracy Issue
============================
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

print("="*70)
print("DIAGNOSING 50% ACCURACY ISSUE")
print("="*70)
print()

# Step 1: Check data loading
print("Step 1: Checking data loading...")
train_dir = 'processed_datasets/eyepacs_train'

def load_images_simple(data_dir, class_name, label):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files[:10]:  # Just check first 10
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

rg_images, rg_labels = load_images_simple(train_dir, 'RG', 1)
nrg_images, nrg_labels = load_images_simple(train_dir, 'NRG', 0)

print(f"  RG samples: {len(rg_images)} (label: {rg_labels[0] if rg_labels else 'N/A'})")
print(f"  NRG samples: {len(nrg_images)} (label: {nrg_labels[0] if nrg_labels else 'N/A'})")
print(f"  Labels are: RG=1, NRG=0")
print()

# Step 2: Check model
print("Step 2: Checking model...")
model_path = 'models/best_model_20251110_193527.h5'
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
    print(f"  Model loaded: {model_path}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Output activation: {model.layers[-1].activation}")
    print(f"  Loss function: {model.loss}")
    print()
    
    # Check if model expects sparse or categorical
    if len(model.output_shape) == 2 and model.output_shape[1] == 2:
        print("  [ISSUE FOUND] Model outputs 2 classes (one-hot)")
        print("  But labels are binary (0/1)")
        print("  Need to use sparse_categorical_crossentropy or convert labels!")
    else:
        print("  Model output format seems OK")
else:
    print("  Model file not found!")

print()

# Step 3: Check label format
print("Step 3: Checking label format...")
print(f"  Labels loaded: {rg_labels[:5] if rg_labels else []} (RG)")
print(f"  Labels loaded: {nrg_labels[:5] if nrg_labels else []} (NRG)")
print()

# Step 4: Summary
print("="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)
print()
print("Possible issues:")
print("  1. Model expects one-hot labels but getting binary labels")
print("  2. Loss function mismatch")
print("  3. Model not actually training (frozen layers issue)")
print()
print("Next: Fix the label format issue")

