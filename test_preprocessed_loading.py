"""
Test Preprocessed Image Loading
================================
Check if we can load preprocessed images without errors
"""

import os
import cv2
import numpy as np

print("="*70)
print("TESTING PREPROCESSED IMAGE LOADING")
print("="*70)
print()

preprocessed_dir = 'processed_datasets/eyepacs_train'

# Test loading a few images
rg_path = os.path.join(preprocessed_dir, 'RG')
nrg_path = os.path.join(preprocessed_dir, 'NRG')

if not os.path.exists(rg_path):
    print(f"[ERROR] RG folder not found: {rg_path}")
    exit(1)

if not os.path.exists(nrg_path):
    print(f"[ERROR] NRG folder not found: {nrg_path}")
    exit(1)

print("Testing image loading...")
print()

# Try loading a few images
rg_files = [f for f in os.listdir(rg_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
nrg_files = [f for f in os.listdir(nrg_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]

print(f"Testing {len(rg_files)} RG images...")
for img_file in rg_files:
    img_path = os.path.join(rg_path, img_file)
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERROR] Cannot read: {img_file}")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            print(f"  [OK] {img_file}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
    except Exception as e:
        print(f"  [ERROR] {img_file}: {e}")

print()
print(f"Testing {len(nrg_files)} NRG images...")
for img_file in nrg_files:
    img_path = os.path.join(nrg_path, img_file)
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERROR] Cannot read: {img_file}")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            print(f"  [OK] {img_file}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
    except Exception as e:
        print(f"  [ERROR] {img_file}: {e}")

print()
print("="*70)
print("If all images loaded OK, training should work")
print("="*70)

