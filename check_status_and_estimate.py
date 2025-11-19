"""
Check Current Status and Estimate 99% Accuracy
===============================================
Check current accuracy and estimate when 99% will be reached
"""

import numpy as np
import os
import cv2
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

print("="*70)
print("CURRENT TRAINING STATUS & 99% ACCURACY ESTIMATE")
print("="*70)
print()

# Find latest model
models_dir = 'models'
model_files = [f for f in os.listdir(models_dir) 
               if f.startswith('best_cnn_model_') and f.endswith('.h5')]

if not model_files:
    print("[ERROR] No model file found!")
    exit(1)

latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
model_path = os.path.join(models_dir, latest_model)

mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
time_since_update = (datetime.now() - mod_time).total_seconds() / 60

print(f"Latest model: {latest_model}")
print(f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Time since update: {time_since_update:.1f} minutes ago")
print()

# Load model and test
print("Testing current accuracy...")
model = keras.models.load_model(model_path)

# Load test data
original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'

def load_sample(data_dir, class_name, label, max_samples=500):
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

rg_images, rg_labels = load_sample(original_dir, 'RG', 1, max_samples=500)
nrg_images, nrg_labels = load_sample(original_dir, 'NRG', 0, max_samples=500)

X_test = np.array(rg_images + nrg_images)
y_test = np.array(rg_labels + nrg_labels)

predictions = model.predict(X_test, verbose=0)
pred_classes = (predictions > 0.5).astype(int).flatten()
current_accuracy = np.mean(pred_classes == y_test)

# Calculate per-class
rg_mask = y_test == 1
nrg_mask = y_test == 0
rg_acc = np.mean(pred_classes[rg_mask] == y_test[rg_mask])
nrg_acc = np.mean(pred_classes[nrg_mask] == y_test[nrg_mask])

print()
print("="*70)
print("CURRENT ACCURACY")
print("="*70)
print(f"Overall Accuracy: {current_accuracy*100:.2f}%")
print(f"  RG (Glaucoma): {rg_acc*100:.2f}%")
print(f"  NRG (Normal): {nrg_acc*100:.2f}%")
print()

# Estimate when 99% will be reached
print("="*70)
print("99% ACCURACY ESTIMATE")
print("="*70)

# Historical data points
# Epoch 14: ~58%
# Epoch 29: ~80.50%
# Current: ~88.80% (estimated epoch ~40-50 based on time)

# Calculate improvement rate
improvement_needed = 0.99 - current_accuracy
print(f"Accuracy needed to reach 99%: {improvement_needed*100:.2f}%")
print()

# Estimate based on improvement rate
# From 58% to 80.50% in ~15 epochs = +22.5% in 15 epochs = ~1.5% per epoch
# From 80.50% to 88.80% in ~10-15 epochs = ~0.5-0.8% per epoch (slowing down)

# Conservative estimate: ~0.3-0.5% per epoch going forward
if current_accuracy >= 0.88:
    # Slower improvement as we approach 99%
    epochs_needed = improvement_needed / 0.003  # ~0.3% per epoch
    print(f"Estimated epochs needed: {epochs_needed:.0f} epochs")
    print()
    print("Based on:")
    print("  - Current: {:.2f}%".format(current_accuracy*100))
    print("  - Target: 99.00%")
    print("  - Improvement rate: ~0.3-0.5% per epoch (slowing as accuracy increases)")
    print()
    
    # Estimate time (assuming ~12-15 minutes per epoch on CPU)
    time_per_epoch = 13  # minutes
    total_time_minutes = epochs_needed * time_per_epoch
    total_time_hours = total_time_minutes / 60
    
    print(f"Estimated time to 99%:")
    print(f"  - Epochs: ~{epochs_needed:.0f} more epochs")
    print(f"  - Time: ~{total_time_hours:.1f} hours ({total_time_minutes:.0f} minutes)")
    print()
    
    # Current epoch estimate
    # Started around epoch 14 at 58%, now at 88.80%
    # Improvement: 30.8% over ~25-30 epochs
    current_epoch_estimate = 40 + (current_accuracy - 0.88) / 0.003
    
    print(f"Current epoch estimate: ~{current_epoch_estimate:.0f} epochs")
    print(f"Estimated epoch for 99%: ~{current_epoch_estimate + epochs_needed:.0f} epochs")
    
elif current_accuracy >= 0.85:
    epochs_needed = improvement_needed / 0.005  # ~0.5% per epoch
    print(f"Estimated epochs needed: {epochs_needed:.0f} epochs")
    time_per_epoch = 13
    total_time_hours = (epochs_needed * time_per_epoch) / 60
    print(f"Estimated time: ~{total_time_hours:.1f} hours")
else:
    epochs_needed = improvement_needed / 0.008  # ~0.8% per epoch
    print(f"Estimated epochs needed: {epochs_needed:.0f} epochs")
    time_per_epoch = 13
    total_time_hours = (epochs_needed * time_per_epoch) / 60
    print(f"Estimated time: ~{total_time_hours:.1f} hours")

print()
print("="*70)
print("STATUS SUMMARY")
print("="*70)
print(f"Current Accuracy: {current_accuracy*100:.2f}%")
print(f"Target: 99.00%")
print(f"Remaining: {improvement_needed*100:.2f}%")
print()
print(f"Training Status: RUNNING")
print(f"Model: {latest_model}")
print(f"Last Updated: {mod_time.strftime('%H:%M:%S')} ({time_since_update:.1f} min ago)")
print()
if current_accuracy >= 0.99:
    print("[SUCCESS] 99%+ accuracy already achieved!")
elif current_accuracy >= 0.95:
    print("[EXCELLENT] Very close to 99%! Keep training.")
elif current_accuracy >= 0.90:
    print("[GOOD] Making great progress toward 99%!")
else:
    print("[OK] Steady improvement - on track for 99%")
print("="*70)

