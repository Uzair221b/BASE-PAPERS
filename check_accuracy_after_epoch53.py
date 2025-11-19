"""
Check Accuracy After Epoch 53
=============================
Check if training stopped and what the actual accuracy is
"""

import numpy as np
import os
import cv2
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

print("="*70)
print("CHECKING ACCURACY AFTER EPOCH 53")
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
print("CURRENT ACCURACY RESULTS")
print("="*70)
print(f"Overall Accuracy: {current_accuracy*100:.2f}%")
print(f"  RG (Glaucoma): {rg_acc*100:.2f}%")
print(f"  NRG (Normal): {nrg_acc*100:.2f}%")
print()

# Check if 0.90188 = 90.188%
val_acc_90188 = 0.90188
print(f"Validation accuracy 0.90188 = {val_acc_90188*100:.2f}%")
print()

print("="*70)
print("ANALYSIS")
print("="*70)

if current_accuracy >= 0.99:
    print("[SUCCESS] 99%+ accuracy achieved!")
    print("Training goal reached!")
elif current_accuracy >= 0.95:
    print(f"[EXCELLENT] Accuracy is {current_accuracy*100:.2f}%")
    print("Very close to 99%! Training should continue.")
elif current_accuracy >= 0.90:
    print(f"[GOOD] Accuracy is {current_accuracy*100:.2f}%")
    print("Good progress, but NOT 99% yet.")
    print("Training should continue to reach 99%.")
else:
    print(f"[OK] Accuracy is {current_accuracy*100:.2f}%")
    print("Training should continue.")

print()
print("About 'accuracy did not improve from 0.90188':")
print("  - This means validation accuracy was 90.188%")
print("  - This is NOT 99% - still need 8.81% more")
print("  - EarlyStopping might have triggered if no improvement for 20 epochs")
print("  - But training can still continue if it improves again")
print()

# Check if training is still running
print("="*70)
print("TRAINING STATUS")
print("="*70)

if time_since_update < 30:  # Updated in last 30 minutes
    print("[RUNNING] Training appears to be running")
    print(f"Model was updated {time_since_update:.1f} minutes ago")
    print("Training is still active and may continue improving")
elif time_since_update < 60:
    print("[POSSIBLY STOPPED] Model not updated recently")
    print(f"Last update: {time_since_update:.1f} minutes ago")
    print("Training might have stopped due to EarlyStopping")
else:
    print("[STOPPED] Training appears to have stopped")
    print(f"Last update: {time_since_update:.1f} minutes ago")
    print("EarlyStopping likely triggered")

print()
print("="*70)
print("RECOMMENDATION")
print("="*70)

if current_accuracy < 0.99:
    print(f"Current: {current_accuracy*100:.2f}% (NOT 99% yet)")
    print("Need: {:.2f}% more to reach 99%".format((0.99 - current_accuracy)*100))
    print()
    print("Options:")
    print("  1. If training stopped: Restart with more patience or continue training")
    print("  2. If training running: Let it continue - it may still improve")
    print("  3. Current accuracy is good but not 99% - need more training")
else:
    print("[SUCCESS] 99%+ accuracy achieved!")

print("="*70)

