"""
Check Current Accuracy (Without Interrupting Training)
=======================================================
Test the latest model checkpoint to see current accuracy
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

print("="*70)
print("CHECKING CURRENT ACCURACY (Original Images Training)")
print("="*70)
print("Testing latest model checkpoint without interrupting training")
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
print(f"Testing model: {latest_model}")
print(f"Last updated: {mod_time}")
print(f"Time since update: {(datetime.now() - mod_time).total_seconds()/60:.1f} minutes ago")
print()

# Load model
print("Loading model...")
try:
    model = keras.models.load_model(model_path)
    print("[OK] Model loaded successfully")
    print()
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit(1)

# Load test data (same as training - original images)
original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'

def load_sample(data_dir, class_name, label, max_samples=500):
    """Load sample for testing"""
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

print("Loading test sample (500 images per class)...")
rg_images, rg_labels = load_sample(original_dir, 'RG', 1, max_samples=500)
nrg_images, nrg_labels = load_sample(original_dir, 'NRG', 0, max_samples=500)

X_test = np.array(rg_images + nrg_images)
y_test = np.array(rg_labels + nrg_labels)

print(f"Test sample: {len(X_test)} images ({len(rg_images)} RG, {len(nrg_images)} NRG)")
print()

# Test predictions
print("Testing model predictions...")
predictions = model.predict(X_test, verbose=0)
pred_classes = (predictions > 0.5).astype(int).flatten()
accuracy = np.mean(pred_classes == y_test)

# Calculate per-class accuracy
rg_mask = y_test == 1
nrg_mask = y_test == 0
rg_accuracy = np.mean(pred_classes[rg_mask] == y_test[rg_mask])
nrg_accuracy = np.mean(pred_classes[nrg_mask] == y_test[nrg_mask])

print()
print("="*70)
print("CURRENT ACCURACY RESULTS")
print("="*70)
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"  RG (Glaucoma) Accuracy: {rg_accuracy*100:.2f}%")
print(f"  NRG (Normal) Accuracy: {nrg_accuracy*100:.2f}%")
print()
print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
print(f"Predictions mean: {predictions.mean():.3f}")
print()

# Analysis
print("="*70)
print("ANALYSIS")
print("="*70)

if accuracy >= 0.99:
    print("[SUCCESS] Reached 99%+ accuracy!")
    print("Training goal achieved!")
elif accuracy >= 0.95:
    print(f"[EXCELLENT] Accuracy is {accuracy*100:.2f}% - very close to 99%!")
    print("Model is performing very well!")
elif accuracy >= 0.90:
    print(f"[GOOD] Accuracy is {accuracy*100:.2f}% - making great progress!")
    print("Model is learning well.")
elif accuracy >= 0.85:
    print(f"[OK] Accuracy is {accuracy*100:.2f}% - good progress!")
    print("Model is improving.")
elif accuracy >= 0.80:
    print(f"[OK] Accuracy is {accuracy*100:.2f}% - steady improvement!")
    print("Model is learning.")
else:
    print(f"[WARNING] Accuracy is {accuracy*100:.2f}% - still learning.")
    print("Model needs more training.")

print()
print(f"Model last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Training is still running - accuracy may improve further!")
print("="*70)

