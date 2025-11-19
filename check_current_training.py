"""
Check Current Training Status
==============================
See if accuracy is increasing and if we can reach 99%
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import cv2
from sklearn.model_selection import train_test_split

print("="*70)
print("CHECKING CURRENT TRAINING STATUS")
print("="*70)
print()

# Find latest model
model_files = [f for f in os.listdir('models') if f.startswith('best_cnn_model_') and f.endswith('.h5')]
if not model_files:
    print("No model file found yet. Training might still be starting...")
    exit(0)

latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join('models', x)))
model_path = os.path.join('models', latest_model)

print(f"Latest model: {latest_model}")
print(f"Last updated: {os.path.getmtime(model_path)}")
print()

# Load model
try:
    model = keras.models.load_model(model_path)
    print("[OK] Model loaded successfully")
    print()
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit(1)

# Load test data
original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'

def load_sample(data_dir, class_name, label, max_samples=100):
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

print("Loading test sample...")
rg_images, rg_labels = load_sample(original_dir, 'RG', 1, max_samples=100)
nrg_images, nrg_labels = load_sample(original_dir, 'NRG', 0, max_samples=100)

X_test = np.array(rg_images + nrg_images)
y_test = np.array(rg_labels + nrg_labels)

# Test predictions
predictions = model.predict(X_test, verbose=0)
pred_classes = (predictions > 0.5).astype(int).flatten()
accuracy = np.mean(pred_classes == y_test)

print(f"Current accuracy on test sample: {accuracy*100:.2f}%")
print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
print()

# Analysis
print("="*70)
print("ANALYSIS")
print("="*70)

if accuracy >= 0.99:
    print("[SUCCESS] Already at 99%+ accuracy!")
    print("Training can stop - goal achieved!")
elif accuracy >= 0.90:
    print("[GOOD] Accuracy is high (>90%)")
    print("Model is learning well. Continue training to reach 99%.")
elif accuracy >= 0.70:
    print("[OK] Accuracy is improving (>70%)")
    print("Model is learning. Continue training.")
elif accuracy >= 0.60:
    print("[WARNING] Accuracy is moderate (>60%)")
    print("Model is learning but slowly. May need more epochs.")
elif accuracy >= 0.55:
    print("[WARNING] Accuracy is low (>55%)")
    print("Model is learning but very slowly. May not reach 99%.")
else:
    print("[PROBLEM] Accuracy is very low (<55%)")
    print("Model might not be learning properly.")

print()
print("About 200 epochs:")
print("  - EarlyStopping will stop automatically if accuracy stops improving")
print("  - It won't waste time if not learning")
print("  - We can resume from checkpoint if needed")
print("="*70)

