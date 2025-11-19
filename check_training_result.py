"""
Check Training Result
=====================
See what actually happened
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import cv2
from sklearn.model_selection import train_test_split

print("="*70)
print("CHECKING TRAINING RESULT")
print("="*70)
print()

# Find latest model
model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
if not model_files:
    print("No model files found!")
    exit(1)

latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join('models', x)))
model_path = os.path.join('models', latest_model)

print(f"Checking model: {latest_model}")
print()

# Load model
model = keras.models.load_model(model_path)

# Load test data
train_dir = 'processed_datasets/eyepacs_train'

def load_sample(data_dir, class_name, label, max_samples=100):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
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

rg_images, rg_labels = load_sample(train_dir, 'RG', 1, max_samples=100)
nrg_images, nrg_labels = load_sample(train_dir, 'NRG', 0, max_samples=100)

X_test = np.array(rg_images + nrg_images)
y_test = np.array(rg_labels + nrg_labels)

# Test predictions
predictions = model.predict(X_test, verbose=0)
pred_classes = (predictions > 0.5).astype(int).flatten()
accuracy = np.mean(pred_classes == y_test)

print(f"Test accuracy: {accuracy*100:.2f}%")
print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
print(f"Predictions mean: {predictions.mean():.3f}")

unique_preds = len(np.unique(predictions))
print(f"Unique prediction values: {unique_preds}")

if accuracy <= 0.50:
    print("\n[PROBLEM] Accuracy is 50% or below!")
    if unique_preds == 1:
        print("   Model always predicts the same value - BROKEN")
    elif predictions.mean() > 0.45 and predictions.mean() < 0.55:
        print("   Model predictions are all around 0.5 - NOT LEARNING")
    else:
        print("   Model is making predictions but they're wrong")
else:
    print(f"\n[OK] Accuracy is above 50%: {accuracy*100:.2f}%")

