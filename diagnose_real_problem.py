"""
Real Diagnosis - Why is Accuracy Stuck at 50%?
===============================================
Actually find the root cause
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

print("="*70)
print("REAL DIAGNOSIS - WHY 50% ACCURACY?")
print("="*70)
print()

# Load model
model_path = 'models/best_model_20251110_193527.h5'
model = keras.models.load_model(model_path)

print("1. Checking model architecture...")
print(f"   Output shape: {model.output_shape}")
print(f"   Output activation: {model.layers[-1].activation}")
print(f"   Loss: {model.loss}")
print()

# Check if model is actually trainable
print("2. Checking trainable layers...")
trainable_layers = [l for l in model.layers if l.trainable]
frozen_layers = [l for l in model.layers if not l.trainable]
print(f"   Trainable: {len(trainable_layers)}")
print(f"   Frozen: {len(frozen_layers)}")

# Check EfficientNet layers
efficientnet_layers = [l for l in model.layers if 'efficientnet' in l.name.lower()]
if efficientnet_layers:
    print(f"   EfficientNet layers: {len(efficientnet_layers)}")
    trainable_eff = [l for l in efficientnet_layers if l.trainable]
    print(f"   EfficientNet trainable: {len(trainable_eff)}/{len(efficientnet_layers)}")
print()

# Load small sample and test
print("3. Testing with actual data...")
train_dir = 'processed_datasets/eyepacs_train'

def load_sample(data_dir, class_name, label, max_samples=50):
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

rg_images, rg_labels = load_sample(train_dir, 'RG', 1, max_samples=50)
nrg_images, nrg_labels = load_sample(train_dir, 'NRG', 0, max_samples=50)

X = np.array(rg_images + nrg_images)
y = np.array(rg_labels + nrg_labels)

print(f"   Loaded: {len(X)} images ({len(rg_images)} RG, {len(nrg_images)} NRG)")
print(f"   Labels: {np.sum(y==1)} RG, {np.sum(y==0)} NRG")
print()

# Test predictions
print("4. Testing model predictions...")
predictions = model.predict(X, verbose=0)
print(f"   Predictions shape: {predictions.shape}")
print(f"   Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
print(f"   Prediction mean: {predictions.mean():.3f}")

if predictions.shape[1] == 1:
    pred_classes = (predictions > 0.5).astype(int).flatten()
    print(f"   Predicted: {np.sum(pred_classes==1)} RG, {np.sum(pred_classes==0)} NRG")
    accuracy = np.mean(pred_classes == y)
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    # Check if predictions are all the same
    unique_preds = np.unique(pred_classes)
    if len(unique_preds) == 1:
        print(f"   [PROBLEM] All predictions are the same: {unique_preds[0]}")
        print(f"   Model is not learning - always predicting same class!")
    
    # Check prediction distribution
    if predictions.mean() > 0.45 and predictions.mean() < 0.55:
        print(f"   [PROBLEM] Predictions are all around 0.5 (random guessing)")
        print(f"   Model output is not confident - always predicting ~50%")
print()

# Check if model weights are actually changing
print("5. Checking if model can actually learn...")
# Freeze base
for layer in model.layers:
    if 'efficientnet' in layer.name.lower():
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Get initial weights
initial_weights = []
for layer in model.layers:
    if layer.trainable:
        initial_weights.append([w.numpy().copy() for w in layer.get_weights()])

# Train for 1 batch
X_small = X[:16]
y_small = y[:16]
model.fit(X_small, y_small, epochs=1, batch_size=16, verbose=0)

# Check if weights changed
weights_changed = False
for i, layer in enumerate([l for l in model.layers if l.trainable]):
    if i < len(initial_weights):
        current_weights = layer.get_weights()
        for w1, w2 in zip(initial_weights[i], current_weights):
            if not np.allclose(w1, w2, atol=1e-6):
                weights_changed = True
                break

if weights_changed:
    print("   [OK] Model weights ARE changing - model can learn")
else:
    print("   [PROBLEM] Model weights NOT changing - model cannot learn!")
    print("   This is the root cause!")

print()

# Final diagnosis
print("="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)

if not weights_changed:
    print("[CRITICAL] Model weights are NOT changing!")
    print("   This means the model cannot learn.")
    print("   Possible causes:")
    print("   1. All layers are frozen (even head layers)")
    print("   2. Learning rate is 0 or too small")
    print("   3. Model architecture issue")
elif len(unique_preds) == 1 if 'unique_preds' in locals() else False:
    print("[CRITICAL] Model always predicts the same class!")
    print("   Model is not learning different patterns.")
elif predictions.mean() > 0.45 and predictions.mean() < 0.55:
    print("[CRITICAL] Model predictions are all around 0.5!")
    print("   Model is not confident - always predicting ~50%")
    print("   This causes 50% accuracy (random guessing)")
else:
    print("[INFO] Model seems OK, but accuracy is still 50%")
    print("   Need to check training process itself")

print("="*70)

