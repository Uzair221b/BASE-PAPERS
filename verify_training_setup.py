"""
Verify Training Setup is Correct
=================================
Check if everything is set up correctly BEFORE training wastes time
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

print("="*70)
print("VERIFYING TRAINING SETUP - BEFORE IT WASTES TIME")
print("="*70)
print()

# Step 1: Check model
print("Step 1: Checking model...")
model_path = 'models/best_model_20251110_193527.h5'
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
    print(f"[OK] Model loaded: {model_path}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Loss: {model.loss}")
    
    # Check if base is frozen
    efficientnet_layers = [l for l in model.layers if 'efficientnet' in l.name.lower()]
    if efficientnet_layers:
        frozen = sum([1 for l in efficientnet_layers if not l.trainable])
        total = len(efficientnet_layers)
        print(f"   EfficientNet layers: {frozen}/{total} frozen")
        if frozen == total:
            print(f"   [OK] Base layers are frozen (correct for Phase 1)")
        else:
            print(f"   [WARNING] Some base layers are not frozen!")
else:
    print(f"[ERROR] Model not found!")
    exit(1)

print()

# Step 2: Check data loading
print("Step 2: Checking data loading...")
train_dir = 'processed_datasets/eyepacs_train'

def quick_check(data_dir, class_name, label, max_check=100):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_check]
    
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

rg_images, rg_labels = quick_check(train_dir, 'RG', 1, max_check=100)
nrg_images, nrg_labels = quick_check(train_dir, 'NRG', 0, max_check=100)

print(f"   RG samples checked: {len(rg_images)} (label: {rg_labels[0] if rg_labels else 'N/A'})")
print(f"   NRG samples checked: {len(nrg_images)} (label: {nrg_labels[0] if nrg_labels else 'N/A'})")

if len(rg_images) == 0 or len(nrg_images) == 0:
    print(f"   [ERROR] No images loaded!")
    exit(1)

# Quick test with small sample
X_test = np.array(rg_images[:10] + nrg_images[:10])
y_test = np.array(rg_labels[:10] + nrg_labels[:10])

print(f"   Test sample: {len(X_test)} images")
print(f"   Labels: {np.sum(y_test==1)} RG, {np.sum(y_test==0)} NRG")
print()

# Step 3: Test model prediction
print("Step 3: Testing model prediction...")
try:
    predictions = model.predict(X_test, verbose=0)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Check if predictions are reasonable
    if predictions.shape[1] == 1:  # Binary output
        pred_classes = (predictions > 0.5).astype(int).flatten()
        print(f"   Predicted classes: {np.sum(pred_classes==1)} RG, {np.sum(pred_classes==0)} NRG")
        
        # Check accuracy on test sample
        accuracy = np.mean(pred_classes == y_test)
        print(f"   Test accuracy: {accuracy*100:.2f}%")
        
        if accuracy == 0.5:
            print(f"   [WARNING] 50% accuracy - model might be random!")
        elif accuracy > 0.5:
            print(f"   [OK] Accuracy > 50% - model is making predictions")
    else:
        pred_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(pred_classes == y_test)
        print(f"   Test accuracy: {accuracy*100:.2f}%")
        
except Exception as e:
    print(f"   [ERROR] Prediction test failed: {e}")

print()

# Step 4: Check label format
print("Step 4: Checking label format...")
output_shape = model.output_shape
print(f"   Model output: {output_shape}")

if len(output_shape) == 2 and output_shape[1] == 1:
    print(f"   [OK] Model expects binary labels (0/1) - correct!")
    print(f"   Labels are: RG=1, NRG=0 - matches!")
elif len(output_shape) == 2 and output_shape[1] == 2:
    print(f"   [WARNING] Model expects one-hot labels (2 classes)")
    print(f"   But we're using binary labels (0/1)")
    print(f"   This might cause issues!")
else:
    print(f"   [INFO] Model output shape: {output_shape}")

print()

# Step 5: Check loss function
print("Step 5: Checking loss function...")
if 'binary_crossentropy' in str(model.loss):
    print(f"   [OK] Loss: binary_crossentropy (correct for binary classification)")
elif 'sparse_categorical' in str(model.loss):
    print(f"   [OK] Loss: sparse_categorical_crossentropy (works with integer labels)")
else:
    print(f"   [INFO] Loss: {model.loss}")

print()

# Step 6: Final verdict
print("="*70)
print("VERIFICATION SUMMARY")
print("="*70)

issues = []
if len(rg_images) == 0 or len(nrg_images) == 0:
    issues.append("No images loaded")
if len(output_shape) == 2 and output_shape[1] == 2 and 'binary' in str(model.loss):
    issues.append("Label format mismatch (model expects one-hot but using binary)")

if issues:
    print("[WARNING] Issues found:")
    for issue in issues:
        print(f"  - {issue}")
    print()
    print("Training might not work correctly. Fix these issues first!")
else:
    print("[OK] Setup looks correct!")
    print("Training should work properly.")
    print()
    print("Monitor accuracy - it should improve above 50% within first few epochs.")
    print("If it stays at 50%, stop training and investigate.")

print("="*70)

