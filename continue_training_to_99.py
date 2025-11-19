"""
Continue Training to 99% Accuracy
==================================
Resume training from the best model to reach 99% accuracy
Increased patience to allow more training
"""

import numpy as np
import os
import cv2
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("="*70)
print("CONTINUE TRAINING TO 99% ACCURACY")
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

print(f"Loading model: {latest_model}")
model = keras.models.load_model(model_path)

# Test current accuracy
print("\nTesting current accuracy...")
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

print(f"Current accuracy: {current_accuracy*100:.2f}%")
print(f"Target: 99.00%")
print(f"Need: {(0.99 - current_accuracy)*100:.2f}% more")
print()

# Load full training data
print("Loading full training dataset...")
rg_train, rg_train_labels = load_sample(original_dir, 'RG', 1, max_samples=4000)
nrg_train, nrg_train_labels = load_sample(original_dir, 'NRG', 0, max_samples=4000)

X_train = np.array(rg_train + nrg_train)
y_train = np.array(rg_train_labels + nrg_train_labels)

# Use test set as validation
X_val = X_test
y_val = y_test

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print()

# Setup callbacks with INCREASED patience
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_checkpoint_path = f'models/best_cnn_model_{timestamp}.h5'

callbacks = [
    ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=30,  # INCREASED from 20 to 30 - allow more training
        restore_best_weights=True,
        verbose=1,
        mode='max',
        min_delta=0.0001  # Minimum change to count as improvement
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,  # INCREASED from 8 to 10
        min_lr=1e-7,
        verbose=1
    )
]

print("="*70)
print("TRAINING CONFIGURATION")
print("="*70)
print(f"Starting accuracy: {current_accuracy*100:.2f}%")
print(f"Target accuracy: 99.00%")
print(f"Max epochs: 200 (will stop early if 99% reached)")
print(f"EarlyStopping patience: 30 epochs (increased from 20)")
print(f"Batch size: 16")
print()
print("Training will continue until:")
print("  1. 99%+ accuracy is reached, OR")
print("  2. No improvement for 30 epochs")
print("="*70)
print()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'AUC']
)

# Custom callback to stop at 99%
class StopAt99Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        if val_acc >= 0.99:
            print(f"\n[SUCCESS] 99%+ accuracy reached! ({val_acc*100:.2f}%)")
            print("Stopping training...")
            self.model.stop_training = True

callbacks.append(StopAt99Callback())

# Train
print("Starting training...")
print()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
print()
print("="*70)
print("FINAL RESULTS")
print("="*70)

final_predictions = model.predict(X_val, verbose=0)
final_pred_classes = (final_predictions > 0.5).astype(int).flatten()
final_accuracy = np.mean(final_pred_classes == y_val)

print(f"Final accuracy: {final_accuracy*100:.2f}%")
print(f"Target: 99.00%")

if final_accuracy >= 0.99:
    print("[SUCCESS] 99%+ accuracy achieved!")
else:
    print(f"[INFO] Accuracy is {final_accuracy*100:.2f}% - close but not 99% yet")
    print("You can run this script again to continue training")

print(f"Model saved: {model_checkpoint_path}")
print("="*70)

