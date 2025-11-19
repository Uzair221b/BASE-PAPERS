"""
Test Training First - Then Train for Real
==========================================
Test with small sample to verify it works BEFORE wasting 50 epochs
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

print("="*70)
print("TESTING FIRST - THEN TRAINING")
print("="*70)
print()

# Load data
train_dir = 'processed_datasets/eyepacs_train'

def load_images(data_dir, class_name, label, max_samples=None):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if max_samples:
        image_files = image_files[:max_samples]
    
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

print("Step 1: Testing with SMALL sample (200 images)...")
rg_images, rg_labels = load_images(train_dir, 'RG', 1, max_samples=100)
nrg_images, nrg_labels = load_images(train_dir, 'NRG', 0, max_samples=100)

X_test = np.array(rg_images + nrg_images)
y_test = np.array(rg_labels + nrg_labels)

X_train_small, X_val_small, y_train_small, y_val_small = train_test_split(
    X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
)

print(f"   Test sample: {len(X_train_small)} train, {len(X_val_small)} val")

# Build model
base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.025)(x)
x = layers.RandomZoom(0.035)(x)
x = layers.Rescaling(1./255)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Test initial accuracy
initial_pred = model.predict(X_val_small, verbose=0)
initial_acc = np.mean((initial_pred > 0.5).astype(int).flatten() == y_val_small)
print(f"   Initial accuracy: {initial_acc*100:.2f}%")

# Train for 5 epochs
print("\nStep 2: Training for 5 epochs (test)...")
history = model.fit(
    X_train_small, y_train_small,
    validation_data=(X_val_small, y_val_small),
    epochs=5,
    batch_size=16,
    verbose=1
)

# Check if accuracy improved
final_acc = max(history.history['val_accuracy'])
print(f"\n   Final accuracy: {final_acc*100:.2f}%")
print(f"   Improvement: {(final_acc - initial_acc)*100:.2f}%")

if final_acc <= 0.50:
    print("\n" + "="*70)
    print("TEST FAILED - Accuracy still 50% or below!")
    print("="*70)
    print("Training will NOT work. Need to fix the problem first.")
    exit(1)
elif final_acc - initial_acc < 0.05:
    print("\n" + "="*70)
    print("TEST WARNING - Accuracy barely improved!")
    print("="*70)
    print("Training might not work well. Proceed with caution.")
    response = input("Continue with full training? (y/n): ")
    if response.lower() != 'y':
        exit(1)
else:
    print("\n" + "="*70)
    print("TEST PASSED - Model is learning!")
    print("="*70)
    print("Accuracy improved. Full training should work.")
    print("="*70)

# Now load full dataset
print("\nStep 3: Loading FULL dataset...")
rg_images, rg_labels = load_images(train_dir, 'RG', 1, max_samples=None)
nrg_images, nrg_labels = load_images(train_dir, 'NRG', 0, max_samples=None)

X_full = np.array(rg_images + nrg_images)
y_full = np.array(rg_labels + nrg_labels)

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

print(f"   Full dataset: {len(X_train)} train, {len(X_val)} val")

# Rebuild model for full training
base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.025)(x)
x = layers.RandomZoom(0.035)(x)
x = layers.Rescaling(1./255)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Setup callbacks
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('models', exist_ok=True)

callbacks = [
    ModelCheckpoint(
        f'models/best_model_{timestamp}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Train
print("\n" + "="*70)
print("STARTING FULL TRAINING (50 epochs)")
print("="*70)
print("This will take several hours on CPU...")
print("="*70)
print()

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

