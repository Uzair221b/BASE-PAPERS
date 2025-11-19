"""
Restart Preprocessed Training (Fixed)
======================================
Restart training with preprocessed images, with better error handling
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime
import sys

print("="*70)
print("RESTARTING PREPROCESSED TRAINING (FIXED)")
print("="*70)
print()

# Use PREPROCESSED images
preprocessed_dir = 'processed_datasets/eyepacs_train'

def load_preprocessed_images(data_dir, class_name, label):
    """Load preprocessed images with error handling"""
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        print(f"[ERROR] Path not found: {class_path}")
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"   Loading {len(image_files)} PREPROCESSED images from {class_name}...")
    
    loaded = 0
    failed = 0
    
    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                failed += 1
                if failed <= 3:  # Show first 3 errors
                    print(f"   [WARNING] Cannot read: {img_file}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))
            
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(label)
            loaded += 1
            
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"   [ERROR] {img_file}: {e}")
            continue
    
    print(f"   Loaded: {loaded}, Failed: {failed}")
    return images, labels

print("Loading PREPROCESSED dataset...")
try:
    rg_images, rg_labels = load_preprocessed_images(preprocessed_dir, 'RG', 1)
    nrg_images, nrg_labels = load_preprocessed_images(preprocessed_dir, 'NRG', 0)
    
    if len(rg_images) == 0 or len(nrg_images) == 0:
        print("[ERROR] No images loaded! Cannot continue.")
        sys.exit(1)
    
    X = np.array(rg_images + nrg_images)
    y = np.array(rg_labels + nrg_labels)
    
    print(f"\n[OK] Loaded {len(X)} images total")
    print(f"  RG: {len(rg_images)}, NRG: {len(nrg_images)}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} ({np.sum(y_train==1)} RG, {np.sum(y_train==0)} NRG)")
    print(f"  Val: {len(X_val)} ({np.sum(y_val==1)} RG, {np.sum(y_val==0)} NRG)")
    print()
    
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Build model
print("Building CNN model...")
model = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Model parameters: {model.count_params():,}")
print()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('models', exist_ok=True)

callbacks = [
    ModelCheckpoint(
        f'models/best_model_preprocessed_{timestamp}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
]

print("="*70)
print("STARTING TRAINING WITH PREPROCESSED IMAGES")
print("="*70)
print("Target: 99%+ validation accuracy")
print("="*70)
print()

try:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    best_acc = max(history.history['val_accuracy'])
    
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation accuracy: {best_acc*100:.2f}%")
    
    if best_acc >= 0.99:
        print("[SUCCESS] Reached 99%+ accuracy!")
    else:
        print(f"Accuracy: {best_acc*100:.2f}% (target: 99%+)")
    
    print("="*70)
    
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()

