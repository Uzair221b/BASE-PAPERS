"""
Train with Preprocessed Images (Memory Efficient)
================================================
Uses smaller batch size and better memory management
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
import gc

print("="*70)
print("TRAINING WITH PREPROCESSED IMAGES (Memory Efficient)")
print("="*70)
print("Using smaller batch size to avoid memory issues")
print("="*70)
print()

# Use PREPROCESSED images
preprocessed_dir = 'processed_datasets/eyepacs_train'

def load_preprocessed_images(data_dir, class_name, label):
    """Load preprocessed images with memory management"""
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
    
    for i, img_file in enumerate(image_files):
        if (i + 1) % 500 == 0:
            print(f"     Progress: {i+1}/{len(image_files)} loaded...")
        
        img_path = os.path.join(class_path, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                failed += 1
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
                print(f"     [WARNING] Error loading {img_file}: {e}")
            continue
    
    print(f"   Loaded: {loaded}, Failed: {failed}")
    return images, labels

print("Loading PREPROCESSED dataset (this may take a moment)...")
try:
    rg_images, rg_labels = load_preprocessed_images(preprocessed_dir, 'RG', 1)
    print()
    nrg_images, nrg_labels = load_preprocessed_images(preprocessed_dir, 'NRG', 0)
    
    if len(rg_images) == 0 or len(nrg_images) == 0:
        print("[ERROR] No images loaded! Cannot continue.")
        exit(1)
    
    print()
    print("Converting to arrays...")
    X = np.array(rg_images + nrg_images)
    y = np.array(rg_labels + nrg_labels)
    
    # Free memory
    del rg_images, nrg_images, rg_labels, nrg_labels
    gc.collect()
    
    print(f"[OK] Loaded {len(X)} images total")
    print(f"  Memory usage: {X.nbytes / (1024**3):.2f} GB")
    print()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(X_train)} ({np.sum(y_train==1)} RG, {np.sum(y_train==0)} NRG)")
    print(f"  Val: {len(X_val)} ({np.sum(y_val==1)} RG, {np.sum(y_val==0)} NRG)")
    print()
    
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

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

# Use smaller batch size to avoid memory issues
BATCH_SIZE = 16  # Reduced from 32
print(f"Using batch size: {BATCH_SIZE} (reduced for memory efficiency)")
print()

print("="*70)
print("STARTING TRAINING WITH PREPROCESSED IMAGES")
print("="*70)
print("Target: 99%+ validation accuracy")
print("Batch size: 16 (memory efficient)")
print("="*70)
print()

try:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=BATCH_SIZE,  # Smaller batch size
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
        print("[SUCCESS] Reached 99%+ accuracy with preprocessed images!")
        print(f"Model saved: models/best_model_preprocessed_{timestamp}.h5")
    else:
        print(f"Accuracy: {best_acc*100:.2f}% (target: 99%+)")
        if best_acc >= 0.95:
            print("[EXCELLENT] Very close to 99%!")
        elif best_acc >= 0.90:
            print("[GOOD] Preprocessed images helped improve accuracy!")
    
    print("="*70)
    
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("This might be a memory issue. Try:")
    print("  1. Wait for original training to finish")
    print("  2. Or use even smaller batch size (8)")

