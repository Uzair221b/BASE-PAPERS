"""
Train with Preprocessed Images (9 Techniques)
==============================================
Use the preprocessed images with 9 techniques to reach 99%+ accuracy
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

print("="*70)
print("TRAINING WITH PREPROCESSED IMAGES (9 TECHNIQUES)")
print("="*70)
print("Using preprocessed images designed for 99%+ accuracy")
print("="*70)
print()

# Use PREPROCESSED images (with 9 techniques)
preprocessed_dir = 'processed_datasets/eyepacs_train'

def load_preprocessed_images(data_dir, class_name, label):
    """Load preprocessed images (9 techniques already applied)"""
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"   Loading {len(image_files)} PREPROCESSED images from {class_name}...")
    
    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)
        try:
            # Load preprocessed image (9 techniques already applied)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize if needed (should already be 224x224)
                if img.shape[:2] != (224, 224):
                    img = cv2.resize(img, (224, 224))
                
                # Normalize to 0-1 (preprocessed images are saved as 0-255)
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(label)
        except Exception as e:
            print(f"   Error loading {img_file}: {e}")
            continue
    
    return images, labels

print("Loading PREPROCESSED dataset (9 techniques applied)...")
rg_images, rg_labels = load_preprocessed_images(preprocessed_dir, 'RG', 1)
nrg_images, nrg_labels = load_preprocessed_images(preprocessed_dir, 'NRG', 0)

X = np.array(rg_images + nrg_images)
y = np.array(rg_labels + nrg_labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nPREPROCESSED Dataset loaded:")
print(f"  Train: {len(X_train)} ({np.sum(y_train==1)} RG, {np.sum(y_train==0)} NRG)")
print(f"  Val: {len(X_val)} ({np.sum(y_val==1)} RG, {np.sum(y_val==0)} NRG)")
print(f"  Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print()
print("These images have 9 preprocessing techniques applied:")
print("  1. Scaling to 224×224")
print("  2. Cropping to optic disc")
print("  3. Color normalization (z-score)")
print("  4. CLAHE enhancement")
print("  5. Gamma correction")
print("  6. Bilateral filtering")
print("  7. Enhanced CLAHE (LAB)")
print("  8. Image sharpening")
print("  9. Class balancing")
print()

# Build same CNN architecture (proven to work)
print("Building CNN model (same architecture as working model)...")
model = models.Sequential([
    # Data augmentation
    layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    
    # First block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Fourth block
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Dense layers
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
print("Preprocessed images should help reach this goal!")
print("="*70)
print()

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
    print("[SUCCESS] Reached 99%+ accuracy with preprocessed images!")
    print(f"Model saved: models/best_model_preprocessed_{timestamp}.h5")
else:
    print(f"Accuracy: {best_acc*100:.2f}% (target: 99%+)")
    if best_acc >= 0.95:
        print("[EXCELLENT] Very close to 99%! Preprocessed images helped!")
    elif best_acc >= 0.90:
        print("[GOOD] Preprocessed images improved accuracy!")
    else:
        print("May need more training or different approach.")

print("="*70)

