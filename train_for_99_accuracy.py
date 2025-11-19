"""
Train for 99%+ Accuracy
========================
Try different models and approaches to reach 99%+ accuracy
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4, ResNet50, DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

print("="*70)
print("TRAINING FOR 99%+ ACCURACY")
print("="*70)
print("Trying multiple approaches to reach 99%+ accuracy")
print("="*70)
print()

# Load FULL dataset
original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'

def load_images(data_dir, class_name, label):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        return images, labels
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"   Loading {len(image_files)} images from {class_name}...")
    
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

print("Loading FULL dataset...")
rg_images, rg_labels = load_images(original_dir, 'RG', 1)
nrg_images, nrg_labels = load_images(original_dir, 'NRG', 0)

X = np.array(rg_images + nrg_images)
y = np.array(rg_labels + nrg_labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset loaded:")
print(f"  Train: {len(X_train)} ({np.sum(y_train==1)} RG, {np.sum(y_train==0)} NRG)")
print(f"  Val: {len(X_val)} ({np.sum(y_val==1)} RG, {np.sum(y_val==0)} NRG)")
print()

# Try Approach 1: EfficientNetB4 with unfrozen layers from start
print("="*70)
print("APPROACH 1: EfficientNetB4 (Unfrozen base)")
print("="*70)

base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# DON'T freeze - let it learn from start
base.trainable = True

inputs = keras.Input(shape=(224, 224, 3))
x = layers.Rescaling(1./255)(inputs)
x = base(x, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model1 = models.Model(inputs, outputs)

# Use lower learning rate for fine-tuning
model1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Model parameters: {model1.count_params():,}")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model1.trainable_weights]):,}")
print()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('models', exist_ok=True)

callbacks = [
    ModelCheckpoint(
        f'models/best_model_99percent_{timestamp}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-8,
        verbose=1
    )
]

print("Training EfficientNetB4 (unfrozen) for 100 epochs...")
print("Target: 99%+ validation accuracy")
print("="*70)
print()

history1 = model1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

best_acc1 = max(history1.history['val_accuracy'])

print()
print("="*70)
print("APPROACH 1 RESULTS")
print("="*70)
print(f"Best validation accuracy: {best_acc1*100:.2f}%")

if best_acc1 >= 0.99:
    print("[SUCCESS] Reached 99%+ accuracy!")
    print(f"Model saved: models/best_model_99percent_{timestamp}.h5")
    exit(0)
else:
    print(f"[NOT YET] Accuracy: {best_acc1*100:.2f}% (need 99%+)")
    print("Trying Approach 2...")
    print()

# If Approach 1 didn't work, try Approach 2: ResNet50
if best_acc1 < 0.99:
    print("="*70)
    print("APPROACH 2: ResNet50 (Unfrozen base)")
    print("="*70)
    
    base2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base2.trainable = True
    
    inputs2 = keras.Input(shape=(224, 224, 3))
    x2 = layers.Rescaling(1./255)(inputs2)
    x2 = base2(x2, training=True)
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.Dropout(0.3)(x2)
    x2 = layers.Dense(512, activation='relu')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.4)(x2)
    x2 = layers.Dense(256, activation='relu')(x2)
    x2 = layers.Dropout(0.3)(x2)
    x2 = layers.Dense(128, activation='relu')(x2)
    outputs2 = layers.Dense(1, activation='sigmoid')(x2)
    
    model2 = models.Model(inputs2, outputs2)
    model2.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks2 = [
        ModelCheckpoint(
            f'models/best_model_resnet50_{timestamp}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    print("Training ResNet50 for 100 epochs...")
    print("="*70)
    print()
    
    history2 = model2.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=callbacks2,
        verbose=1
    )
    
    best_acc2 = max(history2.history['val_accuracy'])
    
    print()
    print("="*70)
    print("APPROACH 2 RESULTS")
    print("="*70)
    print(f"Best validation accuracy: {best_acc2*100:.2f}%")
    
    if best_acc2 >= 0.99:
        print("[SUCCESS] Reached 99%+ accuracy with ResNet50!")
        exit(0)

print()
print("="*70)
print("FINAL RESULTS")
print("="*70)
print(f"EfficientNetB4: {best_acc1*100:.2f}%")
if best_acc1 < 0.99:
    print(f"ResNet50: {best_acc2*100:.2f}%")
print()
print("Best model saved. Continue training if needed to reach 99%+")

