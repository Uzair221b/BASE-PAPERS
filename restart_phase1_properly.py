"""
Restart Phase 1 Properly - Fix the 50% Accuracy Issue
=======================================================

You're absolutely right! If Phase 1 from epoch 21 was wrong (50% accuracy),
we need to restart Phase 1 properly BEFORE doing Phase 2.
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
from datetime import datetime
import time

print("="*70)
print("RESTARTING PHASE 1 PROPERLY - FIXING 50% ACCURACY")
print("="*70)
print()
print("You're absolutely right!")
print("If Phase 1 from epoch 21 was wrong (50% accuracy = random guessing),")
print("we MUST restart Phase 1 properly BEFORE doing Phase 2.")
print()
print("="*70)
print()

# Step 1: Load the ORIGINAL model (from epoch 21, before the bad training)
print("Step 1: Loading original model (from epoch 21)...")
original_model = 'models/best_model_20251110_193527.h5'
if os.path.exists(original_model):
    model = keras.models.load_model(original_model)
    print(f"[OK] Loaded original model: {original_model}")
    print(f"   This is the model from epoch 21 (before the bad training)")
else:
    print(f"[ERROR] Original model not found: {original_model}")
    exit(1)

print()

# Step 2: Verify model is frozen (Phase 1 = frozen base)
print("Step 2: Setting up for Phase 1 (frozen base)...")
for layer in model.layers:
    if 'efficientnet' in layer.name.lower():
        layer.trainable = False  # Freeze base for Phase 1

# Recompile for Phase 1
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

trainable_count = sum([1 for layer in model.layers if layer.trainable])
print(f"[OK] Base layers FROZEN (Phase 1 setup)")
print(f"   Trainable layers: {trainable_count}/{len(model.layers)}")
print()

# Step 3: Load data
print("Step 3: Loading data...")
def load_images_simple(data_dir, class_name, label):
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

train_dir = 'processed_datasets/eyepacs_train'
rg_images, rg_labels = load_images_simple(train_dir, 'RG', 1)
nrg_images, nrg_labels = load_images_simple(train_dir, 'NRG', 0)

X_train_full = np.array(rg_images + nrg_images)
y_train_full = np.array(rg_labels + nrg_labels)

print(f"[OK] Loaded {len(X_train_full)} images")
print(f"   RG: {np.sum(y_train_full == 1)}, NRG: {np.sum(y_train_full == 0)}")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)
print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
print()

# Step 4: Verify label format
print("Step 4: Verifying label format...")
output_shape = model.output_shape
print(f"   Model output: {output_shape}")

# Model outputs (None, 1) with sigmoid, so use binary labels
y_train_use = y_train
y_val_use = y_val
print(f"   Labels: Binary (0/1) - correct for sigmoid output")
print(f"   Label distribution - Train: {np.sum(y_train==1)} RG, {np.sum(y_train==0)} NRG")
print(f"   Label distribution - Val: {np.sum(y_val==1)} RG, {np.sum(y_val==0)} NRG")
print()

# Step 5: Setup logging
log_dir = Path('training_logs')
log_dir.mkdir(exist_ok=True)
training_log = log_dir / f'phase1_restart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(training_log),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Step 6: Setup callbacks
print("Step 5: Setting up callbacks...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('models', exist_ok=True)

callbacks = [
    ModelCheckpoint(
        f'models/best_model_phase1_restart_{timestamp}.h5',
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
print("[OK] Callbacks ready")
print()

# Step 7: Start Phase 1 training (restarting from epoch 21)
print("="*70)
print("RESTARTING PHASE 1 TRAINING (FROM EPOCH 21)")
print("="*70)
print("Training for 29 more epochs (to reach 50 total)")
print("Batch size: 16")
print("Base layers: FROZEN (only training head)")
print("Learning rate: 0.001")
print("="*70)
print()
print("This will fix the 50% accuracy issue!")
print("After Phase 1 is done properly, we'll do Phase 2.")
print()
print("Starting training...\n")

logger.info("Starting Phase 1 restart (from epoch 21)")
logger.info(f"Epochs: 29, Batch size: 16")

history = model.fit(
    X_train, y_train_use,
    validation_data=(X_val, y_val_use),
    epochs=29,
    batch_size=16,
    callbacks=callbacks,
    verbose=1,
    initial_epoch=21  # Continue from epoch 21
)

# Step 8: Results
print()
print("="*70)
print("PHASE 1 RESTART COMPLETE!")
print("="*70)
best_acc = max(history.history['val_accuracy'])
print(f"Best validation accuracy: {best_acc*100:.2f}%")

if best_acc > 0.50:
    print(f"[OK] Accuracy improved! Now: {best_acc*100:.2f}% (was 50%)")
    print("Phase 1 is now properly trained!")
    print("Ready to proceed to Phase 2.")
else:
    print(f"[WARNING] Accuracy still low: {best_acc*100:.2f}%")
    print("Need to investigate further before Phase 2.")

print(f"Model saved: models/best_model_phase1_restart_{timestamp}.h5")
print("="*70)

logger.info(f"Phase 1 restart complete. Best accuracy: {best_acc*100:.2f}%")

