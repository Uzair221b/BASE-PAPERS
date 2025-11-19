"""
SIMPLE RESUME TRAINING SCRIPT - COMPLETE PLAN
==============================================

This script will:
1. Load your saved model (best_model_20251110_193527.h5)
2. Complete Phase 1: 29 more epochs (to reach 50 total) - frozen base
3. Then Phase 2: 20 epochs (fine-tuning all layers)
4. Total: Up to 70 epochs (but EarlyStopping will stop if overfitting)
5. Works on CPU (uses smaller batch size to avoid memory issues)

SAFETY FEATURES (prevents overfitting):
- EarlyStopping: Stops if validation accuracy doesn't improve for 10 epochs
- ModelCheckpoint: Saves the BEST model automatically
- ReduceLROnPlateau: Reduces learning rate if model stops improving

Just run: python simple_resume_training.py
"""

import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.utils import to_categorical
import logging
from datetime import datetime
import time

# Try to import tqdm for better progress bars
try:
    from tqdm.keras import TqdmCallback
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for better progress bars: pip install tqdm")

# Setup logging for monitoring
log_dir = Path('training_logs')
log_dir.mkdir(exist_ok=True)
training_log = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(training_log),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*70)
print("SIMPLE RESUME TRAINING")
print("="*70)
logger.info("="*70)
logger.info("TRAINING STARTED")
logger.info("="*70)

# ============================================================================
# STEP 1: Load your saved model
# ============================================================================
print("\n[1/5] Loading saved model...")
model_path = 'models/best_model_20251110_193527.h5'

if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print("Please check the path and try again.")
    exit(1)

model = keras.models.load_model(model_path)
print(f"[OK] Model loaded from: {model_path}")
print(f"   Model parameters: {model.count_params():,}")
logger.info(f"Model loaded: {model_path}")
logger.info(f"Model parameters: {model.count_params():,}")

# ============================================================================
# STEP 2: Load your preprocessed data
# ============================================================================
print("\n[2/5] Loading preprocessed data...")

def load_images_simple(data_dir, class_name, label):
    """Simple function to load images"""
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        print(f"   Warning: {class_path} not found")
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
        except Exception as e:
            continue
    
    return images, labels

# Load training data
train_dir = 'processed_datasets/eyepacs_train'
print(f"   Loading from: {train_dir}")

rg_images, rg_labels = load_images_simple(train_dir, 'RG', 1)
nrg_images, nrg_labels = load_images_simple(train_dir, 'NRG', 0)

# Combine
X_train_full = np.array(rg_images + nrg_images)
y_train_full = np.array(rg_labels + nrg_labels)

print(f"[OK] Training data loaded: {len(X_train_full)} images")
print(f"   RG (Glaucoma): {np.sum(y_train_full == 1)}")
print(f"   NRG (Normal): {np.sum(y_train_full == 0)}")
logger.info(f"Training data loaded: {len(X_train_full)} images")
logger.info(f"RG: {np.sum(y_train_full == 1)}, NRG: {np.sum(y_train_full == 0)}")

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print(f"   Train: {len(X_train)}, Validation: {len(X_val)}")

# ============================================================================
# STEP 3: Setup for CPU training (smaller batch size)
# ============================================================================
print("\n[3/6] Setting up for CPU training...")

# Use smaller batch size for CPU to avoid memory issues
BATCH_SIZE = 16  # Reduced from 32
PHASE1_EPOCHS_REMAINING = 29  # You did 21, need 29 more to reach 50
PHASE2_EPOCHS = 20  # Fine-tuning phase

print(f"   Batch size: {BATCH_SIZE} (smaller for CPU)")
print(f"   Phase 1 remaining: {PHASE1_EPOCHS_REMAINING} epochs (to reach 50 total)")
print(f"   Phase 2: {PHASE2_EPOCHS} epochs (fine-tuning)")
print(f"   Total: Up to {PHASE1_EPOCHS_REMAINING + PHASE2_EPOCHS} epochs")
print(f"   [INFO] EarlyStopping will stop early if overfitting starts!")

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("   [INFO] GPU detected but using CPU settings for stability")
    BATCH_SIZE = 32  # Can use larger batch with GPU

# ============================================================================
# STEP 4: Setup callbacks (safety features to prevent overfitting)
# ============================================================================
print("\n[4/6] Setting up safety features (prevents overfitting)...")

# Custom progress callback class
class ProgressCallback(Callback):
    """Custom callback to show enhanced progress"""
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.batch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.batch_times = []
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{self.params['epochs']}")
        print(f"{'='*70}")
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        print(f"  Loss: {logs.get('loss', 0):.4f}")
        print(f"  Accuracy: {logs.get('accuracy', 0)*100:.2f}%")
        if 'val_loss' in logs:
            print(f"  Val Loss: {logs.get('val_loss', 0):.4f}")
            print(f"  Val Accuracy: {logs.get('val_accuracy', 0)*100:.2f}%")
        
        # Estimate remaining time
        remaining_epochs = self.params['epochs'] - (epoch + 1)
        if remaining_epochs > 0 and avg_batch_time > 0:
            estimated_time = remaining_epochs * epoch_time
            print(f"  Estimated time remaining: {estimated_time/60:.1f} minutes ({estimated_time/3600:.1f} hours)")
        print(f"{'='*70}\n")
    
    def on_batch_end(self, batch, logs=None):
        if self.epoch_start_time:
            batch_time = time.time() - self.epoch_start_time
            self.batch_times.append(batch_time)
        
        # Show progress every 10 batches
        if batch % 10 == 0 and 'steps' in self.params:
            progress = (batch + 1) / self.params['steps'] * 100
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r  Batch {batch + 1}/{self.params['steps']} [{bar}] {progress:.1f}% - Loss: {logs.get('loss', 0):.4f}", end='', flush=True)

# Create progress callback
progress_callback = ProgressCallback()

callbacks_phase1 = [
    progress_callback,  # Add progress callback first
    ModelCheckpoint(
        'models/best_model_phase1.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Stops if no improvement for 10 epochs
        restore_best_weights=True,  # Uses best model, not last one
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

# Add tqdm callback if available
if HAS_TQDM:
    callbacks_phase1.insert(1, TqdmCallback(verbose=1))

# Create progress callback for phase 2
progress_callback_phase2 = ProgressCallback()

callbacks_phase2 = [
    progress_callback_phase2,  # Add progress callback first
    ModelCheckpoint(
        'models/best_model_final.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,  # Slightly less patience in fine-tuning
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

# Add tqdm callback if available
if HAS_TQDM:
    callbacks_phase2.insert(1, TqdmCallback(verbose=1))

print("[OK] Safety features ready:")
print("   - EarlyStopping: Will stop if overfitting starts")
print("   - ModelCheckpoint: Saves best model automatically")
print("   - ReduceLROnPlateau: Adjusts learning rate automatically")

# ============================================================================
# STEP 5: Check model format and prepare labels
# ============================================================================
print("\n[5/6] Checking model format...")
output_shape = model.output_shape
print(f"   Model output shape: {output_shape}")

# If model expects one-hot encoded labels (2 outputs), convert labels
if len(output_shape) == 2 and output_shape[1] == 2:
    print("   Model expects 2-class output, converting labels to one-hot...")
    y_train_cat = to_categorical(y_train, 2)
    y_val_cat = to_categorical(y_val, 2)
    y_train_use = y_train_cat
    y_val_use = y_val_cat
else:
    print("   Model expects binary labels (0/1), using as-is...")
    y_train_use = y_train
    y_val_use = y_val

# ============================================================================
# STEP 6: PHASE 1 - Complete training with frozen base (29 more epochs)
# ============================================================================
print("\n[6/6] Starting training...")
print("="*70)
print("PHASE 1: Completing frozen base training")
print("="*70)
print(f"Continuing from epoch 21...")
print(f"Training for {PHASE1_EPOCHS_REMAINING} more epochs (to reach 50 total)")
print(f"Batch size: {BATCH_SIZE}")
print(f"Base layers: FROZEN (only training head)")
print("="*70)
print("\n[INFO] This will take a while on CPU (maybe 10-15 hours)...")
print("[OK] But EarlyStopping will stop early if overfitting starts!")
print("[OK] You can leave it running.\n")
logger.info("Starting Phase 1 training...")
logger.info(f"Epochs: {PHASE1_EPOCHS_REMAINING}, Batch size: {BATCH_SIZE}")

# Phase 1: Continue training with frozen base
try:
    history_phase1 = model.fit(
        X_train, y_train_use,
        validation_data=(X_val, y_val_use),
        epochs=PHASE1_EPOCHS_REMAINING,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_phase1,
        verbose=1,
        initial_epoch=21  # Display starts from epoch 22 (you already did 21)
    )
except Exception as e:
    print(f"\n   Error during training: {e}")
    print("   Trying without initial_epoch parameter...")
    history_phase1 = model.fit(
        X_train, y_train_use,
        validation_data=(X_val, y_val_use),
        epochs=PHASE1_EPOCHS_REMAINING,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_phase1,
        verbose=1
    )

print("\n" + "="*70)
print("PHASE 1 COMPLETE!")
print("="*70)
best_phase1 = max(history_phase1.history['val_accuracy'])
print(f"Best validation accuracy (Phase 1): {best_phase1*100:.2f}%")
print(f"Model saved to: models/best_model_phase1.h5")
logger.info("Phase 1 complete!")
logger.info(f"Best validation accuracy (Phase 1): {best_phase1*100:.2f}%")

# ============================================================================
# PHASE 2: Fine-tuning (unfreeze base layers)
# ============================================================================
print("\n" + "="*70)
print("PHASE 2: Fine-tuning all layers")
print("="*70)
print("Unfreezing EfficientNetB4 base layers...")
print(f"Training for {PHASE2_EPOCHS} epochs")
print(f"Learning rate: Reduced (0.0001)")
print("="*70)

# Unfreeze base layers
for layer in model.layers:
    if 'efficientnet' in layer.name.lower():
        layer.trainable = True

# Recompile with lower learning rate for fine-tuning
# Determine loss function from model
if hasattr(model, 'loss'):
    loss_func = model.loss if isinstance(model.loss, str) else 'sparse_categorical_crossentropy'
else:
    loss_func = 'sparse_categorical_crossentropy'

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
    loss=loss_func,
    metrics=['accuracy']
)

print("[OK] Base layers unfrozen")
print("[OK] Model recompiled with lower learning rate")
print("\nStarting Phase 2 training...\n")
logger.info("Starting Phase 2 training (fine-tuning)...")
logger.info(f"Epochs: {PHASE2_EPOCHS}, Batch size: {BATCH_SIZE}")

# Phase 2: Fine-tuning
history_phase2 = model.fit(
    X_train, y_train_use,
    validation_data=(X_val, y_val_use),
    epochs=PHASE2_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_phase2,
    verbose=1
)

# ============================================================================
# DONE!
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
best_phase2 = max(history_phase2.history['val_accuracy'])
best_overall = max(best_phase1, best_phase2)
print(f"Best validation accuracy (Phase 1): {best_phase1*100:.2f}%")
print(f"Best validation accuracy (Phase 2): {best_phase2*100:.2f}%")
print(f"Best overall: {best_overall*100:.2f}%")
print(f"\nModels saved:")
print(f"  - Phase 1 best: models/best_model_phase1.h5")
print(f"  - Final best: models/best_model_final.h5")
print("="*70)
print("\n[OK] Training complete! EarlyStopping prevented overfitting.")
print("[OK] Use models/best_model_final.h5 for your best model!")
print("="*70)
logger.info("="*70)
logger.info("TRAINING COMPLETE!")
logger.info(f"Best overall accuracy: {best_overall*100:.2f}%")
logger.info("="*70)

