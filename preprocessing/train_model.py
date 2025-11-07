"""
Training script for glaucoma detection model
Optimized to achieve 99%+ accuracy using advanced preprocessing and model architecture
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse
import cv2
from pipeline import GlaucomaPreprocessingPipeline
from data_loading import load_dataset
import config

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB4, ResNet50, DenseNet121
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not installed. Install with: pip install tensorflow")


class HighAccuracyGlaucomaModel:
    """
    Advanced model architecture for achieving 99%+ accuracy in glaucoma detection.
    Uses transfer learning with optimized preprocessing.
    """
    
    def __init__(self, input_size=(224, 224, 3), num_classes=2):
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.pipeline = GlaucomaPreprocessingPipeline()
    
    def build_model(self, base_model='EfficientNetB4'):
        """
        Build advanced model using transfer learning.
        
        Args:
            base_model: Base architecture ('EfficientNetB4', 'ResNet50', 'DenseNet121')
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow required for model building")
            return None
        
        print(f"\nBuilding model with {base_model} architecture...")
        
        # Load pre-trained base model
        if base_model == 'EfficientNetB4':
            base = EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_size
            )
        elif base_model == 'ResNet50':
            base = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_size
            )
        elif base_model == 'DenseNet121':
            base = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_size
            )
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        # Freeze base layers initially
        base.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.input_size)
        
        # Data augmentation layers
        x = layers.RandomRotation(0.025)(inputs)
        x = layers.RandomZoom(0.035)(x)
        x = layers.Rescaling(1./255)(x)
        
        # Base model
        x = base(x, training=False)
        
        # Advanced head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        
        self.model = models.Model(inputs, outputs)
        
        # Compile with optimized settings
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=loss,
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        print(f"Model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def prepare_data(self, image_paths, labels):
        """
        Prepare training data with advanced preprocessing.
        
        Args:
            image_paths: List of image paths
            labels: Corresponding labels
            
        Returns:
            Processed X and y arrays
        """
        print("\nPreparing data with advanced preprocessing...")
        
        X_processed, y_processed = self.pipeline.process_dataset(
            [cv2.imread(str(p)) for p in image_paths],
            np.array(labels),
            balance_classes=True,
            apply_augmentation=False  # Augmentation handled by model
        )
        
        return X_processed, y_processed
    
    def train(self, X_train, y_train, X_val, y_val,
              epochs=50, batch_size=32, fine_tune_epochs=20):
        """
        Train model with fine-tuning for high accuracy.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Initial training epochs
            batch_size: Batch size
            fine_tune_epochs: Epochs for fine-tuning
        """
        if self.model is None:
            print("Error: Model not built. Call build_model() first.")
            return
        
        # Callbacks for training
        callbacks = [
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
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"\nPhase 1: Training with frozen base layers...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Phase 1: Train with frozen base
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning
        print(f"\nPhase 2: Fine-tuning (unfreezing base layers)...")
        
        # Unfreeze some layers for fine-tuning
        for layer in self.model.layers:
            if 'efficientnet' in layer.name or 'resnet' in layer.name or 'densenet' in layer.name:
                layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss='binary_crossentropy' if self.num_classes == 2 else 'categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        # Fine-tune
        fine_tune_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=fine_tune_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best accuracy
        best_acc = max(max(self.history.history['val_accuracy']), 
                      max(fine_tune_history.history['val_accuracy']))
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
        print(f"{'='*70}")
        
        return self.history


def load_dataset_with_labels(data_dir, label_file=None):
    """
    Load dataset with labels.
    
    Args:
        data_dir: Directory containing images, organized as:
                  data_dir/
                    positive/  (or glaucoma/)
                      image1.jpg
                      image2.jpg
                    negative/  (or normal/)
                      image3.jpg
        label_file: Optional CSV file with image names and labels
        
    Returns:
        image_paths, labels
    """
    image_paths = []
    labels = []
    
    if label_file and os.path.exists(label_file):
        # Load from CSV
        df = pd.read_csv(label_file)
        for _, row in df.iterrows():
            img_path = os.path.join(data_dir, row['Image_Name'])
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(row['Label'])
    else:
        # Load from folder structure
        positive_dir = None
        negative_dir = None
        
        # Try common folder names
        for folder in ['positive', 'glaucoma', '1', 'case']:
            test_path = os.path.join(data_dir, folder)
            if os.path.exists(test_path):
                positive_dir = test_path
                break
        
        for folder in ['negative', 'normal', '0', 'control']:
            test_path = os.path.join(data_dir, folder)
            if os.path.exists(test_path):
                negative_dir = test_path
                break
        
        # Load positive images
        if positive_dir:
            pos_images = load_dataset(positive_dir)
            image_paths.extend([str(p) for p in pos_images])
            labels.extend([1] * len(pos_images))
            print(f"Loaded {len(pos_images)} positive images from {positive_dir}")
        
        # Load negative images
        if negative_dir:
            neg_images = load_dataset(negative_dir)
            image_paths.extend([str(p) for p in neg_images])
            labels.extend([0] * len(neg_images))
            print(f"Loaded {len(neg_images)} negative images from {negative_dir}")
    
    return image_paths, np.array(labels)


def main():
    parser = argparse.ArgumentParser(description='Train glaucoma detection model for 99%+ accuracy')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training images (organized as positive/negative folders or CSV)')
    parser.add_argument('--label_file', type=str, default=None,
                       help='CSV file with image names and labels (optional)')
    parser.add_argument('--model_name', type=str, default='EfficientNetB4',
                       choices=['EfficientNetB4', 'ResNet50', 'DenseNet121'],
                       help='Base model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output', type=str, default='glaucoma_model.h5',
                       help='Output model filename')
    
    args = parser.parse_args()
    
    if not TENSORFLOW_AVAILABLE:
        print("Error: TensorFlow required for training. Install with: pip install tensorflow")
        return
    
    print("="*70)
    print("Glaucoma Detection Model Training")
    print("Target: 99.53%+ Accuracy")
    print("="*70)
    
    # Load data
    print(f"\nLoading dataset from: {args.data_dir}")
    image_paths, labels = load_dataset_with_labels(args.data_dir, args.label_file)
    
    if len(image_paths) == 0:
        print("Error: No images found!")
        return
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Positive: {np.sum(labels == 1)}, Negative: {np.sum(labels == 0)}")
    
    # Build model
    model_builder = HighAccuracyGlaucomaModel()
    model = model_builder.build_model(base_model=args.model_name)
    
    # Prepare data
    X, y = model_builder.prepare_data(image_paths, labels)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Train
    history = model_builder.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_accuracy = test_results[1]
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS:")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"{'='*70}")
    
    # Save final model
    model.save(args.output)
    print(f"\nModel saved to: {args.output}")
    print(f"You can now use this model with classify_images.py --model {args.output}")


if __name__ == "__main__":
    main()

