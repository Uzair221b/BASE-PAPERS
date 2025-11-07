"""
Main preprocessing pipeline orchestrator for glaucoma detection
"""

import numpy as np
from typing import List, Tuple, Optional
import config

# Import all preprocessing modules
import data_loading
import cropping
import color_normalization
import clahe_processing
import class_balancing
import data_augmentation

# Try to import advanced preprocessing
try:
    import advanced_preprocessing
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False


class GlaucomaPreprocessingPipeline:
    """
    Complete preprocessing pipeline for glaucoma detection in fundus images.
    
    This pipeline implements the five selected best techniques:
    1. Scaling to 224×224 pixels
    2. Cropping to center optic disc region
    3. Color normalization
    4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    5. Smart class balancing (1:2 ratio)
    """
    
    def __init__(self):
        self.config = config
    
    def process_single_image(self, image: np.ndarray, 
                            apply_clahe: bool = True,
                            apply_normalization: bool = True,
                            apply_augmentation: bool = False) -> np.ndarray:
        """
        Process a single image through the preprocessing pipeline.
        
        Args:
            image: Input fundus image array
            apply_clahe: Whether to apply CLAHE
            apply_normalization: Whether to apply color normalization
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Step 1: Scale to target size (if needed)
        if processed.shape[:2] != config.IMAGE_SIZE:
            processed = data_loading.scale_image(processed, config.IMAGE_SIZE)
        
        # Step 2: Crop to center optic disc region
        if config.CROP_ENABLED:
            processed = cropping.smart_crop(processed, 
                                           crop_size=config.CROP_SIZE,
                                           auto_detect=config.AUTO_CROP_ENABLED)
            # Ensure final size is correct
            if processed.shape[:2] != config.IMAGE_SIZE:
                processed = data_loading.scale_image(processed, config.IMAGE_SIZE)
        
        # Step 3: Color normalization
        if apply_normalization and config.NORMALIZATION_ENABLED:
            processed = color_normalization.normalize_color(
                processed,
                method=config.NORMALIZATION_METHOD
            )
            # Convert back to uint8 if needed
            if processed.dtype != np.uint8:
                processed = (processed * 255).astype(np.uint8)
                processed = np.clip(processed, 0, 255)
        
        # Step 4: CLAHE or Advanced Preprocessing (for 99%+ accuracy)
        if config.ADVANCED_PREPROCESSING and ADVANCED_AVAILABLE:
            # Use advanced preprocessing pipeline for higher accuracy
            processed = advanced_preprocessing.advanced_preprocessing_pipeline(processed)
        elif apply_clahe and config.CLAHE_ENABLED:
            processed = clahe_processing.apply_clahe(
                processed,
                tile_size=config.CLAHE_TILE_SIZE,
                clip_limit=config.CLAHE_CLIP_LIMIT,
                apply_to_green_only=config.CLAHE_APPLY_TO_GREEN_CHANNEL_ONLY
            )
        
        # Step 5: Optional augmentation
        if apply_augmentation and config.AUGMENTATION_ENABLED:
            processed = data_augmentation.augment_image(processed)
        
        return processed
    
    def process_dataset(self, 
                       X: List[np.ndarray],
                       y: np.ndarray,
                       balance_classes: bool = True,
                       apply_augmentation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an entire dataset through the preprocessing pipeline.
        
        Args:
            X: List of images
            y: Label array
            balance_classes: Whether to apply class balancing
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Processed X and y arrays
        """
        print("Starting preprocessing pipeline...")
        print(f"Initial dataset size: {len(X)} images")
        
        # Process each image
        processed_images = []
        for i, image in enumerate(X):
            if (i + 1) % 100 == 0:
                print(f"Processing image {i+1}/{len(X)}...")
            
            processed_img = self.process_single_image(
                image,
                apply_clahe=True,
                apply_normalization=True,
                apply_augmentation=apply_augmentation
            )
            processed_images.append(processed_img)
        
        X_processed = np.array(processed_images)
        y_processed = y.copy()
        
        print(f"\nPreprocessing complete: {len(X_processed)} images processed")
        
        # Class balancing
        if balance_classes and config.BALANCE_CLASSES:
            print("\nApplying class balancing...")
            X_processed, y_processed = class_balancing.random_undersampling(
                X_processed,
                y_processed,
                target_ratio=config.TARGET_RATIO,
                random_seed=config.RANDOM_SEED
            )
        
        return X_processed, y_processed
    
    def load_and_process_images(self,
                                image_paths: List[str],
                                labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from paths and process them through the pipeline.
        
        Args:
            image_paths: List of paths to image files
            labels: Corresponding labels
            
        Returns:
            Processed X and y arrays
        """
        print(f"Loading {len(image_paths)} images...")
        
        X_loaded = []
        for path in image_paths:
            image = data_loading.load_and_scale(path, config.IMAGE_SIZE)
            X_loaded.append(image)
        
        return self.process_dataset(X_loaded, labels)
    
    def get_train_val_test_split(self,
                                 X: List[np.ndarray],
                                 y: np.ndarray) -> Tuple:
        """
        Get stratified train/validation/test splits with balancing.
        
        Args:
            X: List of images
            y: Label array
            
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        return class_balancing.get_balanced_split(
            X, y,
            train_ratio=config.TRAIN_SPLIT,
            val_ratio=config.VAL_SPLIT,
            test_ratio=config.TEST_SPLIT,
            target_ratio=config.TARGET_RATIO,
            random_seed=config.RANDOM_SEED
        )


def quick_preprocess(image_path: str) -> np.ndarray:
    """
    Quick preprocessing function for a single image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image
    """
    # Load image
    image = data_loading.load_image(image_path)
    
    # Process through pipeline
    pipeline = GlaucomaPreprocessingPipeline()
    processed = pipeline.process_single_image(image)
    
    return processed


def batch_preprocess(image_dir: str, label_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch preprocessing for entire directory.
    
    Args:
        image_dir: Directory containing images
        label_file: Optional file with labels (CSV format: filename,label)
        
    Returns:
        Processed X and y arrays
    """
    import pandas as pd
    
    # Load image paths
    image_paths = data_loading.load_dataset(image_dir)
    
    # Load labels if provided
    if label_file:
        df = pd.read_csv(label_file)
        labels = df['label'].values
    else:
        # Assume all images are in same directory structure
        # You would need to implement label loading based on your structure
        raise NotImplementedError("Label loading must be implemented based on your dataset structure")
    
    # Process through pipeline
    pipeline = GlaucomaPreprocessingPipeline()
    X, y = pipeline.load_and_process_images(image_paths, labels)
    
    return X, y


if __name__ == "__main__":
    print("=" * 60)
    print("Glaucoma Preprocessing Pipeline")
    print("=" * 60)
    print("\nFive Selected Techniques:")
    print("1. Scaling to 224×224 pixels")
    print("2. Cropping to center optic disc region")
    print("3. Color normalization (z-score)")
    print("4. CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    print("5. Smart class balancing (1:2 ratio)")
    print("\nPipeline loaded successfully!")
    print("\nConfiguration:")
    print(f"  Image size: {config.IMAGE_SIZE}")
    print(f"  CLAHE enabled: {config.CLAHE_ENABLED}")
    print(f"  Normalization enabled: {config.NORMALIZATION_ENABLED}")
    print(f"  Class balancing enabled: {config.BALANCE_CLASSES}")
    print(f"  Augmentation enabled: {config.AUGMENTATION_ENABLED}")
