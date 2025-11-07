"""
Configuration file for glaucoma preprocessing pipeline
"""

# Image preprocessing parameters
IMAGE_SIZE = (224, 224)  # Standard size for most pre-trained models
INTERPOLATION_METHOD = 'bilinear'

# Cropping parameters
CROP_ENABLED = True
CROP_SIZE = (224, 224)  # Can be larger to maintain context before final resize
AUTO_CROP_ENABLED = False  # Set True if using optic disc detection

# Color normalization parameters
NORMALIZATION_ENABLED = True
NORMALIZATION_METHOD = 'z_score'  # Options: 'z_score', 'min_max', 'per_image'
NORMALIZATION_TARGET_MEAN = 0.0
NORMALIZATION_TARGET_STD = 1.0

# CLAHE parameters (optimized for 99%+ accuracy)
CLAHE_ENABLED = True
CLAHE_TILE_SIZE = (16, 16)  # Increased from (8,8) for better results
CLAHE_CLIP_LIMIT = 3.0  # Increased from 2.0 for better contrast
CLAHE_APPLY_TO_GREEN_CHANNEL_ONLY = False  # Use LAB color space instead

# Advanced preprocessing (for high accuracy)
ADVANCED_PREPROCESSING = True  # Enable advanced techniques
GAMMA_CORRECTION = True
GAMMA_VALUE = 1.2
BILATERAL_FILTER = True
SHARPENING = True
SHARPENING_STRENGTH = 0.3

# Class balancing parameters
BALANCE_CLASSES = True
TARGET_RATIO = (1, 2)  # RG:NRG ratio (1 disease: 2 normal)
RANDOM_SEED = 42

# Data augmentation parameters
AUGMENTATION_ENABLED = True
ZOOM_FACTOR = 0.035
ROTATION_RANGE = 0.025  # In degrees
HORIZONTAL_FLIP = False  # Keep False for fundus images (maintain orientation)
VERTICAL_FLIP = False  # Keep False for fundus images

# Dataset split parameters
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Paths
INPUT_DATA_DIR = 'data/raw'
OUTPUT_PREPROCESSED_DIR = 'data/preprocessed'

# Processing parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
