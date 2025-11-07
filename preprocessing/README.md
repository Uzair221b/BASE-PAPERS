# Glaucoma Preprocessing Pipeline

A comprehensive preprocessing pipeline for glaucoma detection in fundus images, implementing five state-of-the-art techniques based on literature review of leading research papers.

## Overview

This pipeline implements the most effective preprocessing techniques for glaucoma detection, as identified through systematic review of two key research papers:

- **Paper 1:** Esengönül & Cunha (2023) - "Glaucoma Detection using CNNs for Mobile Use"
- **Paper 2:** Milad et al. (2025) - "Code-Free Deep Learning Glaucoma Detection"

## Five Selected Best Techniques

### 1. ✅ Scaling to 224×224 Pixels
- **Purpose:** Standardize image dimensions for deep learning models
- **Why:** Essential for transfer learning and model compatibility
- **Configuration:** Defined in `config.IMAGE_SIZE`

### 2. ✅ Cropping to Center Optic Disc Region
- **Purpose:** Focus on relevant anatomical region
- **Methods:** Center crop or automatic optic disc detection
- **Configuration:** Enable in `config.CROP_ENABLED`

### 3. ✅ Color Normalization
- **Purpose:** Handle varying illumination and camera differences
- **Methods:** Z-score, min-max, or per-image normalization
- **Configuration:** Set in `config.NORMALIZATION_METHOD`

### 4. ✅ CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Purpose:** Enhance optic nerve visibility
- **Parameters:** Tile size (8×8), clip limit (2.0)
- **Configuration:** Enable in `config.CLAHE_ENABLED`
- **Performance:** Part of pipeline achieving 96.7% accuracy

### 5. ✅ Smart Class Balancing (1:2 Ratio)
- **Purpose:** Address severe class imbalance
- **Method:** Random undersampling to RG:NRG = 1:2
- **Configuration:** Enable in `config.BALANCE_CLASSES`
- **Performance:** Achieved 95% sensitivity @ 95% specificity

## Installation

```bash
# Install required dependencies
pip install opencv-python scikit-learn numpy
```

## Quick Start

### Single Image Processing

```python
from preprocessing.pipeline import quick_preprocess

# Preprocess a single fundus image
image_path = "path/to/fundus_image.jpg"
preprocessed = quick_preprocess(image_path)
```

### Batch Processing

```python
from preprocessing.pipeline import GlaucomaPreprocessingPipeline
from preprocessing import data_loading

# Initialize pipeline
pipeline = GlaucomaPreprocessingPipeline()

# Load images
image_paths = data_loading.load_dataset("data/raw/images")
labels = np.array([0, 1, 0, 1, ...])  # Your labels

# Process entire dataset
X_processed, y_processed = pipeline.load_and_process_images(
    image_paths,
    labels,
    balance_classes=True,
    apply_augmentation=True
)

# Get train/validation/test splits
X_train, y_train, X_val, y_val, X_test, y_test = pipeline.get_train_val_test_split(
    X_processed,
    y_processed
)
```

### Custom Single Image Processing

```python
from preprocessing.pipeline import GlaucomaPreprocessingPipeline
from preprocessing import data_loading
import cv2

# Initialize pipeline
pipeline = GlaucomaPreprocessingPipeline()

# Load and process
image = cv2.imread("path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

processed = pipeline.process_single_image(
    image,
    apply_clahe=True,
    apply_normalization=True,
    apply_augmentation=False
)
```

## Configuration

Edit `config.py` to customize the preprocessing pipeline:

```python
# Image preprocessing
IMAGE_SIZE = (224, 224)  # Standard size

# Cropping
CROP_ENABLED = True
CROP_SIZE = (224, 224)

# Color normalization
NORMALIZATION_ENABLED = True
NORMALIZATION_METHOD = 'z_score'  # Options: 'z_score', 'min_max', 'per_image'

# CLAHE
CLAHE_ENABLED = True
CLAHE_TILE_SIZE = (8, 8)
CLAHE_CLIP_LIMIT = 2.0

# Class balancing
BALANCE_CLASSES = True
TARGET_RATIO = (1, 2)  # RG:NRG ratio

# Data augmentation
AUGMENTATION_ENABLED = True
ZOOM_FACTOR = 0.035
ROTATION_RANGE = 0.025
```

## Pipeline Architecture

```
Input Images
    ↓
[1. Scale to 224×224]
    ↓
[2. Crop to Center Optic Disc]
    ↓
[3. Color Normalization]
    ↓
[4. CLAHE Enhancement]
    ↓
[5. Data Augmentation (Optional)]
    ↓
[6. Class Balancing]
    ↓
Preprocessed Dataset
```

## Module Structure

```
preprocessing/
├── __init__.py              # Package initialization
├── config.py                 # Configuration parameters
├── data_loading.py           # Load and scale images
├── cropping.py               # Crop to optic disc region
├── color_normalization.py    # Color normalization methods
├── clahe_processing.py       # CLAHE enhancement
├── class_balancing.py        # Handle class imbalance
├── data_augmentation.py      # Data augmentation
├── pipeline.py              # Main orchestrator
└── README.md                # This file
```

## Example Usage

### Example 1: Complete Pipeline

```python
import numpy as np
from preprocessing import GlaucomaPreprocessingPipeline, data_loading

# Initialize
pipeline = GlaucomaPreprocessingPipeline()

# Load dataset
image_paths = data_loading.load_dataset("data/raw/train")
labels = np.random.randint(0, 2, len(image_paths))  # Binary labels

# Process with all techniques
X_processed, y_processed = pipeline.load_and_process_images(
    image_paths,
    labels,
    balance_classes=True,
    apply_augmentation=True
)

print(f"Processed {len(X_processed)} images")
print(f"Class distribution: {np.bincount(y_processed)}")
```

### Example 2: Custom Configuration

```python
import preprocessing.config as config

# Modify configuration
config.CLAHE_CLIP_LIMIT = 3.0  # Higher contrast
config.TARGET_RATIO = (1, 3)   # Different class ratio
config.IMAGE_SIZE = (256, 256) # Larger image size

# Use modified configuration in pipeline
pipeline = GlaucomaPreprocessingPipeline()
```

### Example 3: Individual Modules

```python
from preprocessing import data_loading, cropping, color_normalization, clahe_processing

# Load image
image = data_loading.load_image("image.jpg")

# Scale
image = data_loading.scale_image(image, (224, 224))

# Crop
image = cropping.smart_crop(image)

# Normalize
image = color_normalization.normalize_color(image)

# Apply CLAHE
image = clahe_processing.apply_clahe(image)

# Result: Preprocessed image
```

## Performance Metrics

Based on research papers, this pipeline has demonstrated:

- **Accuracy:** Up to 96.7% (Paper 1)
- **Area under PR Curve (AuPRC):** 0.988 (Paper 2)
- **Sensitivity @ 95% Specificity:** 95% (Paper 2)
- **AUC:** 0.960-0.994 on external validation (Paper 2)

## Dataset Requirements

- **Format:** JPG, PNG, JPEG
- **Resolution:** Flexible (will be resized to 224×224)
- **Labels:** Binary (0=Normal, 1=Glaucoma)
- **Directory structure:** 
  ```
  data/
  ├── raw/
  │   └── images/
  │       ├── image1.jpg
  │       ├── image2.jpg
  │       └── ...
  └── preprocessed/
      ├── X.npy
      └── y.npy
  ```

## Citation

If you use this preprocessing pipeline, please cite the original papers:

```
@article{esengonul2023glaucoma,
  title={Glaucoma Detection using Convolutional Neural Networks for Mobile Use},
  author={Esengönül, Meltem and Cunha, António},
  journal={Procedia Computer Science},
  volume={219},
  pages={1153--1160},
  year={2023}
}

@article{milad2025coded,
  title={Code-Free Deep Learning Glaucoma Detection on Color Fundus Images},
  author={Milad, Daniel and Antaki, Fares and others},
  journal={Ophthalmology Science},
  volume={5},
  number={4},
  year={2025}
}
```

## License

This implementation is provided for research purposes.

## Contact

For questions or issues, please open an issue on the repository.




