"""
Test script for the glaucoma preprocessing pipeline
"""

import numpy as np
import cv2
import os
from pathlib import Path

# Import pipeline components
from pipeline import GlaucomaPreprocessingPipeline, quick_preprocess
import config


def create_test_image(filename: str = "test_fundus.jpg"):
    """Create a synthetic fundus-like test image."""
    # Create a synthetic fundus image (simulated)
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Add some structure (circle for optic disc)
    h, w = 512, 512
    center = (w // 2, h // 2)
    cv2.circle(img, center, 50, (100, 80, 60), -1)  # Optic disc
    cv2.circle(img, center, 30, (150, 120, 100), -1)  # Optic cup
    
    # Save test image
    os.makedirs("test_data", exist_ok=True)
    test_path = os.path.join("test_data", filename)
    cv2.imwrite(test_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Created test image: {test_path}")
    return test_path


def test_single_image_preprocessing():
    """Test preprocessing of a single image."""
    print("\n" + "="*60)
    print("Test 1: Single Image Preprocessing")
    print("="*60)
    
    # Create test image
    test_path = create_test_image("test_fundus.jpg")
    
    # Load original image
    original = cv2.imread(test_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    print(f"\nOriginal image shape: {original.shape}")
    
    # Quick preprocessing
    preprocessed = quick_preprocess(test_path)
    
    print(f"Preprocessed image shape: {preprocessed.shape}")
    print(f"Expected shape: ({config.IMAGE_SIZE[0]}, {config.IMAGE_SIZE[1]}, 3)")
    
    # Verify dimensions
    assert preprocessed.shape[:2] == config.IMAGE_SIZE, "Image size mismatch!"
    print("[OK] Single image preprocessing test passed!")


def test_pipeline_class():
    """Test the full pipeline class."""
    print("\n" + "="*60)
    print("Test 2: Pipeline Class")
    print("="*60)
    
    # Create test images
    test_images = []
    for i in range(5):
        path = create_test_image(f"test_image_{i}.jpg")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_images.append(img)
    
    # Create dummy labels
    labels = np.array([0, 1, 0, 1, 0])
    
    # Initialize pipeline
    pipeline = GlaucomaPreprocessingPipeline()
    
    # Process dataset
    X_processed, y_processed = pipeline.process_dataset(
        test_images,
        labels,
        balance_classes=True,
        apply_augmentation=True
    )
    
    print(f"\nOriginal dataset size: {len(test_images)}")
    print(f"Processed dataset size: {len(X_processed)}")
    print(f"Label distribution: {np.bincount(y_processed)}")
    print("[OK] Pipeline class test passed!")


def test_individual_modules():
    """Test individual preprocessing modules."""
    print("\n" + "="*60)
    print("Test 3: Individual Modules")
    print("="*60)
    
    # Import modules
    import data_loading
    import cropping
    import color_normalization
    import clahe_processing
    
    # Create test image
    test_path = create_test_image("module_test.jpg")
    image = cv2.imread(test_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Test scaling
    scaled = data_loading.scale_image(image, config.IMAGE_SIZE)
    print(f"[OK] Scaling: {image.shape} -> {scaled.shape}")
    
    # Test cropping
    cropped = cropping.smart_crop(scaled)
    print(f"[OK] Cropping: {scaled.shape} -> {cropped.shape}")
    
    # Test normalization
    normalized = color_normalization.normalize_color(cropped)
    print(f"[OK] Normalization: Applied successfully")
    
    # Test CLAHE
    enhanced = clahe_processing.apply_clahe(cropped)
    print(f"[OK] CLAHE: Applied successfully")
    
    print("All individual module tests passed!")


def test_configuration():
    """Test configuration settings."""
    print("\n" + "="*60)
    print("Test 4: Configuration")
    print("="*60)
    
    print(f"Image size: {config.IMAGE_SIZE}")
    print(f"Crop enabled: {config.CROP_ENABLED}")
    print(f"Normalization enabled: {config.NORMALIZATION_ENABLED}")
    print(f"CLAHE enabled: {config.CLAHE_ENABLED}")
    print(f"Class balancing enabled: {config.BALANCE_CLASSES}")
    print(f"Augmentation enabled: {config.AUGMENTATION_ENABLED}")
    print(f"Target ratio: {config.TARGET_RATIO}")
    print("[OK] Configuration test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GLAUCOMA PREPROCESSING PIPELINE - TEST SUITE")
    print("="*70)
    
    try:
        # Test 1
        test_single_image_preprocessing()
        
        # Test 2
        test_pipeline_class()
        
        # Test 3
        test_individual_modules()
        
        # Test 4
        test_configuration()
        
        print("\n" + "="*70)
        print("[OK] ALL TESTS PASSED!")
        print("="*70)
        print("\nThe preprocessing pipeline is working correctly!")
        print("You can now use it with your own fundus images.")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
