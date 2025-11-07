"""
Complete preprocessing pipeline that applies all techniques and saves cleaned images
Applies the 5 core + 4 advanced preprocessing techniques
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from pipeline import GlaucomaPreprocessingPipeline
from data_loading import load_dataset, load_image
import config

# Import advanced preprocessing if available
try:
    import advanced_preprocessing
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False


def preprocess_all_images(input_folder: str, output_folder: str = "preprocessed_images"):
    """
    Apply all preprocessing techniques to images and save cleaned versions.
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder to save preprocessed images
    """
    print("="*70)
    print("COMPLETE PREPROCESSING PIPELINE")
    print("="*70)
    print("\nApplying ALL preprocessing techniques:")
    print("1. Scaling to 224×224 pixels")
    print("2. Cropping to center optic disc region")
    print("3. Color normalization (z-score)")
    print("4. CLAHE enhancement (optimized)")
    if config.ADVANCED_PREPROCESSING:
        print("5. Gamma correction")
        print("6. Bilateral filtering (noise reduction)")
        print("7. Enhanced CLAHE (LAB color space)")
        print("8. Image sharpening")
    print("="*70)
    
    # Load images
    print(f"\nLoading images from: {input_folder}")
    image_paths = load_dataset(input_folder)
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {input_folder}")
        return
    
    print(f"Found {len(image_paths)} images\n")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}\n")
    
    # Initialize pipeline
    pipeline = GlaucomaPreprocessingPipeline()
    
    # Process each image
    processed_count = 0
    failed_count = 0
    
    for i, image_path in enumerate(image_paths):
        image_name = Path(image_path).name
        print(f"[{i+1}/{len(image_paths)}] Processing: {image_name}", end=" ... ")
        
        try:
            # Load image
            image = load_image(image_path)
            
            # Apply ALL preprocessing techniques
            processed = pipeline.process_single_image(
                image,
                apply_clahe=True,
                apply_normalization=True,
                apply_augmentation=False  # Don't augment for saving clean images
            )
            
            # If advanced preprocessing is enabled, apply additional techniques
            if config.ADVANCED_PREPROCESSING and ADVANCED_AVAILABLE:
                processed = advanced_preprocessing.advanced_preprocessing_pipeline(processed)
            
            # Ensure image is in correct format for saving
            if processed.dtype != np.uint8:
                if processed.max() <= 1.0:
                    processed = (processed * 255).astype(np.uint8)
                else:
                    processed = processed.astype(np.uint8)
            
            processed = np.clip(processed, 0, 255).astype(np.uint8)
            
            # Save preprocessed image
            output_path = os.path.join(output_folder, f"processed_{image_name}")
            
            # Convert RGB to BGR for OpenCV
            if len(processed.shape) == 3:
                save_image = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            else:
                save_image = processed
            
            cv2.imwrite(output_path, save_image)
            
            processed_count += 1
            print(f"SAVED")
            
        except Exception as e:
            failed_count += 1
            print(f"ERROR: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total images: {len(image_paths)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"\nPreprocessed images saved to: {output_folder}/")
    print(f"\nPreprocessing Techniques Applied:")
    print(f"  [OK] Scaling: {config.IMAGE_SIZE}")
    print(f"  [OK] Cropping: {'Enabled' if config.CROP_ENABLED else 'Disabled'}")
    print(f"  [OK] Color Normalization: {config.NORMALIZATION_METHOD}")
    print(f"  [OK] CLAHE: Tile {config.CLAHE_TILE_SIZE}, Clip {config.CLAHE_CLIP_LIMIT}")
    if config.ADVANCED_PREPROCESSING:
        print(f"  [OK] Gamma Correction: {config.GAMMA_VALUE}")
        print(f"  [OK] Bilateral Filtering: Enabled")
        print(f"  [OK] Image Sharpening: Enabled")
    print(f"{'='*70}\n")


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Preprocess fundus images applying all techniques and save cleaned images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Preprocess images from folder:
    python preprocess_and_save.py --input preprocessing/glaucoma
    
  Custom output folder:
    python preprocess_and_save.py --input preprocessing/glaucoma --output cleaned_images
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input folder containing images to preprocess')
    parser.add_argument('--output', type=str, default='preprocessed_images',
                       help='Output folder for preprocessed images (default: preprocessed_images)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input):
        print(f"Error: Input folder not found: {args.input}")
        return
    
    # Run preprocessing
    preprocess_all_images(args.input, args.output)


if __name__ == "__main__":
    main()

