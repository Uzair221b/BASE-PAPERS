"""
Script to analyze fundus images and apply preprocessing pipeline
You can use this to preprocess your images and prepare them for classification
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from pipeline import GlaucomaPreprocessingPipeline, quick_preprocess
import config


def process_single_image(image_path: str, output_dir: str = "processed_output"):
    """
    Process a single fundus image and save the preprocessed version.
    
    Args:
        image_path: Path to input fundus image
        output_dir: Directory to save processed images
    """
    print(f"\nProcessing: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return None
    
    # Process image through pipeline
    try:
        preprocessed = quick_preprocess(image_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed image
        filename = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{filename}_processed.jpg")
        
        # Convert RGB to BGR for OpenCV save
        output_image = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image)
        
        print(f"  Original shape: {cv2.imread(image_path).shape[:2]}")
        print(f"  Preprocessed shape: {preprocessed.shape}")
        print(f"  Saved to: {output_path}")
        
        return preprocessed, output_path
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def process_directory(image_dir: str, output_dir: str = "processed_output"):
    """
    Process all images in a directory.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save processed images
    """
    from data_loading import load_dataset
    
    print(f"\nProcessing directory: {image_dir}")
    
    # Load all images
    image_paths = load_dataset(image_dir)
    
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Process each image
    processed_images = []
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] ", end="")
        result = process_single_image(str(image_path), output_dir)
        if result:
            processed_images.append(result)
    
    print(f"\n{'='*60}")
    print(f"Processing complete! {len(processed_images)} images processed.")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def display_preprocessing_steps(image_path: str):
    """
    Display all preprocessing steps applied to an image.
    
    Args:
        image_path: Path to input fundus image
    """
    import matplotlib.pyplot as plt
    
    print(f"\nDisplaying preprocessing steps for: {image_path}")
    
    # Load original
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    pipeline = GlaucomaPreprocessingPipeline()
    
    # Step 1: Scaling
    import data_loading
    scaled = data_loading.scale_image(original, config.IMAGE_SIZE)
    
    # Step 2: Cropping
    import cropping
    cropped = cropping.smart_crop(scaled)
    
    # Step 3: Normalization
    import color_normalization
    normalized = color_normalization.normalize_color(cropped)
    # Convert back to uint8 for display
    if normalized.dtype != np.uint8:
        normalized = (normalized * 255).astype(np.uint8)
        normalized = np.clip(normalized, 0, 255)
    
    # Step 4: CLAHE
    import clahe_processing
    clahe_enhanced = clahe_processing.apply_clahe(cropped)
    
    # Final processed
    final = pipeline.process_single_image(original, apply_augmentation=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f"Original\n{original.shape}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(scaled)
    axes[0, 1].set_title(f"Scaled\n{scaled.shape}")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cropped)
    axes[0, 2].set_title(f"Cropped\n{cropped.shape}")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(normalized)
    axes[1, 0].set_title("Normalized")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(clahe_enhanced)
    axes[1, 1].set_title("CLAHE Enhanced")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(final)
    axes[1, 2].set_title(f"Final Processed\n{final.shape}")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = "preprocessing_steps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(description='Process fundus images for glaucoma detection')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory containing images')
    parser.add_argument('--output', type=str, default='processed_output', 
                       help='Output directory for processed images')
    parser.add_argument('--visualize', action='store_true',
                       help='Display preprocessing steps (single image only)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Glaucoma Fundus Image Preprocessing")
    print("="*60)
    print(f"Configuration:")
    print(f"  Image size: {config.IMAGE_SIZE}")
    print(f"  CLAHE enabled: {config.CLAHE_ENABLED}")
    print(f"  Normalization enabled: {config.NORMALIZATION_ENABLED}")
    print(f"  Cropping enabled: {config.CROP_ENABLED}")
    print("="*60)
    
    if args.image:
        if args.visualize:
            display_preprocessing_steps(args.image)
        else:
            process_single_image(args.image, args.output)
    elif args.dir:
        process_directory(args.dir, args.output)
    else:
        print("\nUsage:")
        print("  Process single image:")
        print("    python analyze_images.py --image path/to/image.jpg")
        print("\n  Process directory:")
        print("    python analyze_images.py --dir path/to/images/")
        print("\n  Visualize preprocessing steps:")
        print("    python analyze_images.py --image path/to/image.jpg --visualize")
        print("\n  Custom output directory:")
        print("    python analyze_images.py --image path/to/image.jpg --output my_output/")


if __name__ == "__main__":
    main()

