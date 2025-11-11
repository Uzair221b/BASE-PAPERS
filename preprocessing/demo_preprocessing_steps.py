"""
Demonstration of All 9 Preprocessing Techniques
Saves each step as a separate image for visualization
"""

import cv2
import numpy as np
import os
from pathlib import Path

# Import our preprocessing modules
from config import *
from data_loading import load_image, scale_image
from cropping import smart_crop
from color_normalization import z_score_normalize
from clahe_processing import apply_clahe, apply_clahe_to_l_channel
from advanced_preprocessing import apply_gamma_correction, apply_bilateral_filter, enhanced_clahe, sharpen_image

def demonstrate_preprocessing_steps(input_image_path, output_folder):
    """
    Apply each preprocessing step and save intermediate results
    """
    
    # Create output folder
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PREPROCESSING TECHNIQUES DEMONSTRATION")
    print("="*70)
    print(f"Input image: {input_image_path}")
    print(f"Output folder: {output_folder}")
    print()
    
    # Load original image
    print("[Step 0] Loading original image...")
    original = cv2.imread(input_image_path)
    if original is None:
        print(f"ERROR: Could not load image: {input_image_path}")
        return
    
    cv2.imwrite(str(output_dir / "step_0_original.jpg"), original)
    print(f"  [OK] Saved: step_0_original.jpg (Size: {original.shape})")
    
    # Step 1: Scaling
    print("\n[Step 1] Image Scaling (224x224)...")
    scaled = scale_image(original, IMAGE_SIZE)
    cv2.imwrite(str(output_dir / "step_1_scaled_224x224.jpg"), scaled)
    print(f"  [OK] Saved: step_1_scaled_224x224.jpg (Size: {scaled.shape})")
    print(f"  Effect: Standardized to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels")
    
    # Step 2: Smart Cropping
    print("\n[Step 2] Smart Cropping (Center Optic Disc)...")
    cropped = smart_crop(scaled.copy())
    cv2.imwrite(str(output_dir / "step_2_cropped.jpg"), cropped)
    print(f"  [OK] Saved: step_2_cropped.jpg")
    print(f"  Effect: Centered on optic disc region")
    
    # Step 3: Color Normalization
    print("\n[Step 3] Color Normalization (Z-Score)...")
    normalized = z_score_normalize(cropped.copy())
    # Convert to uint8 for saving
    norm_display = ((normalized - normalized.min()) / (normalized.max() - normalized.min()) * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "step_3_color_normalized.jpg"), norm_display)
    print(f"  [OK] Saved: step_3_color_normalized.jpg")
    print(f"  Effect: Standardized color distribution")
    
    # Step 4: CLAHE Enhancement (RGB)
    print("\n[Step 4] CLAHE Enhancement (16x16 tiles, clip 3.0)...")
    clahe_rgb = apply_clahe(cropped.copy(), CLAHE_TILE_SIZE, CLAHE_CLIP_LIMIT)
    cv2.imwrite(str(output_dir / "step_4_clahe_rgb.jpg"), clahe_rgb)
    print(f"  [OK] Saved: step_4_clahe_rgb.jpg")
    print(f"  Effect: Enhanced contrast (RGB channels)")
    
    # Step 5: Gamma Correction
    print("\n[Step 5] Gamma Correction (gamma=1.2)...")
    gamma_corrected = apply_gamma_correction(clahe_rgb.copy(), GAMMA_VALUE)
    cv2.imwrite(str(output_dir / "step_5_gamma_corrected.jpg"), gamma_corrected)
    print(f"  [OK] Saved: step_5_gamma_corrected.jpg")
    print(f"  Effect: Adjusted brightness (gamma={GAMMA_VALUE})")
    
    # Step 6: Bilateral Filtering
    print("\n[Step 6] Bilateral Filtering (Noise Reduction)...")
    bilateral = apply_bilateral_filter(gamma_corrected.copy())
    cv2.imwrite(str(output_dir / "step_6_bilateral_filtered.jpg"), bilateral)
    print(f"  [OK] Saved: step_6_bilateral_filtered.jpg")
    print(f"  Effect: Reduced noise while preserving edges")
    
    # Step 7: LAB-CLAHE Enhancement
    print("\n[Step 7] Enhanced LAB-CLAHE...")
    lab_clahe = enhanced_clahe(bilateral.copy(), CLAHE_TILE_SIZE, CLAHE_CLIP_LIMIT)
    cv2.imwrite(str(output_dir / "step_7_lab_clahe.jpg"), lab_clahe)
    print(f"  [OK] Saved: step_7_lab_clahe.jpg")
    print(f"  Effect: Advanced contrast in LAB color space")
    
    # Step 8: Adaptive Sharpening
    print("\n[Step 8] Adaptive Sharpening...")
    sharpened = sharpen_image(lab_clahe.copy(), SHARPENING_STRENGTH)
    cv2.imwrite(str(output_dir / "step_8_sharpened.jpg"), sharpened)
    print(f"  [OK] Saved: step_8_sharpened.jpg")
    print(f"  Effect: Enhanced fine details")
    
    # Step 9: Final preprocessed image
    print("\n[Step 9] Final Preprocessed Image...")
    final = sharpened.copy()
    cv2.imwrite(str(output_dir / "step_9_final_preprocessed.jpg"), final)
    print(f"  [OK] Saved: step_9_final_preprocessed.jpg")
    print(f"  Effect: All 9 techniques applied!")
    
    # Create comparison: Original vs Final
    print("\n[Creating Comparison]...")
    original_resized = cv2.resize(original, IMAGE_SIZE)
    comparison = np.hstack([original_resized, final])
    cv2.imwrite(str(output_dir / "comparison_before_after.jpg"), comparison)
    print(f"  [OK] Saved: comparison_before_after.jpg")
    print(f"  Effect: Side-by-side comparison")
    
    print()
    print("="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print(f"\nAll {10} images saved to: {output_folder}")
    print("\nYou can now:")
    print("  1. Open the folder to view each preprocessing step")
    print("  2. See the before/after comparison")
    print("  3. Understand what each technique does")
    print("\nNext: Run full batch preprocessing on 8,000 images!")
    print("="*70)

if __name__ == "__main__":
    import sys
    
    # Use a sample image from EYEPACS or test_data
    # Check multiple possible locations
    possible_images = [
        "../EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train/RG",
        "../EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train/NRG",
        "../test_data",
        "../ACRIMA/test/Glaucoma",
    ]
    
    sample_image = None
    for folder in possible_images:
        folder_path = Path(folder)
        if folder_path.exists():
            image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
            if image_files:
                sample_image = str(image_files[0])
                break
    
    if sample_image is None:
        print("ERROR: No sample image found!")
        print("Please provide image path: python demo_preprocessing_steps.py [image_path]")
        sys.exit(1)
    
    output_folder = "../preprocessing_demo"
    
    # Allow custom input
    if len(sys.argv) > 1:
        sample_image = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    demonstrate_preprocessing_steps(sample_image, output_folder)

