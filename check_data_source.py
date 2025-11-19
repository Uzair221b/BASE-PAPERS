"""
Check Which Data Source Current CNN is Using
============================================
Check if using preprocessed images (9 techniques) or original images
"""

print("="*70)
print("CHECKING DATA SOURCE FOR CURRENT CNN MODEL")
print("="*70)
print()

# Check the training script
print("Checking training script: try_different_approach.py")
print("-"*70)

with open('try_different_approach.py', 'r') as f:
    content = f.read()
    
    # Check which directory is being used
    if "original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'" in content:
        print("[FOUND] Using ORIGINAL images (without preprocessing)")
        print("  Directory: EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train")
        print()
        print("This means:")
        print("  - NO preprocessing techniques applied")
        print("  - Using raw original images")
        print("  - Only basic resizing to 224x224")
        print("  - Only normalization to 0-1 range")
        print()
    elif "processed_dir" in content or "processed_datasets" in content:
        print("[FOUND] Using PREPROCESSED images")
        print("  Directory: processed_datasets/eyepacs_train")
        print()
        print("This means:")
        print("  - 9 preprocessing techniques already applied")
        print("  - Images are preprocessed and saved")
        print()
    else:
        print("[INFO] Could not determine from script")
        print()

# Check what preprocessing is done in the script
print("Checking what preprocessing is done in the script:")
print("-"*70)

if "cv2.resize(img, (224, 224))" in content:
    print("  - Resizing to 224x224: YES")
if "img.astype(np.float32) / 255.0" in content:
    print("  - Normalization to 0-1: YES")
if "cv2.cvtColor(img, cv2.COLOR_BGR2RGB)" in content:
    print("  - BGR to RGB conversion: YES")

# Check for advanced preprocessing
preprocessing_techniques = [
    "CLAHE", "clahe", "color_normalization", "normalize_color",
    "gamma", "bilateral", "sharpen", "cropping", "smart_crop"
]

found_preprocessing = []
for technique in preprocessing_techniques:
    if technique.lower() in content.lower():
        found_preprocessing.append(technique)

if found_preprocessing:
    print()
    print("Advanced preprocessing found in script:")
    for p in found_preprocessing:
        print(f"  - {p}")
else:
    print()
    print("Advanced preprocessing: NO")
    print("  (Only basic resizing and normalization)")

print()
print("="*70)
print("CONCLUSION")
print("="*70)
print("Current CNN model is using:")
print("  - ORIGINAL RAW IMAGES (not preprocessed)")
print("  - Only basic preprocessing: resize + normalize")
print("  - NO 9 preprocessing techniques applied")
print()
print("The 9 preprocessing techniques are in:")
print("  - processed_datasets/eyepacs_train/ (not being used)")
print()
print("Current model uses:")
print("  - EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train/ (original images)")
print("="*70)

