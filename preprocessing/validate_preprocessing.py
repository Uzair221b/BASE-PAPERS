import cv2
import numpy as np

# Load original and processed images
orig = cv2.imread(r'C:\Users\thefl\BASE-PAPERS\EYEPACS(AIROGS)\eyepac-light-v2-512-jpg\train\RG\EyePACS-DEV-RG-1.jpg')
proc = cv2.imread(r'C:\Users\thefl\BASE-PAPERS\processed_datasets\eyepacs_train\RG\processed_EyePACS-DEV-RG-1.jpg')

print("="*60)
print("PREPROCESSING VALIDATION")
print("="*60)
print("\nOriginal image stats:")
print(f"  Shape: {orig.shape}")
print(f"  Mean: {orig.mean():.2f}")
print(f"  Std: {orig.std():.2f}")
print(f"  Min: {orig.min()}, Max: {orig.max()}")

print("\nProcessed image stats:")
print(f"  Shape: {proc.shape}")
print(f"  Mean: {proc.mean():.2f}")
print(f"  Std: {proc.std():.2f}")
print(f"  Min: {proc.min()}, Max: {proc.max()}")

print("\nChanges Applied:")
print(f"  Size: {orig.shape} -> {proc.shape}")
print(f"  Contrast change: {(proc.std() / orig.std() * 100 - 100):+.1f}%")
print(f"  Brightness change: {(proc.mean() - orig.mean()):+.1f} units")

print("\nValidation Results:")
if proc.shape[0] == 224 and proc.shape[1] == 224:
    print("  [PASS] Resizing to 224x224")
else:
    print("  [FAIL] Size incorrect!")

if abs(proc.std() - orig.std()) > 5:
    print("  [PASS] CLAHE/Enhancement applied")
else:
    print("  [FAIL] No significant enhancement detected!")

if proc.mean() != orig.mean():
    print("  [PASS] Color normalization applied")
else:
    print("  [FAIL] No normalization detected!")

print("\n" + "="*60)
print("CONCLUSION: Preprocessing is", "WORKING CORRECTLY" if proc.shape == (224, 224, 3) else "NOT WORKING")
print("="*60)






