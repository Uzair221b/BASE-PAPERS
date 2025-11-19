"""
Diagnose Why Preprocessed Training Won't Start
==============================================
Find the exact problem preventing preprocessed training from starting
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

print("="*70)
print("DIAGNOSING PREPROCESSED TRAINING ISSUE")
print("="*70)
print()

# Check 1: Preprocessed images exist
print("1. Checking if preprocessed images exist...")
preprocessed_dir = 'processed_datasets/eyepacs_train'
rg_path = os.path.join(preprocessed_dir, 'RG')
nrg_path = os.path.join(preprocessed_dir, 'NRG')

if not os.path.exists(rg_path):
    print(f"   [ERROR] RG folder not found: {rg_path}")
    sys.exit(1)
else:
    rg_files = [f for f in os.listdir(rg_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   [OK] RG folder exists: {len(rg_files)} images")

if not os.path.exists(nrg_path):
    print(f"   [ERROR] NRG folder not found: {nrg_path}")
    sys.exit(1)
else:
    nrg_files = [f for f in os.listdir(nrg_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   [OK] NRG folder exists: {len(nrg_files)} images")

print()

# Check 2: Test loading images
print("2. Testing image loading...")
test_count = 0
error_count = 0
errors = []

for i, img_file in enumerate(rg_files[:10]):  # Test first 10
    img_path = os.path.join(rg_path, img_file)
    try:
        img = cv2.imread(img_path)
        if img is None:
            error_count += 1
            errors.append(f"Cannot read: {img_file}")
        else:
            test_count += 1
            # Check if it's actually a valid image
            if img.shape[0] == 0 or img.shape[1] == 0:
                error_count += 1
                errors.append(f"Invalid image: {img_file}")
    except Exception as e:
        error_count += 1
        errors.append(f"{img_file}: {e}")

print(f"   Tested {test_count + error_count} images")
print(f"   Success: {test_count}, Errors: {error_count}")

if error_count > 0:
    print("   [WARNING] Some images have errors:")
    for err in errors[:5]:  # Show first 5 errors
        print(f"     - {err}")
else:
    print("   [OK] All test images loaded successfully")

print()

# Check 3: Memory check
print("3. Checking memory availability...")
try:
    import psutil
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    used_percent = memory.percent
    
    print(f"   Total RAM: {total_gb:.2f} GB")
    print(f"   Available: {available_gb:.2f} GB")
    print(f"   Used: {used_percent:.1f}%")
    
    if available_gb < 2:
        print("   [WARNING] Low memory! Might cause training to fail")
    elif available_gb < 4:
        print("   [WARNING] Limited memory - might have issues")
    else:
        print("   [OK] Sufficient memory available")
except ImportError:
    print("   [INFO] psutil not installed - cannot check memory")
except Exception as e:
    print(f"   [INFO] Could not check memory: {e}")

print()

# Check 4: Test full loading process
print("4. Testing full data loading process...")
def load_test(data_dir, class_name, label, max_samples=100):
    images = []
    labels = []
    class_path = os.path.join(data_dir, class_name)
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_samples]
    
    print(f"   Loading {len(image_files)} images from {class_name}...")
    
    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"   [ERROR] {img_file}: {e}")
            continue
    
    return images, labels

try:
    print("   Testing RG loading...")
    rg_test, _ = load_test(preprocessed_dir, 'RG', 1, max_samples=100)
    print(f"   [OK] Loaded {len(rg_test)} RG images")
    
    print("   Testing NRG loading...")
    nrg_test, _ = load_test(preprocessed_dir, 'NRG', 0, max_samples=100)
    print(f"   [OK] Loaded {len(nrg_test)} NRG images")
    
    if len(rg_test) > 0 and len(nrg_test) > 0:
        X_test = np.array(rg_test + nrg_test)
        print(f"   [OK] Combined array shape: {X_test.shape}")
        print(f"   [OK] Data range: [{X_test.min():.3f}, {X_test.max():.3f}]")
        print("   [OK] Data loading works correctly!")
    else:
        print("   [ERROR] Could not load enough images for testing")
        
except Exception as e:
    print(f"   [ERROR] Data loading failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Check 5: Check script for errors
print("5. Checking training script...")
script_path = 'restart_preprocessed_training.py'
if os.path.exists(script_path):
    print(f"   [OK] Script exists: {script_path}")
    
    # Check for common issues
    with open(script_path, 'r') as f:
        content = f.read()
        
    if 'processed_datasets/eyepacs_train' in content:
        print("   [OK] Correct path to preprocessed images")
    else:
        print("   [WARNING] Path might be wrong")
        
    if 'load_preprocessed_images' in content:
        print("   [OK] Loading function exists")
    else:
        print("   [WARNING] Loading function might be missing")
else:
    print(f"   [ERROR] Script not found: {script_path}")

print()

# Summary
print("="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

if error_count == 0 and test_count > 0:
    print("[OK] Images load correctly")
    print("Problem might be:")
    print("  1. Memory issue (running two trainings)")
    print("  2. Script error during model building")
    print("  3. Process crashing silently")
    print()
    print("Solution: Try running with smaller batch size or less data")
elif error_count > 0:
    print("[PROBLEM] Some images have errors")
    print("Need to fix image loading first")
else:
    print("[UNKNOWN] Need more investigation")

print("="*70)

