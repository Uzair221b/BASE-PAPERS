"""
Cleanup Failed Training Files
==============================
Delete incorrectly trained models from failed attempts
"""

import os
from pathlib import Path
from datetime import datetime

print("="*70)
print("CLEANING UP FAILED TRAINING FILES")
print("="*70)
print()

# Files to delete (from failed training with 50% accuracy)
files_to_delete = [
    'models/best_model_phase1.h5',  # Failed Phase 1 (50% accuracy)
]

# Also check for any other failed training files
failed_logs = [
    'training_logs/training_20251119_022812.log',  # Failed training log
]

print("Files to delete:")
print("-"*70)

deleted_count = 0
for file_path in files_to_delete:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"  - {file_path}")
        print(f"    Size: {file_size:.2f} MB")
        print(f"    Modified: {mod_time}")
        try:
            os.remove(file_path)
            print(f"    [DELETED]")
            deleted_count += 1
        except Exception as e:
            print(f"    [ERROR] Could not delete: {e}")
    else:
        print(f"  - {file_path} (not found)")

print()

# Keep the good files
print("Files to KEEP (correct training):")
print("-"*70)
keep_files = [
    'models/best_model_20251110_193527.h5',  # Original model (epoch 21)
    'models/best_model_phase1_restart_*.h5',  # Current Phase 1 restart
]

for pattern in keep_files:
    if '*' in pattern:
        import glob
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.exists(match):
                size = os.path.getsize(match) / (1024*1024)
                mod_time = datetime.fromtimestamp(os.path.getmtime(match))
                print(f"  - {match}")
                print(f"    Size: {size:.2f} MB, Modified: {mod_time}")
    else:
        if os.path.exists(pattern):
            size = os.path.getsize(pattern) / (1024*1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(pattern))
            print(f"  - {pattern}")
            print(f"    Size: {size:.2f} MB, Modified: {mod_time}")

print()
print("="*70)
print(f"CLEANUP COMPLETE - Deleted {deleted_count} file(s)")
print("="*70)
print()
print("Remaining files are from correct training attempts.")

