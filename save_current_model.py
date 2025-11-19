"""
Save Current Model - Emergency Backup
=====================================
Save the current best model before restarting
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

print("="*70)
print("SAVING CURRENT MODEL - EMERGENCY BACKUP")
print("="*70)
print()

models_dir = 'models'
backup_dir = 'models/backup'

# Create backup directory
os.makedirs(backup_dir, exist_ok=True)

# Find latest model
model_files = [f for f in os.listdir(models_dir) 
               if f.startswith('best_cnn_model_') and f.endswith('.h5')]

if not model_files:
    print("[ERROR] No model file found!")
    exit(1)

latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
model_path = os.path.join(models_dir, latest_model)

# Create backup with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_name = f"SAVED_MODEL_EPOCH56_92.40pct_{timestamp}.h5"
backup_path = os.path.join(backup_dir, backup_name)

print(f"Latest model: {latest_model}")
print(f"Backup name: {backup_name}")
print()

# Copy model
print("Copying model...")
shutil.copy2(model_path, backup_path)

# Verify
if os.path.exists(backup_path):
    size_mb = os.path.getsize(backup_path) / (1024 * 1024)
    print(f"[SUCCESS] Model saved to: {backup_path}")
    print(f"Size: {size_mb:.2f} MB")
    print()
    
    # Also save as a named file for easy access
    easy_access = os.path.join(backup_dir, "CURRENT_BEST_MODEL_EPOCH56.h5")
    shutil.copy2(model_path, easy_access)
    print(f"[SUCCESS] Also saved as: CURRENT_BEST_MODEL_EPOCH56.h5")
else:
    print("[ERROR] Backup failed!")

print()
print("="*70)
print("BACKUP COMPLETE")
print("="*70)
print(f"Original: {model_path}")
print(f"Backup: {backup_path}")
print(f"Easy access: {easy_access}")
print("="*70)

