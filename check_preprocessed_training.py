"""
Check Preprocessed Training Status
===================================
See if it stopped and why
"""

import os
import subprocess
from datetime import datetime

print("="*70)
print("CHECKING PREPROCESSED TRAINING STATUS")
print("="*70)
print()

# Check Python processes
result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                       capture_output=True, text=True)

python_processes = result.stdout.count('python.exe')
print(f"Python processes running: {python_processes}")

if python_processes == 0:
    print("[WARNING] No Python processes found - all training stopped!")
elif python_processes == 1:
    print("[INFO] Only 1 Python process - preprocessed training might have stopped")
elif python_processes >= 2:
    print("[OK] Multiple Python processes - both trainings likely running")

print()

# Check for preprocessed model files
models_dir = 'models'
preprocessed_models = [f for f in os.listdir(models_dir) 
                      if f.startswith('best_model_preprocessed_') and f.endswith('.h5')]

print(f"Preprocessed model files found: {len(preprocessed_models)}")
if preprocessed_models:
    latest = max(preprocessed_models, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
    mod_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(models_dir, latest)))
    print(f"  Latest: {latest}")
    print(f"  Last updated: {mod_time}")
    
    # Check if recently updated (within last 5 minutes)
    time_diff = (datetime.now() - mod_time).total_seconds()
    if time_diff < 300:  # 5 minutes
        print(f"  [OK] Recently updated ({time_diff:.0f} seconds ago) - training likely running")
    else:
        print(f"  [WARNING] Not updated recently ({time_diff/60:.1f} minutes ago) - might have stopped")
else:
    print("  [WARNING] No preprocessed model files found!")
    print("  Training might have failed to start or crashed early")

print()
print("="*70)
print("DIAGNOSIS")
print("="*70)

if len(preprocessed_models) == 0:
    print("[PROBLEM] Preprocessed training likely failed to start or crashed")
    print("Possible reasons:")
    print("  1. Error loading preprocessed images")
    print("  2. Memory issue (two trainings at once)")
    print("  3. Script error")
    print()
    print("Should restart preprocessed training?")
elif python_processes < 2:
    print("[WARNING] Preprocessed training might have stopped")
    print("Check for errors or memory issues")
else:
    print("[OK] Preprocessed training appears to be running")

print("="*70)

