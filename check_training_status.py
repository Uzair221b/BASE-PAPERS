"""
Check Training Status
=====================
"""

import os
from pathlib import Path
from datetime import datetime

print("="*70)
print("TRAINING STATUS CHECK")
print("="*70)
print()

# Check latest log
log_dir = Path('training_logs')
if log_dir.exists():
    log_files = list(log_dir.glob('training_*.log'))
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"Latest log: {latest_log.name}")
        print(f"Last updated: {datetime.fromtimestamp(os.path.getmtime(latest_log))}")
        print()
        
        # Read last 30 lines
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            print("Last 30 lines:")
            print("-"*70)
            for line in lines[-30:]:
                print(line.rstrip())
            print("-"*70)
        print()

# Check for model files
models_dir = Path('models')
if models_dir.exists():
    model_files = list(models_dir.glob('*.h5'))
    if model_files:
        print("Model files found:")
        for mf in sorted(model_files, key=os.path.getmtime, reverse=True):
            size_mb = os.path.getsize(mf) / (1024*1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(mf))
            print(f"  - {mf.name} ({size_mb:.1f} MB, modified: {mod_time})")
        print()

# Check if Python processes are running
import subprocess
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                          capture_output=True, text=True)
    if 'python.exe' in result.stdout:
        print("Python processes running:")
        print("  (Training might still be running)")
    else:
        print("No Python processes found - training is not running")
except:
    pass

print()
print("="*70)

