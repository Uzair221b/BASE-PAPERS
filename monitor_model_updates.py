"""
Monitor Model File Updates
==========================
Check if training is progressing by watching model file updates
"""

import os
import time
from pathlib import Path
from datetime import datetime

print("="*70)
print("MONITORING TRAINING PROGRESS (via model file updates)")
print("="*70)
print("Press Ctrl+C to stop")
print("="*70)
print()

model_pattern = "models/best_model_phase1_restart_*.h5"
last_size = 0
last_time = None
check_count = 0

try:
    while True:
        check_count += 1
        model_files = list(Path('models').glob('best_model_phase1_restart_*.h5'))
        
        if model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            current_size = os.path.getsize(latest_model)
            current_time = datetime.fromtimestamp(os.path.getmtime(latest_model))
            
            if last_time is None:
                last_time = current_time
                last_size = current_size
                print(f"Monitoring: {latest_model.name}")
                print(f"Initial size: {current_size/(1024*1024):.2f} MB")
                print(f"Last updated: {current_time}")
                print()
            
            if current_time > last_time or current_size != last_size:
                elapsed = (current_time - last_time).total_seconds() if last_time else 0
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Model updated!")
                print(f"  File: {latest_model.name}")
                print(f"  Size: {current_size/(1024*1024):.2f} MB")
                print(f"  Updated: {current_time}")
                if elapsed > 0:
                    print(f"  Time since last update: {elapsed:.0f} seconds")
                print()
                last_time = current_time
                last_size = current_size
            else:
                if check_count % 10 == 0:  # Print status every 10 checks
                    elapsed = (datetime.now() - last_time).total_seconds() if last_time else 0
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No updates in last {elapsed:.0f} seconds...")
                    print(f"  Last update: {last_time}")
                    print()
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No model file found yet...")
        
        time.sleep(5)  # Check every 5 seconds
        
except KeyboardInterrupt:
    print("\n\nStopped monitoring. Training continues in background.")

