"""
Continuous Monitor for Preprocessed Training
=============================================
Monitors and auto-restarts preprocessed training if it stops
Does NOT interrupt original training
"""

import os
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

print("="*70)
print("CONTINUOUS MONITOR FOR PREPROCESSED TRAINING")
print("="*70)
print("Monitoring every 2 minutes")
print("Will auto-restart if training stops")
print("Won't interrupt original training")
print("="*70)
print()

# Find original training process (started earliest)
def get_original_training_pid():
    """Find the PID of original training (started earliest)"""
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                              capture_output=True, text=True)
        if 'python.exe' not in result.stdout:
            return None
        
        # Parse PIDs and start times
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        processes = []
        for line in lines:
            if 'python.exe' in line:
                parts = line.split('","')
                if len(parts) >= 2:
                    pid = parts[1].strip('"')
                    try:
                        processes.append(int(pid))
                    except:
                        pass
        
        # Return the oldest one (original training)
        if processes:
            return min(processes)  # Assuming oldest is original
        return None
    except:
        return None

original_pid = get_original_training_pid()
if original_pid:
    print(f"Original training PID: {original_pid} (will not interrupt)")
else:
    print("Original training PID: Not found (might have stopped)")

print()

# Check if preprocessed training script exists
preprocessed_script = 'restart_preprocessed_training.py'
if not os.path.exists(preprocessed_script):
    print(f"[ERROR] Script not found: {preprocessed_script}")
    sys.exit(1)

print(f"Monitoring script: {preprocessed_script}")
print()

check_count = 0
restart_count = 0

try:
    while True:
        check_count += 1
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Check Python processes
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                              capture_output=True, text=True)
        
        python_count = result.stdout.count('python.exe')
        
        # Check for preprocessed model file
        models_dir = Path('models')
        preprocessed_models = list(models_dir.glob('best_model_preprocessed_*.h5'))
        
        # Check if preprocessed training is running
        # Look for process that's NOT the original training
        is_running = False
        if python_count > 1:  # More than just original training
            # Check if preprocessed model was recently updated
            if preprocessed_models:
                latest_model = max(preprocessed_models, key=os.path.getmtime)
                mod_time = datetime.fromtimestamp(os.path.getmtime(latest_model))
                time_diff = (datetime.now() - mod_time).total_seconds()
                
                if time_diff < 300:  # Updated in last 5 minutes
                    is_running = True
        
        print(f"[{current_time}] Check #{check_count}: ", end="")
        
        if is_running:
            print("Preprocessed training is RUNNING")
            if preprocessed_models:
                latest = max(preprocessed_models, key=os.path.getmtime)
                mod_time = datetime.fromtimestamp(os.path.getmtime(latest))
                time_diff = (datetime.now() - mod_time).total_seconds() / 60
                print(f"  Latest model: {latest.name}")
                print(f"  Last updated: {time_diff:.1f} minutes ago")
        else:
            print("Preprocessed training is NOT RUNNING")
            print("  Restarting...")
            
            # Restart preprocessed training
            try:
                # Start in background
                if sys.platform == 'win32':
                    subprocess.Popen([sys.executable, preprocessed_script], 
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen([sys.executable, preprocessed_script],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                
                restart_count += 1
                print(f"  [OK] Restarted (restart #{restart_count})")
                print("  Waiting 30 seconds before next check...")
                time.sleep(30)  # Wait longer after restart
            except Exception as e:
                print(f"  [ERROR] Failed to restart: {e}")
        
        print()
        
        # Wait 2 minutes before next check
        time.sleep(120)  # 2 minutes
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped by user.")
    print(f"Total checks: {check_count}")
    print(f"Total restarts: {restart_count}")

