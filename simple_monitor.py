"""
SIMPLE AUTO MONITOR (No extra dependencies needed)
===================================================

Checks every 10 minutes if training is running.
If not, restarts it automatically.

Just run: python simple_monitor.py
"""

import subprocess
import time
import os
import sys
from datetime import datetime
from pathlib import Path

TRAINING_SCRIPT = 'simple_resume_training.py'
CHECK_INTERVAL = 600  # 10 minutes
MAX_RESTARTS = 20

def is_training_running():
    """Check if training script is running"""
    try:
        # Check for Python processes
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True,
            text=True,
            shell=True
        )
        
        # Check if our script is in the process list
        if TRAINING_SCRIPT in result.stdout:
            return True
        
        # Also check if there's a training log being updated
        log_dir = Path('training_logs')
        if log_dir.exists():
            log_files = list(log_dir.glob('*.log'))
            if log_files:
                latest = max(log_files, key=os.path.getmtime)
                age = time.time() - os.path.getmtime(latest)
                if age < 600:  # Updated in last 10 minutes
                    return True
        
        return False
    except:
        return False

def start_training():
    """Start training script"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
        process = subprocess.Popen(
            [sys.executable, TRAINING_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(5)  # Wait a bit to see if it starts
        if process.poll() is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Training started (PID: {process.pid})")
            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Training failed to start")
            return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error starting training: {e}")
        return False

def main():
    print("="*70)
    print("SIMPLE AUTO TRAINING MONITOR")
    print("="*70)
    print(f"Monitoring: {TRAINING_SCRIPT}")
    print(f"Check interval: {CHECK_INTERVAL/60:.0f} minutes")
    print("="*70)
    print("\nPress Ctrl+C to stop\n")
    
    restart_count = 0
    check_count = 0
    
    # Start training initially
    if not start_training():
        print("Failed to start training. Check if script exists.")
        return
    
    restart_count += 1
    
    try:
        while restart_count < MAX_RESTARTS:
            check_count += 1
            time.sleep(CHECK_INTERVAL)
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Check #{check_count}")
            
            if not is_training_running():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Training not running!")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Restarting... (attempt {restart_count + 1}/{MAX_RESTARTS})")
                time.sleep(60)  # Wait 1 minute before restart
                if start_training():
                    restart_count += 1
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Training is running")
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Maximum restarts reached. Stopping monitor.")
        
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitor stopped by user")

if __name__ == "__main__":
    main()


