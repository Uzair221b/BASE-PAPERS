"""
Watch Training Progress in Real-Time
=====================================
Shows training progress as it happens
"""

import os
import time
from pathlib import Path
from datetime import datetime

def watch_training_live():
    """Watch the latest training log in real-time"""
    
    log_dir = Path('training_logs')
    if not log_dir.exists():
        print("No training logs found. Training may not have started yet.")
        return
    
    print("="*70)
    print("WATCHING TRAINING PROGRESS - REAL-TIME")
    print("="*70)
    print("Press Ctrl+C to stop watching")
    print("="*70)
    print()
    
    last_size = 0
    last_file = None
    
    try:
        while True:
            # Find latest log file
            log_files = list(log_dir.glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=os.path.getmtime)
                
                # If file changed, show new content
                if latest_log != last_file or os.path.getsize(latest_log) != last_size:
                    if latest_log != last_file:
                        print(f"\n{'='*70}")
                        print(f"Watching: {latest_log.name}")
                        print(f"{'='*70}\n")
                        last_file = latest_log
                    
                    # Read new content
                    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        if new_content:
                            print(new_content, end='', flush=True)
                        last_size = f.tell()
                
                time.sleep(1)  # Check every second
            else:
                print("Waiting for training to start...")
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("\n\nStopped watching. Training continues in background.")

if __name__ == "__main__":
    watch_training_live()

