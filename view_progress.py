"""
View Training Progress in Real-Time
====================================

Shows the latest training progress from log files
"""

import os
from pathlib import Path
import time
from datetime import datetime

def view_latest_log():
    """View the latest training log"""
    log_dir = Path('training_logs')
    
    if not log_dir.exists():
        print("No training logs found. Training may not have started yet.")
        return
    
    # Find latest training log
    log_files = list(log_dir.glob('training_*.log'))
    if not log_files:
        print("No training log files found.")
        return
    
    latest_log = max(log_files, key=os.path.getmtime)
    
    print("="*70)
    print("VIEWING TRAINING PROGRESS")
    print("="*70)
    print(f"Log file: {latest_log}")
    print(f"Last updated: {datetime.fromtimestamp(os.path.getmtime(latest_log)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\nLatest entries:\n")
    
    # Read last 50 lines
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line.rstrip())
    except Exception as e:
        print(f"Error reading log: {e}")

def watch_progress():
    """Watch progress in real-time (updates every 5 seconds)"""
    print("="*70)
    print("WATCHING TRAINING PROGRESS (Press Ctrl+C to stop)")
    print("="*70)
    print("Updates every 5 seconds...\n")
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            view_latest_log()
            print("\n" + "="*70)
            print("Press Ctrl+C to stop watching")
            print("="*70)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nStopped watching. Training continues in background.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        watch_progress()
    else:
        view_latest_log()
        print("\nTip: Run 'python view_progress.py --watch' to watch in real-time")


