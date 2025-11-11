"""
Auto-Recovery Training Starter
Automatically restarts training if it fails (up to 3 attempts)
"""

import subprocess
import sys
import time
from datetime import datetime
import os

MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds

def log_message(message):
    """Print and log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    with open("training_starter_log.txt", "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

def run_training():
    """Run the training script"""
    log_message("Starting training process...")
    
    try:
        # Run the training script
        result = subprocess.run(
            [sys.executable, "train_robust.py"],
            capture_output=False,
            text=True
        )
        
        return result.returncode
    
    except Exception as e:
        log_message(f"Error running training: {e}")
        return 1

def main():
    """Main function with auto-recovery"""
    print("="*80)
    print(" ROBUST TRAINING STARTER")
    print(" Automatic Error Recovery Enabled")
    print("="*80)
    print()
    
    log_message("="*80)
    log_message(" TRAINING SESSION STARTED")
    log_message("="*80)
    
    attempt = 1
    
    while attempt <= MAX_RETRIES:
        log_message(f"\nAttempt {attempt}/{MAX_RETRIES}")
        log_message(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        return_code = run_training()
        elapsed_time = (time.time() - start_time) / 3600
        
        if return_code == 0:
            log_message("="*80)
            log_message(" TRAINING COMPLETED SUCCESSFULLY!")
            log_message("="*80)
            log_message(f"Total time: {elapsed_time:.2f} hours")
            print("\n" + "="*80)
            print(" SUCCESS! Training completed successfully!")
            print("="*80)
            print(f"\nCheck results in the results/ folder")
            print(f"Check models in the models/ folder")
            print(f"Full log: training_starter_log.txt")
            return 0
        
        elif return_code == 130:  # Keyboard interrupt
            log_message("Training interrupted by user (Ctrl+C)")
            print("\nTraining interrupted by user.")
            return 130
        
        else:
            log_message(f"Training failed with exit code {return_code}")
            log_message(f"Ran for {elapsed_time:.2f} hours before failure")
            
            if attempt < MAX_RETRIES:
                log_message(f"Waiting {RETRY_DELAY} seconds before retry...")
                print(f"\nTraining failed. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                attempt += 1
            else:
                log_message("="*80)
                log_message(" TRAINING FAILED AFTER ALL RETRIES")
                log_message("="*80)
                print("\n" + "="*80)
                print(" TRAINING FAILED")
                print("="*80)
                print(f"\nFailed after {MAX_RETRIES} attempts.")
                print("Check the logs for details:")
                print("  - training_starter_log.txt")
                print("  - logs/training_log_*.txt")
                print("  - results/error_report_*.json")
                return 1
    
    return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining startup interrupted by user.")
        log_message("\nTraining startup interrupted by user (Ctrl+C)")
        sys.exit(130)

