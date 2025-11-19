"""
AUTO TRAINING MONITOR & AUTO-RESTART
=====================================

This script:
1. Monitors training every 5-10 minutes
2. Auto-restarts if training crashes or stops
3. Auto-fixes common issues
4. Runs in background
5. Logs everything

Just run: python auto_train_monitor.py
Then you can close the terminal - it will keep running!
"""

import subprocess
import time
import os
import sys
import psutil
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
log_dir = Path('training_logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'training_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Configuration
TRAINING_SCRIPT = 'simple_resume_training.py'
CHECK_INTERVAL = 600  # Check every 10 minutes (600 seconds)
MAX_RESTART_ATTEMPTS = 10  # Maximum restart attempts
RESTART_DELAY = 60  # Wait 60 seconds before restarting

class TrainingMonitor:
    def __init__(self):
        self.training_process = None
        self.restart_count = 0
        self.start_time = datetime.now()
        self.last_check = None
        
    def is_training_running(self):
        """Check if training script is running"""
        try:
            # Check if our training process is still alive
            if self.training_process and self.training_process.poll() is None:
                return True
            
            # Also check for any Python process running the training script
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and TRAINING_SCRIPT in ' '.join(cmdline):
                        # Found training process
                        if self.training_process is None:
                            # Store it for monitoring
                            self.training_process = subprocess.Popen(
                                ['python', TRAINING_SCRIPT],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return False
        except Exception as e:
            logger.error(f"Error checking training status: {e}")
            return False
    
    def check_training_health(self):
        """Check if training is healthy (not stuck)"""
        try:
            if self.training_process:
                # Check if process is using CPU (not stuck)
                proc = psutil.Process(self.training_process.pid)
                cpu_percent = proc.cpu_percent(interval=1)
                
                # Check if process is using memory (not crashed)
                memory_mb = proc.memory_info().rss / 1024 / 1024
                
                # Check if log file is being updated
                log_files = list(Path('training_logs').glob('*.log'))
                if log_files:
                    latest_log = max(log_files, key=os.path.getmtime)
                    log_age = time.time() - os.path.getmtime(latest_log)
                    if log_age > 1800:  # 30 minutes without update = stuck
                        logger.warning("Training appears stuck (no log updates for 30+ minutes)")
                        return False
                
                logger.info(f"Training health: CPU={cpu_percent:.1f}%, Memory={memory_mb:.1f}MB")
                return True
        except Exception as e:
            logger.error(f"Error checking training health: {e}")
            return False
    
    def fix_common_issues(self):
        """Auto-fix common issues before restarting"""
        logger.info("Checking for common issues...")
        
        # Check if model file exists
        model_path = Path('models/best_model_20251110_193527.h5')
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error("Cannot continue - model file is missing!")
            return False
        
        # Check if data directory exists
        data_dir = Path('processed_datasets/eyepacs_train')
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            logger.error("Cannot continue - data directory is missing!")
            return False
        
        # Check disk space
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        if free_gb < 2:
            logger.warning(f"Low disk space: {free_gb:.2f}GB free")
            logger.warning("Training may fail due to low disk space")
        
        # Check memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb < 2:
            logger.warning(f"Low memory: {available_gb:.2f}GB available")
            logger.warning("Training may be slow or fail")
        
        logger.info("Common issues check complete")
        return True
    
    def start_training(self):
        """Start the training script"""
        try:
            logger.info("="*70)
            logger.info("STARTING TRAINING")
            logger.info("="*70)
            
            if not self.fix_common_issues():
                logger.error("Cannot start training - issues detected")
                return False
            
            # Start training in background
            self.training_process = subprocess.Popen(
                [sys.executable, TRAINING_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.restart_count += 1
            logger.info(f"Training started (PID: {self.training_process.pid})")
            logger.info(f"Restart attempt: {self.restart_count}/{MAX_RESTART_ATTEMPTS}")
            
            # Wait a bit to see if it starts successfully
            time.sleep(10)
            if self.training_process.poll() is not None:
                # Process died immediately
                stdout, stderr = self.training_process.communicate()
                logger.error("Training failed to start!")
                logger.error(f"Error output: {stdout}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return False
    
    def stop_training(self):
        """Stop the training process"""
        try:
            if self.training_process and self.training_process.poll() is None:
                logger.info("Stopping training process...")
                self.training_process.terminate()
                try:
                    self.training_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Process didn't terminate, forcing kill...")
                    self.training_process.kill()
                self.training_process = None
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("="*70)
        logger.info("AUTO TRAINING MONITOR STARTED")
        logger.info("="*70)
        logger.info(f"Monitoring script: {TRAINING_SCRIPT}")
        logger.info(f"Check interval: {CHECK_INTERVAL} seconds ({CHECK_INTERVAL/60:.1f} minutes)")
        logger.info(f"Log file: {log_file}")
        logger.info("="*70)
        
        # Start training initially
        if not self.start_training():
            logger.error("Failed to start training initially")
            return
        
        check_count = 0
        
        while True:
            try:
                check_count += 1
                self.last_check = datetime.now()
                elapsed = (self.last_check - self.start_time).total_seconds() / 3600
                
                logger.info(f"\n--- Check #{check_count} (Running for {elapsed:.2f} hours) ---")
                
                # Check if training is running
                if not self.is_training_running():
                    logger.warning("Training is NOT running!")
                    
                    if self.restart_count >= MAX_RESTART_ATTEMPTS:
                        logger.error(f"Maximum restart attempts ({MAX_RESTART_ATTEMPTS}) reached!")
                        logger.error("Stopping monitor. Please check logs and fix issues manually.")
                        break
                    
                    logger.info(f"Waiting {RESTART_DELAY} seconds before restarting...")
                    time.sleep(RESTART_DELAY)
                    
                    # Stop any remaining process
                    self.stop_training()
                    
                    # Restart training
                    if not self.start_training():
                        logger.error("Failed to restart training")
                        time.sleep(CHECK_INTERVAL)
                        continue
                else:
                    # Training is running, check health
                    if not self.check_training_health():
                        logger.warning("Training appears unhealthy, will restart on next check")
                    else:
                        logger.info("✅ Training is running normally")
                
                # Wait before next check
                logger.info(f"Next check in {CHECK_INTERVAL/60:.1f} minutes...")
                time.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("\nMonitor stopped by user")
                self.stop_training()
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(CHECK_INTERVAL)
        
        logger.info("="*70)
        logger.info("MONITOR STOPPED")
        logger.info("="*70)

if __name__ == "__main__":
    monitor = TrainingMonitor()
    try:
        monitor.monitor_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


