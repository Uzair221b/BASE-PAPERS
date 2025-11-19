"""
Monitor Original Training and Auto-Stop at 99%
==============================================
Continuously monitors accuracy and stops training when 99% is reached
"""

import numpy as np
import os
import cv2
import time
import subprocess
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

print("="*70)
print("MONITORING ORIGINAL TRAINING - AUTO-STOP AT 99%")
print("="*70)
print("Will check accuracy every 5 minutes")
print("Will stop training automatically when 99%+ is reached")
print("="*70)
print()

# Find original training PID
ORIGINAL_PID = 8780  # From previous check

def check_accuracy():
    """Check current accuracy without interrupting training"""
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) 
                   if f.startswith('best_cnn_model_') and f.endswith('.h5')]
    
    if not model_files:
        return None, None
    
    latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
    model_path = os.path.join(models_dir, latest_model)
    
    try:
        model = keras.models.load_model(model_path)
        
        # Load test sample
        original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'
        
        def load_sample(data_dir, class_name, label, max_samples=500):
            images = []
            labels = []
            class_path = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_path):
                return images, labels
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_samples]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
                        labels.append(label)
                except:
                    continue
            
            return images, labels
        
        rg_images, rg_labels = load_sample(original_dir, 'RG', 1, max_samples=500)
        nrg_images, nrg_labels = load_sample(original_dir, 'NRG', 0, max_samples=500)
        
        X_test = np.array(rg_images + nrg_images)
        y_test = np.array(rg_labels + nrg_labels)
        
        predictions = model.predict(X_test, verbose=0)
        pred_classes = (predictions > 0.5).astype(int).flatten()
        accuracy = np.mean(pred_classes == y_test)
        
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        
        # Clear model from memory
        del model
        keras.backend.clear_session()
        
        return accuracy, mod_time
        
    except Exception as e:
        print(f"[ERROR] Could not check accuracy: {e}")
        return None, None

check_count = 0
last_accuracy = 0

print(f"Monitoring training process (PID: {ORIGINAL_PID})")
print("Checking accuracy every 5 minutes...")
print("Will stop automatically when 99%+ is reached")
print()
print("="*70)
print()

try:
    while True:
        check_count += 1
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Check if process is still running
        try:
            process = subprocess.run(['tasklist', '/FI', f'PID eq {ORIGINAL_PID}', '/FO', 'CSV'], 
                                   capture_output=True, text=True)
            if str(ORIGINAL_PID) not in process.stdout:
                print(f"[{current_time}] Training process stopped (PID {ORIGINAL_PID} not found)")
                print("Training has completed or stopped.")
                break
        except:
            pass
        
        # Check accuracy
        print(f"[{current_time}] Check #{check_count}: Checking accuracy...")
        accuracy, mod_time = check_accuracy()
        
        if accuracy is not None:
            print(f"  Current accuracy: {accuracy*100:.2f}%")
            print(f"  Model last updated: {mod_time.strftime('%H:%M:%S')}")
            
            if accuracy >= 0.99:
                print()
                print("="*70)
                print("[SUCCESS] 99%+ ACCURACY REACHED!")
                print("="*70)
                print(f"Accuracy: {accuracy*100:.2f}%")
                print("Stopping training process...")
                
                # Stop the training process
                try:
                    subprocess.run(['taskkill', '/F', '/PID', str(ORIGINAL_PID)], 
                                 capture_output=True)
                    print(f"[OK] Training stopped (PID: {ORIGINAL_PID})")
                    print()
                    print("Training goal achieved! Model saved.")
                    break
                except Exception as e:
                    print(f"[WARNING] Could not stop process: {e}")
                    print("You may need to stop it manually.")
                    break
            else:
                improvement = accuracy - last_accuracy
                if improvement > 0:
                    print(f"  Improvement: +{improvement*100:.2f}%")
                elif improvement < 0:
                    print(f"  Change: {improvement*100:.2f}% (validation can fluctuate)")
                
                last_accuracy = accuracy
                
                if accuracy >= 0.95:
                    print(f"  [EXCELLENT] Very close to 99%! ({accuracy*100:.2f}%)")
                elif accuracy >= 0.90:
                    print(f"  [GOOD] Making great progress! ({accuracy*100:.2f}%)")
                elif accuracy >= 0.85:
                    print(f"  [OK] Steady improvement ({accuracy*100:.2f}%)")
        else:
            print("  [WARNING] Could not check accuracy")
        
        print()
        print("Next check in 5 minutes...")
        print("-"*70)
        print()
        
        # Wait 5 minutes
        time.sleep(300)  # 5 minutes
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped by user.")
    print(f"Total checks: {check_count}")
    print(f"Last accuracy: {last_accuracy*100:.2f}%")

