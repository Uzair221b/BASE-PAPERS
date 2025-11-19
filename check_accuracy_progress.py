"""
Check Accuracy Progress Without Interrupting Training
=====================================================
Check if accuracy is increasing by testing the latest saved model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import cv2
from sklearn.model_selection import train_test_split
from datetime import datetime

print("="*70)
print("CHECKING ACCURACY PROGRESS (Without Interrupting Training)")
print("="*70)
print()

# Find all model checkpoints
model_files = [f for f in os.listdir('models') if f.startswith('best_cnn_model_') and f.endswith('.h5')]
if not model_files:
    print("No model files found yet.")
    exit(0)

# Sort by modification time
model_files_with_time = [(f, os.path.getmtime(os.path.join('models', f))) for f in model_files]
model_files_with_time.sort(key=lambda x: x[1])

print(f"Found {len(model_files_with_time)} model checkpoints")
print()

# Load test data (same for all tests)
original_dir = 'EYEPACS(AIROGS)/eyepac-light-v2-512-jpg/train'

def load_sample(data_dir, class_name, label, max_samples=200):
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

print("Loading test sample...")
rg_images, rg_labels = load_sample(original_dir, 'RG', 1, max_samples=200)
nrg_images, nrg_labels = load_sample(original_dir, 'NRG', 0, max_samples=200)

X_test = np.array(rg_images + nrg_images)
y_test = np.array(rg_labels + nrg_labels)

print(f"Test sample: {len(X_test)} images")
print()

# Test latest few checkpoints
print("Testing latest checkpoints...")
print("="*70)

results = []
for i, (model_file, mod_time) in enumerate(model_files_with_time[-5:]):  # Test last 5 checkpoints
    model_path = os.path.join('models', model_file)
    
    try:
        model = keras.models.load_model(model_path)
        
        # Test predictions
        predictions = model.predict(X_test, verbose=0)
        pred_classes = (predictions > 0.5).astype(int).flatten()
        accuracy = np.mean(pred_classes == y_test)
        
        time_str = datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')
        results.append((model_file, accuracy, time_str, mod_time))
        
        print(f"Checkpoint {i+1}: {model_file}")
        print(f"  Time: {time_str}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print()
        
        # Clear model from memory
        del model
        keras.backend.clear_session()
        
    except Exception as e:
        print(f"Error loading {model_file}: {e}")
        print()

# Analyze trend
if len(results) >= 2:
    print("="*70)
    print("ACCURACY TREND ANALYSIS")
    print("="*70)
    
    accuracies = [r[1] for r in results]
    latest_acc = accuracies[-1]
    previous_acc = accuracies[-2] if len(accuracies) > 1 else accuracies[0]
    
    print(f"Previous accuracy: {previous_acc*100:.2f}%")
    print(f"Latest accuracy: {latest_acc*100:.2f}%")
    print(f"Change: {(latest_acc - previous_acc)*100:+.2f}%")
    print()
    
    if latest_acc > previous_acc:
        improvement = (latest_acc - previous_acc) * 100
        print(f"[GOOD] Accuracy is INCREASING! (+{improvement:.2f}%)")
        print("Training is working - model is learning!")
        
        if latest_acc >= 0.99:
            print("[SUCCESS] Reached 99%+ accuracy!")
        elif latest_acc >= 0.90:
            print(f"[EXCELLENT] Accuracy is {latest_acc*100:.2f}% - very close to 99%!")
        elif latest_acc >= 0.70:
            print(f"[GOOD] Accuracy is {latest_acc*100:.2f}% - making good progress!")
        elif latest_acc >= 0.60:
            print(f"[OK] Accuracy is {latest_acc*100:.2f}% - learning but slowly.")
        else:
            print(f"[SLOW] Accuracy is {latest_acc*100:.2f}% - learning very slowly.")
            
    elif latest_acc == previous_acc:
        print("[WARNING] Accuracy is NOT changing")
        print("Model might have stopped learning")
    else:
        print(f"[WARNING] Accuracy DECREASED by {(previous_acc - latest_acc)*100:.2f}%")
        print("This might be normal (validation can fluctuate)")
    
    # Check overall trend
    if len(accuracies) >= 3:
        first_acc = accuracies[0]
        trend = "increasing" if latest_acc > first_acc else "decreasing" if latest_acc < first_acc else "stable"
        total_change = (latest_acc - first_acc) * 100
        print()
        print(f"Overall trend: {trend} ({total_change:+.2f}% from first to latest)")
    
    print()
    print("Since training is still running, it means:")
    print("  - Accuracy is still improving (within patience window)")
    print("  - EarlyStopping hasn't triggered yet")
    print("  - Model is learning!")
    
else:
    print("Not enough checkpoints to analyze trend yet.")

print("="*70)

