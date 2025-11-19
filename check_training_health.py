"""
Check if Training is Working Correctly
=======================================
Verifies that accuracy is improving (not stuck at 50%)
"""

import os
import re
from pathlib import Path
from datetime import datetime

print("="*70)
print("CHECKING IF TRAINING IS WORKING CORRECTLY")
print("="*70)
print()

# Find latest log
log_dir = Path('training_logs')
log_files = list(log_dir.glob('phase1_restart_*.log'))
if not log_files:
    print("No Phase 1 restart log found. Training may not have started.")
    exit(1)

latest_log = max(log_files, key=os.path.getmtime)
print(f"Checking log: {latest_log.name}")
print(f"Last updated: {datetime.fromtimestamp(os.path.getmtime(latest_log))}")
print()

# Read log content
with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Check for accuracy values
accuracy_pattern = r'accuracy[:\s]+([\d.]+)'
val_accuracy_pattern = r'val_accuracy[:\s]+([\d.]+)'

accuracies = re.findall(accuracy_pattern, content, re.IGNORECASE)
val_accuracies = re.findall(val_accuracy_pattern, content, re.IGNORECASE)

print("="*70)
print("ACCURACY ANALYSIS")
print("="*70)

if accuracies or val_accuracies:
    print("\nTraining Accuracy Values Found:")
    if accuracies:
        # Convert to percentages and show last 10
        acc_values = [float(a) * 100 if float(a) < 1 else float(a) for a in accuracies[-10:]]
        for i, acc in enumerate(acc_values, 1):
            print(f"  Epoch {i}: {acc:.2f}%")
        print(f"\n  Latest: {acc_values[-1]:.2f}%")
        
        # Check if improving
        if len(acc_values) >= 2:
            if acc_values[-1] > acc_values[0]:
                print(f"  [OK] Accuracy is IMPROVING! ({acc_values[0]:.2f}% → {acc_values[-1]:.2f}%)")
            elif acc_values[-1] == acc_values[0] and acc_values[-1] == 50.0:
                print(f"  [WARNING] Accuracy stuck at 50% - Model not learning!")
            else:
                print(f"  [INFO] Accuracy: {acc_values[-1]:.2f}%")
    
    print("\nValidation Accuracy Values Found:")
    if val_accuracies:
        val_acc_values = [float(a) * 100 if float(a) < 1 else float(a) for a in val_accuracies[-10:]]
        for i, acc in enumerate(val_acc_values, 1):
            print(f"  Epoch {i}: {acc:.2f}%")
        print(f"\n  Latest: {val_acc_values[-1]:.2f}%")
        
        # Check if improving
        if len(val_acc_values) >= 2:
            if val_acc_values[-1] > val_acc_values[0]:
                print(f"  [OK] Validation accuracy is IMPROVING! ({val_acc_values[0]:.2f}% → {val_acc_values[-1]:.2f}%)")
            elif val_acc_values[-1] == 50.0:
                print(f"  [WARNING] Validation accuracy stuck at 50% - Model not learning!")
            elif val_acc_values[-1] > 50.0:
                print(f"  [OK] Validation accuracy is above 50% - Model is learning!")
            else:
                print(f"  [WARNING] Validation accuracy below 50% - Something is wrong!")
else:
    print("\nNo accuracy values found in log yet.")
    print("Training might still be starting or loading data...")
    print("\nLast 20 lines of log:")
    print("-"*70)
    lines = content.split('\n')
    for line in lines[-20:]:
        print(line)
    print("-"*70)

# Check for errors
error_keywords = ['error', 'Error', 'ERROR', 'exception', 'Exception', 'failed', 'Failed']
errors_found = [line for line in content.split('\n') if any(kw in line for kw in error_keywords)]

if errors_found:
    print("\n" + "="*70)
    print("ERRORS FOUND:")
    print("="*70)
    for error in errors_found[-5:]:  # Show last 5 errors
        print(f"  {error}")

# Check for epoch progress
epoch_pattern = r'Epoch (\d+)/(\d+)'
epochs = re.findall(epoch_pattern, content)

if epochs:
    print("\n" + "="*70)
    print("EPOCH PROGRESS")
    print("="*70)
    latest_epoch = epochs[-1]
    print(f"Current: Epoch {latest_epoch[0]}/{latest_epoch[1]}")
    print(f"Progress: {int(latest_epoch[0])/int(latest_epoch[1])*100:.1f}%")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if val_accuracies:
    latest_val_acc = float(val_accuracies[-1]) * 100 if float(val_accuracies[-1]) < 1 else float(val_accuracies[-1])
    
    if latest_val_acc > 50.0:
        print("[OK] Training is working correctly!")
        print(f"    Validation accuracy: {latest_val_acc:.2f}% (above 50%)")
        print("    Model is learning properly.")
    elif latest_val_acc == 50.0:
        print("[WARNING] Training might not be working correctly!")
        print(f"    Validation accuracy: {latest_val_acc:.2f}% (stuck at 50%)")
        print("    Model appears to be random guessing.")
        print("    Need to investigate further.")
    else:
        print("[WARNING] Training has issues!")
        print(f"    Validation accuracy: {latest_val_acc:.2f}% (below 50%)")
        print("    Something is wrong with the training setup.")
else:
    print("[INFO] Training just started - no accuracy values yet.")
    print("    Check back in a few minutes to see if accuracy improves.")

print("="*70)

