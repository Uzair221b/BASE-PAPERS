"""
Restart Training - Fix Phase 2 Issue
=====================================
"""

import subprocess
import os
import sys

print("="*70)
print("RESTARTING TRAINING - FIXING PHASE 2 ISSUE")
print("="*70)
print()

# Step 1: Stop stuck Python processes
print("Step 1: Stopping stuck Python processes...")
try:
    result = subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                          capture_output=True, text=True)
    if 'successfully' in result.stdout.lower() or result.returncode == 0:
        print("[OK] Stopped Python processes")
    else:
        print("[INFO] No Python processes to stop (or already stopped)")
except Exception as e:
    print(f"[INFO] Could not stop processes: {e}")

print()

# Step 2: Check for Phase 1 model
print("Step 2: Checking for Phase 1 checkpoint...")
phase1_model = "models/best_model_phase1.h5"
if os.path.exists(phase1_model):
    print(f"[OK] Found Phase 1 model: {phase1_model}")
    print("   We can continue from Phase 2")
else:
    print("[WARNING] Phase 1 model not found")
    print("   Will need to restart from beginning")

print()

# Step 3: Important note about 50% accuracy
print("="*70)
print("IMPORTANT: 50% ACCURACY ISSUE")
print("="*70)
print()
print("Phase 1 showed 50% validation accuracy - this is RANDOM GUESSING!")
print()
print("This means the model is NOT learning. Possible causes:")
print("  1. Labels might be wrong")
print("  2. Model architecture issue")
print("  3. Data not loading correctly")
print("  4. Learning rate too high/low")
print()
print("Before restarting, we should investigate this issue.")
print()

# Step 4: Ask what to do
print("="*70)
print("OPTIONS:")
print("="*70)
print()
print("Option 1: Restart training from Phase 1 checkpoint")
print("  - Continue Phase 2 from saved model")
print("  - But 50% accuracy issue will persist")
print()
print("Option 2: Investigate 50% accuracy issue first")
print("  - Check labels")
print("  - Check data loading")
print("  - Fix the problem")
print("  - Then restart")
print()
print("Option 3: Restart from beginning")
print("  - Start fresh")
print("  - But need to fix 50% issue first")
print()

print("="*70)
print("RECOMMENDATION: Investigate 50% accuracy issue first!")
print("="*70)

