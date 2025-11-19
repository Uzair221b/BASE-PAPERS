"""
Training with Enhanced Real-Time Progress Bars
==============================================

Shows detailed progress bars and real-time updates
"""

import sys
import os

# Add tqdm for better progress bars if available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for better progress bars: pip install tqdm")

# Import the training script
print("="*70)
print("STARTING TRAINING WITH REAL-TIME PROGRESS")
print("="*70)
print("\nYou will see:")
print("  ✅ Progress bars for each epoch")
print("  ✅ Real-time accuracy and loss")
print("  ✅ Batch-by-batch progress")
print("  ✅ Estimated time remaining")
print("\n" + "="*70 + "\n")

# Run the training script
os.system(f"{sys.executable} simple_resume_training.py")


