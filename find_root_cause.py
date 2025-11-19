"""
Find Root Cause - Why Model Always Predicts Same Thing
=======================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

print("="*70)
print("FINDING ROOT CAUSE - WHY 50% ACCURACY")
print("="*70)
print()

# Load model
model_path = 'models/best_model_20251110_193527.h5'
model = keras.models.load_model(model_path)

print("CRITICAL FINDING:")
print("-"*70)
print("All predictions are: 0.528 (always the same!)")
print("Model always predicts: RG (class 1)")
print("This means model is BROKEN or NOT TRAINABLE")
print()

# Check model structure
print("Checking model structure...")
print(f"Total layers: {len(model.layers)}")

# Check each layer
print("\nLayer details:")
for i, layer in enumerate(model.layers):
    print(f"  {i}. {layer.name}")
    print(f"     Trainable: {layer.trainable}")
    if hasattr(layer, 'output_shape'):
        print(f"     Output: {layer.output_shape}")
    if 'dense' in layer.name.lower() or 'output' in layer.name.lower():
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1] if len(layer.get_weights()) > 1 else None
            print(f"     Weights shape: {weights.shape}")
            print(f"     Weights mean: {weights.mean():.6f}")
            if biases is not None:
                print(f"     Bias: {biases}")
    print()

# Check if output layer is broken
print("Checking output layer...")
output_layer = model.layers[-1]
print(f"Output layer: {output_layer.name}")
print(f"Activation: {output_layer.activation}")

if hasattr(output_layer, 'get_weights'):
    weights = output_layer.get_weights()
    if len(weights) > 0:
        print(f"Weights shape: {weights[0].shape}")
        print(f"Weights: {weights[0]}")
        if len(weights) > 1:
            print(f"Bias: {weights[1]}")
            # Check if bias is causing the issue
            if len(weights[1].shape) == 1 and weights[1][0] > 0:
                print(f"\n[PROBLEM FOUND] Output bias is {weights[1][0]:.3f}")
                print("   High bias might cause model to always predict one class!")

print()
print("="*70)
print("ROOT CAUSE:")
print("="*70)
print("Model always predicts 0.528 (same value for all inputs)")
print("This means:")
print("  1. Model weights might be broken")
print("  2. Model might not be trainable")
print("  3. Output layer might have wrong bias")
print()
print("SOLUTION: Need to rebuild model or check if saved model is corrupted")

