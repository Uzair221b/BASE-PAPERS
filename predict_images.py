"""
Predict Glaucoma on Random Images
==================================
After achieving 99%+ accuracy, use this script to predict on your test images.
You provide images (mixed positive/negative), model predicts each one.
"""

import numpy as np
import cv2
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import argparse
from datetime import datetime

print("="*70)
print("GLAUCOMA DETECTION - IMAGE PREDICTION")
print("="*70)
print()

def load_image(image_path):
    """Load and preprocess a single image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def predict_images(model_path, images_folder, output_file=None):
    """
    Predict glaucoma on images in a folder.
    
    Args:
        model_path: Path to trained model (.h5 file)
        images_folder: Folder containing images to predict
        output_file: Optional file to save results
    """
    # Load model
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    model = keras.models.load_model(model_path)
    print("[OK] Model loaded successfully")
    print()
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_folder).glob(f'*{ext}'))
        image_files.extend(Path(images_folder).glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"[ERROR] No images found in: {images_folder}")
        return
    
    print(f"Found {len(image_files)} images")
    print()
    print("="*70)
    print("PREDICTIONS")
    print("="*70)
    print()
    
    # Predict each image
    predictions = []
    results = []
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        # Load and preprocess
        img = load_image(img_path)
        if img is None:
            print(f"  [SKIP] Could not load image")
            continue
        
        # Predict
        img_batch = np.expand_dims(img, axis=0)
        prediction = model.predict(img_batch, verbose=0)[0][0]
        
        # Interpret result
        is_glaucoma = prediction > 0.5
        confidence = prediction * 100 if is_glaucoma else (1 - prediction) * 100
        label = "Glaucoma Positive" if is_glaucoma else "Glaucoma Negative"
        
        # Store result
        result = {
            'image': img_path.name,
            'path': str(img_path),
            'prediction': label,
            'confidence': confidence,
            'raw_score': float(prediction)
        }
        results.append(result)
        predictions.append((img_path.name, label, confidence))
        
        # Print result
        print(f"  Prediction: {label}")
        print(f"  Confidence: {confidence:.2f}%")
        print()
    
    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images processed: {len(results)}")
    print(f"Glaucoma Positive: {sum(1 for r in results if 'Positive' in r['prediction'])}")
    print(f"Glaucoma Negative: {sum(1 for r in results if 'Negative' in r['prediction'])}")
    print()
    
    # Save results to file if requested
    if output_file:
        print(f"Saving results to: {output_file}")
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GLAUCOMA DETECTION PREDICTIONS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Images Folder: {images_folder}\n")
            f.write("="*70 + "\n\n")
            
            for result in results:
                f.write(f"Image: {result['image']}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Confidence: {result['confidence']:.2f}%\n")
                f.write(f"Raw Score: {result['raw_score']:.4f}\n")
                f.write("-"*70 + "\n")
        
        print("[OK] Results saved")
        print()
    
    # Print all predictions
    print("="*70)
    print("ALL PREDICTIONS")
    print("="*70)
    for img_name, label, conf in predictions:
        print(f"{img_name:30s} → {label:20s} ({conf:.2f}%)")
    
    print()
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review predictions above")
    print("2. Verify predictions against actual labels (if you know them)")
    print("3. If you have ground truth labels, we can calculate accuracy")
    print("="*70)
    
    return results

def calculate_accuracy(predictions_file, ground_truth_file):
    """
    Calculate accuracy if you provide ground truth labels.
    
    Format for ground_truth_file (CSV):
    image_name,label
    image1.jpg,1
    image2.jpg,0
    ...
    (1 = Glaucoma Positive, 0 = Glaucoma Negative)
    """
    import pandas as pd
    
    # Load ground truth
    gt = pd.read_csv(ground_truth_file)
    gt_dict = dict(zip(gt['image_name'], gt['label']))
    
    # Load predictions
    results = []  # Would load from predictions_file
    
    # Calculate accuracy
    correct = 0
    total = 0
    
    for result in results:
        img_name = result['image']
        if img_name in gt_dict:
            predicted = 1 if 'Positive' in result['prediction'] else 0
            actual = gt_dict[img_name]
            if predicted == actual:
                correct += 1
            total += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict glaucoma on images')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--images', type=str, required=True,
                       help='Folder containing images to predict')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: Save predictions to file')
    
    args = parser.parse_args()
    
    # Predict
    results = predict_images(args.model, args.images, args.output)
    
    print()
    print("To calculate accuracy with ground truth labels:")
    print("1. Create a CSV file with: image_name,label")
    print("2. Run: python calculate_accuracy.py --predictions results.txt --ground_truth labels.csv")

