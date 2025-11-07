"""
Data augmentation module for glaucoma detection
"""

import numpy as np
import cv2
from typing import Tuple
import config


def apply_zoom(image: np.ndarray, zoom_factor: float = config.ZOOM_FACTOR) -> np.ndarray:
    """
    Apply subtle zoom to image.
    
    For fundus images, we use very subtle zoom to maintain anatomical correctness.
    Large zoom factors would distort the optic disc proportions.
    
    Args:
        image: Input image array
        zoom_factor: Zoom factor (positive for zoom in, negative for zoom out)
                   Typical: 0.035
        
    Returns:
        Zoomed image
    """
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h = int(h * (1 + zoom_factor))
    new_w = int(w * (1 + zoom_factor))
    
    # Resize image
    zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Crop to original size (center crop)
    start_h = (new_h - h) // 2
    start_w = (new_w - w) // 2
    
    if len(image.shape) == 3:
        cropped = zoomed[start_h:start_h + h, start_w:start_w + w, :]
    else:
        cropped = zoomed[start_h:start_h + h, start_w:start_w + w]
    
    return cropped


def apply_rotation(image: np.ndarray, angle: float = None, 
                  rotation_range: float = config.ROTATION_RANGE) -> np.ndarray:
    """
    Apply subtle rotation to image.
    
    For fundus images, only very small rotations are acceptable to maintain
    the orientation of anatomical structures.
    
    Args:
        image: Input image array
        angle: Rotation angle in degrees (if None, random within range)
        rotation_range: Maximum rotation range in degrees (default: 0.025)
        
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation angle
    if angle is None:
        angle = np.random.uniform(-rotation_range, rotation_range)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    
    return rotated


def apply_horizontal_flip(image: np.ndarray) -> np.ndarray:
    """
    Apply horizontal flip.
    
    Note: Generally NOT recommended for fundus images as it changes the
    orientation of the optic disc and blood vessels.
    
    Args:
        image: Input image array
        
    Returns:
        Horizontally flipped image
    """
    return cv2.flip(image, 1)


def apply_vertical_flip(image: np.ndarray) -> np.ndarray:
    """
    Apply vertical flip.
    
    Note: Generally NOT recommended for fundus images as it changes
    the anatomical orientation.
    
    Args:
        image: Input image array
        
    Returns:
        Vertically flipped image
    """
    return cv2.flip(image, 0)


def apply_brightness_adjustment(image: np.ndarray, factor: float = 0.1) -> np.ndarray:
    """
    Apply subtle brightness adjustment.
    
    Args:
        image: Input image array
        factor: Brightness adjustment factor (-1 to 1)
        
    Returns:
        Brightness-adjusted image
    """
    if len(image.shape) == 3:
        adjusted = image.astype(np.float32) + (factor * 255)
    else:
        adjusted = image.astype(np.float32) + (factor * 255)
    
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def apply_contrast_adjustment(image: np.ndarray, factor: float = 0.1) -> np.ndarray:
    """
    Apply subtle contrast adjustment.
    
    Args:
        image: Input image array
        factor: Contrast adjustment factor (-1 to 1)
        
    Returns:
        Contrast-adjusted image
    """
    if len(image.shape) == 3:
        adjusted = image.astype(np.float32) * (1 + factor)
    else:
        adjusted = image.astype(np.float32) * (1 + factor)
    
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def augment_image(image: np.ndarray, 
                 do_zoom: bool = True,
                 do_rotation: bool = True,
                 do_brightness: bool = False,
                 do_contrast: bool = False) -> np.ndarray:
    """
    Apply augmentation to an image based on configuration.
    
    Args:
        image: Input image array
        do_zoom: Whether to apply zoom
        do_rotation: Whether to apply rotation
        do_brightness: Whether to apply brightness adjustment
        do_contrast: Whether to apply contrast adjustment
        
    Returns:
        Augmented image
    """
    if not config.AUGMENTATION_ENABLED:
        return image
    
    augmented = image.copy()
    
    if do_zoom:
        augmented = apply_zoom(augmented)
    
    if do_rotation:
        augmented = apply_rotation(augmented)
    
    if do_brightness and np.random.random() > 0.5:
        factor = np.random.uniform(-0.1, 0.1)
        augmented = apply_brightness_adjustment(augmented, factor)
    
    if do_contrast and np.random.random() > 0.5:
        factor = np.random.uniform(-0.1, 0.1)
        augmented = apply_contrast_adjustment(augmented, factor)
    
    return augmented


def augment_dataset(X: np.ndarray, y: np.ndarray,
                   augmentation_prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment a dataset by applying augmentation to a subset of images.
    
    Args:
        X: Image array (or list of images)
        y: Label array
        augmentation_prob: Probability of augmenting each image
        
    Returns:
        Augmented X and y
    """
    if not config.AUGMENTATION_ENABLED:
        return X, y
    
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        image = X[i] if isinstance(X, (list, np.ndarray)) else X[i]
        label = y[i]
        
        # Original image
        augmented_X.append(image)
        augmented_y.append(label)
        
        # Augmented version with probability
        if np.random.random() < augmentation_prob:
            augmented_img = augment_image(image)
            augmented_X.append(augmented_img)
            augmented_y.append(label)
    
    return np.array(augmented_X), np.array(augmented_y)


if __name__ == "__main__":
    print("Data augmentation module loaded successfully!")
    print(f"Augmentation enabled: {config.AUGMENTATION_ENABLED}")
    print(f"Zoom factor: {config.ZOOM_FACTOR}")
    print(f"Rotation range: {config.ROTATION_RANGE}")
