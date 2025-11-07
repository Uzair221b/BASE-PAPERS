"""
Data loading and image scaling module
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
import config


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in RGB format
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def scale_image(image: np.ndarray, target_size: Tuple[int, int] = config.IMAGE_SIZE) -> np.ndarray:
    """
    Scale/Resize image to target dimensions.
    
    This is the first step in the preprocessing pipeline to standardize
    all images to the same dimensions required for deep learning models.
    
    Args:
        image: Input image as numpy array
        target_size: Target dimensions (height, width)
        
    Returns:
        Resized image
    """
    height, width = target_size
    
    # Use bilinear interpolation for resizing
    if config.INTERPOLATION_METHOD == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif config.INTERPOLATION_METHOD == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR
    
    resized_image = cv2.resize(image, (width, height), interpolation=interpolation)
    return resized_image


def load_and_scale(image_path: Union[str, Path], target_size: Tuple[int, int] = config.IMAGE_SIZE) -> np.ndarray:
    """
    Load and scale an image in one operation.
    
    Args:
        image_path: Path to the image file
        target_size: Target dimensions (height, width)
        
    Returns:
        Loaded and scaled image
    """
    image = load_image(image_path)
    scaled_image = scale_image(image, target_size)
    return scaled_image


def load_dataset(image_dir: Union[str, Path], file_extensions: List[str] = ['.jpg', '.png', '.jpeg']) -> List[Path]:
    """
    Load all image files from a directory.
    
    Args:
        image_dir: Directory containing images
        file_extensions: List of valid file extensions
        
    Returns:
        List of image file paths (without duplicates)
    """
    image_dir = Path(image_dir)
    image_files = []
    
    for ext in file_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    # Remove duplicates by converting to set and back to list
    unique_files = list(set(image_files))
    
    return sorted(unique_files)


def process_image_path(image_path: Union[str, Path], scale: bool = True) -> np.ndarray:
    """
    Process a single image path through the data loading step.
    
    Args:
        image_path: Path to the image file
        scale: Whether to scale the image
        
    Returns:
        Processed image array
    """
    image = load_image(image_path)
    
    if scale:
        image = scale_image(image)
    
    return image


if __name__ == "__main__":
    # Test the data loading functions
    print("Testing data loading module...")
    print(f"Target image size: {config.IMAGE_SIZE}")
    print("Data loading module loaded successfully!")
