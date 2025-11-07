"""
Cropping module to center optic disc region
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import config


def center_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop an image to the specified dimensions.
    
    Args:
        image: Input image array
        crop_size: Desired output size (height, width)
        
    Returns:
        Center-cropped image
    """
    h, w = image.shape[:2]
    target_h, target_w = crop_size
    
    # Calculate crop coordinates (center crop)
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    # Perform center crop
    if len(image.shape) == 3:  # Color image
        cropped = image[start_h:start_h + target_h, start_w:start_w + target_w, :]
    else:  # Grayscale image
        cropped = image[start_h:start_h + target_h, start_w:start_w + target_w]
    
    return cropped


def adaptive_crop(image: np.ndarray, center: Tuple[int, int], size: int) -> np.ndarray:
    """
    Crop image around a specific center point.
    
    Args:
        image: Input image array
        center: Center point coordinates (x, y)
        size: Size of the square crop
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    cx, cy = center
    
    # Calculate crop boundaries
    half_size = size // 2
    y_min = max(0, cy - half_size)
    y_max = min(h, cy + half_size)
    x_min = max(0, cx - half_size)
    x_max = min(w, cx + half_size)
    
    # Perform crop
    if len(image.shape) == 3:
        cropped = image[y_min:y_max, x_min:x_max, :]
    else:
        cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped


def detect_optic_disc(image: np.ndarray, method: str = 'template') -> Optional[Tuple[int, int]]:
    """
    Detect optic disc location in fundus image.
    
    Args:
        image: Fundus image (preferably RGB or grayscale)
        method: Detection method ('template' or 'color')
        
    Returns:
        Center coordinates of optic disc (x, y) or None if not detected
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    if method == 'template':
        # Simple template-based detection using circular Hough transform
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Apply HoughCircles to detect optic disc (circular region)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(gray.shape[0]/2),
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=int(gray.shape[0]/3)
        )
        
        if circles is not None and len(circles) > 0:
            # Return center of first detected circle
            circle = circles[0][0]
            return (int(circle[0]), int(circle[1]))
    
    elif method == 'color':
        # Detect optic disc based on brightness (typically brighter region)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely optic disc)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:  # Minimum size threshold
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
    
    return None


def smart_crop(image: np.ndarray, crop_size: Tuple[int, int] = config.CROP_SIZE, 
                auto_detect: bool = config.AUTO_CROP_ENABLED) -> np.ndarray:
    """
    Smart crop that attempts to center on optic disc if detected.
    
    Args:
        image: Input image
        crop_size: Desired crop size
        auto_detect: Whether to automatically detect optic disc
        
    Returns:
        Cropped image centered on optic disc or center of image
    """
    if not config.CROP_ENABLED:
        return image
    
    if auto_detect:
        # Try to detect optic disc
        optic_disc_center = detect_optic_disc(image)
        
        if optic_disc_center:
            # Crop around detected optic disc
            crop_dimension = min(crop_size)
            cropped = adaptive_crop(image, optic_disc_center, crop_dimension)
            
            # Resize to target size if needed
            if cropped.shape[:2] != crop_size:
                cropped = cv2.resize(cropped, (crop_size[1], crop_size[0]))
            
            return cropped
    
    # Default: center crop
    return center_crop(image, crop_size)


if __name__ == "__main__":
    print("Cropping module loaded successfully!")
    print(f"Crop enabled: {config.CROP_ENABLED}")
    print(f"Auto detect: {config.AUTO_CROP_ENABLED}")



