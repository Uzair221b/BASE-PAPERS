"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) module
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import config


def apply_clahe(image: np.ndarray, 
                tile_size: Tuple[int, int] = config.CLAHE_TILE_SIZE,
                clip_limit: float = config.CLAHE_CLIP_LIMIT,
                apply_to_green_only: bool = config.CLAHE_APPLY_TO_GREEN_CHANNEL_ONLY) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.
    
    CLAHE is crucial for glaucoma detection as it improves visibility of:
    - Optic disc and cup boundaries
    - Neuroretinal rim
    - Retinal nerve fiber layer defects
    
    Args:
        image: Input image array (H, W, C)
        tile_size: Grid size for CLAHE (height, width)
        clip_limit: Contrast limiting factor
        apply_to_green_only: If True, apply CLAHE only to green channel (common for fundus)
        
    Returns:
        Image with enhanced contrast
    """
    if not config.CLAHE_ENABLED:
        return image
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    if len(image.shape) == 2:  # Grayscale image
        return clahe.apply(image)
    
    elif len(image.shape) == 3:  # Color image
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        if apply_to_green_only:
            # Common practice: apply CLAHE to green channel only (best contrast for fundus)
            # Convert to HSV and work with Value channel
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return enhanced
        else:
            # Apply CLAHE to L channel in LAB color space
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced
    
    return image


def apply_clahe_to_green_channel(image: np.ndarray,
                                   tile_size: Tuple[int, int] = config.CLAHE_TILE_SIZE,
                                   clip_limit: float = config.CLAHE_CLIP_LIMIT) -> np.ndarray:
    """
    Apply CLAHE specifically to the green channel of the image.
    
    In fundus images, the green channel typically has the best contrast for:
    - Optic disc visibility
    - Vessel clarity
    - Nerve fiber layer details
    
    Args:
        image: Input RGB image
        tile_size: CLAHE tile grid size
        clip_limit: Contrast limiting factor
        
    Returns:
        Enhanced image with improved green channel contrast
    """
    if len(image.shape) != 3:
        raise ValueError("Input must be a color image")
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    # Split channels
    r, g, b = cv2.split(image)
    
    # Apply CLAHE to green channel only
    g_enhanced = clahe.apply(g)
    
    # Merge channels
    enhanced = cv2.merge([r, g_enhanced, b])
    
    return enhanced


def apply_clahe_to_l_channel(image: np.ndarray,
                              tile_size: Tuple[int, int] = config.CLAHE_TILE_SIZE,
                              clip_limit: float = config.CLAHE_CLIP_LIMIT) -> np.ndarray:
    """
    Apply CLAHE to the L (lightness) channel in LAB color space.
    
    This method preserves color while enhancing contrast.
    
    Args:
        image: Input RGB image
        tile_size: CLAHE tile grid size
        clip_limit: Contrast limiting factor
        
    Returns:
        Enhanced image in RGB color space
    """
    if len(image.shape) != 3:
        raise ValueError("Input must be a color image")
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def adaptive_clahe(image: np.ndarray, 
                   tile_size: Optional[Tuple[int, int]] = None,
                   clip_limit: Optional[float] = None) -> np.ndarray:
    """
    Apply CLAHE with adaptive parameters based on image characteristics.
    
    Args:
        image: Input image
        tile_size: Tile size (if None, auto-calculate)
        clip_limit: Clip limit (if None, auto-calculate)
        
    Returns:
        Enhanced image
    """
    # Auto-calculate tile size if not provided
    if tile_size is None:
        height, width = image.shape[:2]
        # Use smaller tiles for smaller images
        if min(height, width) < 256:
            tile_size = (4, 4)
        elif min(height, width) < 512:
            tile_size = (8, 8)
        else:
            tile_size = (16, 16)
    
    # Auto-calculate clip limit based on image variance
    if clip_limit is None:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        variance = np.var(gray)
        # Lower clip limit for high variance (more detail), higher for low variance
        clip_limit = max(2.0, min(4.0, 3.0 - variance / 10000))
    
    return apply_clahe(image, tile_size, clip_limit)


if __name__ == "__main__":
    print("CLAHE processing module loaded successfully!")
    print(f"CLAHE enabled: {config.CLAHE_ENABLED}")
    print(f"Tile size: {config.CLAHE_TILE_SIZE}")
    print(f"Clip limit: {config.CLAHE_CLIP_LIMIT}")


