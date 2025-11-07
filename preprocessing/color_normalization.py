"""
Color normalization module for handling varying illumination across different imaging devices
"""

import numpy as np
import cv2
from typing import Tuple
import config


def z_score_normalize(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per channel.
    
    Normalizes each channel independently to have mean=0 and std=1.
    
    Args:
        image: Input image array (H, W, C)
        
    Returns:
        Normalized image array
    """
    normalized = image.copy().astype(np.float32)
    
    if len(image.shape) == 3:  # Color image
        for i in range(image.shape[2]):
            channel = image[:, :, i].astype(np.float32)
            mean = np.mean(channel)
            std = np.std(channel)
            
            if std > 0:
                normalized[:, :, i] = (channel - mean) / std
            else:
                normalized[:, :, i] = channel - mean
    else:  # Grayscale
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            normalized = (image.astype(np.float32) - mean) / std
        else:
            normalized = image.astype(np.float32) - mean
    
    # Clip values to reasonable range
    normalized = np.clip(normalized, -3, 3)
    
    return normalized


def min_max_normalize(image: np.ndarray, output_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Min-max normalization per channel.
    
    Args:
        image: Input image array
        output_range: Target range (min, max)
        
    Returns:
        Normalized image array
    """
    normalized = image.copy().astype(np.float32)
    
    if len(image.shape) == 3:  # Color image
        for i in range(image.shape[2]):
            channel = image[:, :, i].astype(np.float32)
            min_val = np.min(channel)
            max_val = np.max(channel)
            
            if max_val > min_val:
                normalized[:, :, i] = (channel - min_val) / (max_val - min_val)
                # Scale to target range
                normalized[:, :, i] = (output_range[1] - output_range[0]) * normalized[:, :, i] + output_range[0]
    else:  # Grayscale
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            normalized = (image.astype(np.float32) - min_val) / (max_val - min_val)
            normalized = (output_range[1] - output_range[0]) * normalized + output_range[0]
    
    return normalized


def per_image_normalize(image: np.ndarray) -> np.ndarray:
    """
    Per-image normalization to handle inter-image variations.
    
    Args:
        image: Input image array
        
    Returns:
        Normalized image array
    """
    # Convert to float
    img_float = image.astype(np.float32)
    
    # Normalize to 0-1 range
    img_min = img_float.min()
    img_max = img_float.max()
    
    if img_max > img_min:
        normalized = (img_float - img_min) / (img_max - img_min)
    else:
        normalized = img_float
    
    # Scale to 0-255 for display
    normalized = (normalized * 255).astype(np.uint8)
    
    return normalized


def histogram_matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histogram of source image to reference image.
    
    Args:
        source: Image to transform
        reference: Reference image
        
    Returns:
        Histogram-matched image
    """
    if len(source.shape) == 3:
        result = np.zeros_like(source)
        for i in range(3):
            source_channel = source[:, :, i]
            ref_channel = reference[:, :, i]
            
            # Calculate histograms
            src_hist, src_bins = np.histogram(source_channel.flatten(), 256, [0, 256])
            ref_hist, ref_bins = np.histogram(ref_channel.flatten(), 256, [0, 256])
            
            # Cumulative distribution functions
            src_cdf = np.cumsum(src_hist)
            ref_cdf = np.cumsum(ref_hist)
            
            # Normalize
            src_cdf_norm = (src_cdf - src_cdf[0]) / (src_cdf[-1] - src_cdf[0])
            ref_cdf_norm = (ref_cdf - ref_cdf[0]) / (ref_cdf[-1] - ref_cdf[0])
            
            # Mapping
            mapping = np.interp(src_cdf_norm, ref_cdf_norm, src_bins[:-1])
            result[:, :, i] = mapping[source_channel]
        
        return result.astype(np.uint8)
    else:
        # Single channel
        src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
        
        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)
        
        src_cdf_norm = (src_cdf - src_cdf[0]) / (src_cdf[-1] - src_cdf[0])
        ref_cdf_norm = (ref_cdf - ref_cdf[0]) / (ref_cdf[-1] - ref_cdf[0])
        
        mapping = np.interp(src_cdf_norm, ref_cdf_norm, np.arange(256))
        result = mapping[source].astype(np.uint8)
        
        return result


def normalize_color(image: np.ndarray, method: str = config.NORMALIZATION_METHOD) -> np.ndarray:
    """
    Apply color normalization to an image.
    
    Args:
        image: Input image array
        method: Normalization method ('z_score', 'min_max', 'per_image')
        
    Returns:
        Normalized image array
    """
    if not config.NORMALIZATION_ENABLED:
        return image
    
    if method == 'z_score':
        return z_score_normalize(image)
    elif method == 'min_max':
        return min_max_normalize(image)
    elif method == 'per_image':
        return per_image_normalize(image)
    else:
        print(f"Unknown normalization method: {method}. Using z_score.")
        return z_score_normalize(image)


if __name__ == "__main__":
    print("Color normalization module loaded successfully!")
    print(f"Normalization enabled: {config.NORMALIZATION_ENABLED}")
    print(f"Normalization method: {config.NORMALIZATION_METHOD}")




