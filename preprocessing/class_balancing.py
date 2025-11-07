"""
Class balancing module for handling imbalanced glaucoma datasets
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.utils import resample
import config


def random_undersampling(X: np.ndarray, y: np.ndarray, 
                        target_ratio: Tuple[int, int] = config.TARGET_RATIO,
                        random_seed: int = config.RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform random undersampling of majority class to achieve target ratio.
    
    For glaucoma screening:
    - Class 0: Normal (NRG - No Referable Glaucoma)
    - Class 1: Glaucoma (RG - Referable Glaucoma)
    - Typical ratio: 1:2 (1 disease: 2 normal)
    
    Args:
        X: Feature array (images)
        y: Label array (binary: 0=normal, 1=glaucoma)
        target_ratio: Target ratio as (minority, majority) e.g., (1, 2)
        random_seed: Random seed for reproducibility
        
    Returns:
        Balanced X and y arrays
    """
    if not config.BALANCE_CLASSES:
        return X, y
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Separate majority and minority classes
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    class_0_count = len(class_0_indices)
    class_1_count = len(class_1_indices)
    
    print(f"Original class distribution:")
    print(f"  Class 0 (Normal): {class_0_count}")
    print(f"  Class 1 (Glaucoma): {class_1_count}")
    print(f"  Ratio: {class_1_count} : {class_0_count}")
    
    # Determine which is minority and majority
    if class_1_count < class_0_count:
        minority_class = 1
        minority_indices = class_1_indices
        majority_indices = class_0_indices
        majority_count = class_0_count
    else:
        minority_class = 0
        minority_indices = class_0_indices
        majority_indices = class_1_indices
        majority_count = class_1_count
    
    minority_count = len(minority_indices)
    
    # Calculate target size for majority class
    # If target ratio is (1, 2) and we have 100 minority samples,
    # we want 200 majority samples
    target_majority_size = int(minority_count * (target_ratio[1] / target_ratio[0]))
    
    # Undersample majority class
    if target_majority_size < majority_count:
        # Random undersample
        undersampled_indices = np.random.choice(
            majority_indices,
            size=target_majority_size,
            replace=False
        )
        balanced_majority_indices = undersampled_indices
    else:
        balanced_majority_indices = majority_indices
    
    # Combine indices
    balanced_indices = np.concatenate([minority_indices, balanced_majority_indices])
    balanced_indices = np.sort(balanced_indices)
    
    # Create balanced datasets
    if isinstance(X, (list, np.ndarray)) and isinstance(X[0], (list, np.ndarray)):
        # X is a list of images/arrays
        X_balanced = [X[i] for i in balanced_indices]
    else:
        # X is a numpy array
        X_balanced = X[balanced_indices]
    
    y_balanced = y[balanced_indices]
    
    # Print results
    balanced_class_0_count = np.sum(y_balanced == 0)
    balanced_class_1_count = np.sum(y_balanced == 1)
    
    print(f"\nBalanced class distribution:")
    print(f"  Class 0 (Normal): {balanced_class_0_count}")
    print(f"  Class 1 (Glaucoma): {balanced_class_1_count}")
    print(f"  Ratio: {balanced_class_1_count} : {balanced_class_0_count}")
    print(f"  Total samples: {len(X_balanced)} (Reduction: {len(X) - len(X_balanced)} samples)")
    
    return X_balanced, y_balanced


def stratified_resample(X: np.ndarray, y: np.ndarray,
                       target_ratio: Tuple[int, int] = config.TARGET_RATIO,
                       random_seed: int = config.RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stratified resampling to achieve target ratio.
    
    Uses sklearn's resample function for more robust resampling.
    
    Args:
        X: Feature array
        y: Label array
        target_ratio: Target ratio (minority, majority)
        random_seed: Random seed
        
    Returns:
        Balanced X and y
    """
    from sklearn.utils import resample
    
    # Separate by class
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    X_class_0 = X[class_0_indices] if isinstance(X, np.ndarray) else [X[i] for i in class_0_indices]
    X_class_1 = X[class_1_indices] if isinstance(X, np.ndarray) else [X[i] for i in class_1_indices]
    y_class_0 = y[class_0_indices]
    y_class_1 = y[class_1_indices]
    
    # Determine minority and majority
    if len(X_class_1) < len(X_class_0):
        majority_class = 0
        majority_X = X_class_0
        majority_y = y_class_0
        minority_X = X_class_1
        minority_y = y_class_1
    else:
        majority_class = 1
        majority_X = X_class_1
        majority_y = y_class_1
        minority_X = X_class_0
        minority_y = y_class_0
    
    # Calculate target size
    target_size = int(len(minority_X) * (target_ratio[1] / target_ratio[0]))
    
    # Resample majority class
    if len(majority_X) > target_size:
        if isinstance(X, np.ndarray):
            majority_X_resampled = resample(
                majority_X,
                replace=False,
                n_samples=target_size,
                random_state=random_seed
            )
        else:
            resampled_indices = np.random.choice(
                len(majority_X),
                size=target_size,
                replace=False
            )
            majority_X_resampled = [majority_X[i] for i in resampled_indices]
            majority_y_resampled = majority_y[resampled_indices]
    else:
        majority_X_resampled = majority_X
        majority_y_resampled = majority_y
    
    # Combine
    if isinstance(X, np.ndarray):
        X_balanced = np.concatenate([minority_X, majority_X_resampled])
        y_balanced = np.concatenate([minority_y, majority_y_resampled])
    else:
        X_balanced = minority_X + list(majority_X_resampled)
        y_balanced = np.concatenate([minority_y, majority_y_resampled])
    
    return X_balanced, y_balanced


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced dataset.
    
    Useful for loss functions that accept class weights.
    
    Args:
        y: Label array
        
    Returns:
        Dictionary mapping class to weight
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (len(unique) * count)
    
    return weights


def get_balanced_split(X: np.ndarray, y: np.ndarray,
                      train_ratio: float = config.TRAIN_SPLIT,
                      val_ratio: float = config.VAL_SPLIT,
                      test_ratio: float = config.TEST_SPLIT,
                      target_ratio: Tuple[int, int] = config.TARGET_RATIO,
                      random_seed: int = config.RANDOM_SEED) -> Tuple:
    """
    Get stratified train/validation/test splits with balancing.
    
    Args:
        X: Feature array
        y: Label array
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        target_ratio: Target class ratio
        random_seed: Random seed
        
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_seed)
    
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, stratify=y
    )
    
    # Second split: train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio/(train_ratio+val_ratio),
        random_state=random_seed, stratify=y_temp
    )
    
    # Apply balancing to training set only
    if config.BALANCE_CLASSES:
        X_train, y_train = random_undersampling(X_train, y_train, target_ratio, random_seed)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    print("Class balancing module loaded successfully!")
    print(f"Balance classes: {config.BALANCE_CLASSES}")
    print(f"Target ratio: {config.TARGET_RATIO}")



