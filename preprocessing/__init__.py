"""
Glaucoma Preprocessing Pipeline

A comprehensive preprocessing pipeline for glaucoma detection in fundus images,
implementing five state-of-the-art techniques based on research analysis.
"""

from .pipeline import GlaucomaPreprocessingPipeline, quick_preprocess, batch_preprocess
from . import config

__version__ = "1.0.0"
__author__ = "Glaucoma Detection Research Team"

__all__ = [
    'GlaucomaPreprocessingPipeline',
    'quick_preprocess',
    'batch_preprocess',
    'config'
]




