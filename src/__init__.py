"""
llm-from-scratch: A from-scratch implementation of a Language Model for educational purposes.

This package provides tools and utilities for building and training transformer-based
language models from the ground up.
"""

# Version of the llm-from-scratch package
__version__ = "0.1.0"

# Data processing components
from .data.tokenizer import Tokenizer
from .data.dataset import Dataset

# what can be imported from the package
__all__ = [
    "Tokenizer",
    "Dataset",
]

# Package metadata
__author__ = "Gabriel Pastor"
__email__ = "contact@1ou2.com"
__description__ = "A from-scratch implementation of a Language Model"