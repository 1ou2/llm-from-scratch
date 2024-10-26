# src/data/__init__.py
"""Data processing and dataset utilities."""

import logging
from pathlib import Path
from .tokenizer import Tokenizer
from .dataset import Dataset, DatasetFactory
#from .preprocessing import clean_text, normalize_text

# Setup logging for the data subpackage
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Create necessary directories
data_dir = Path(__file__).parent.parent.parent / 'data'
(data_dir / 'raw').mkdir(exist_ok=True)
(data_dir / 'processed').mkdir(exist_ok=True)

#__all__ = ["Tokenizer", "Dataset", "clean_text", "normalize_text"]
__all__ = ["Tokenizer", "Dataset", "DatasetFactory"]