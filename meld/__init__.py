"""MELD: A multilingual and multi-domain dataset for named entity recognition.

Supports fully reproducible downloading, processing, and format normalization of NER datasets.
"""

from meld.formats import local_dataset_names, local_datasets

__all__ = [
    "local_dataset_names",
    "local_datasets",
]
