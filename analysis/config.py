"""
TODO: set global project root in the main file and not for each folder (i.e. data pipeline and analysis pipeline)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
CROPS_DIR = DATASET_DIR / "images"
METADATA_DIR = DATASET_DIR / "metadata" / "per_page"
COMPLETED_DIR = DATASET_DIR / "completed_pages"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"


