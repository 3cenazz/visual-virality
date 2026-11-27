import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset not saved on Github. Default: a local folder one level above PROJECT_ROOT.
# Point VV_DATASET_DIR at your copy of the shared OneDrive folder to use it directly, e.g.
#   export VV_DATASET_DIR="$HOME/Library/CloudStorage/OneDrive-epfl.ch/Dateien von Ecenaz Egri - dataset"
DATASET_DIR = Path(os.environ.get("VV_DATASET_DIR", PROJECT_ROOT.parent / "dataset")).expanduser()
CROPS_DIR = DATASET_DIR / "images"
METADATA_DIR = DATASET_DIR / "metadata" / "per_page"
COMPLETED_DIR = DATASET_DIR / "completed_pages"

# Embeddings live in the shared dataset folder (dataset/embeddings/), not in the repo.
# Override with VV_EMBEDDINGS_DIR if yours are somewhere else.
EMBEDDINGS_DIR = Path(os.environ.get("VV_EMBEDDINGS_DIR", DATASET_DIR / "embeddings")).expanduser()
