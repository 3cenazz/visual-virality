from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# Dataset not saved on Github. Saved on a local folder one level above PROJECT_ROOT
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
CROPS_DIR = DATASET_DIR / "images"
METADATA_DIR = DATASET_DIR / "metadata" / "per_page"
COMPLETED_DIR = DATASET_DIR / "completed_pages"

