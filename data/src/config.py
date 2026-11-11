'''
It centralizes the paths and model settings. 
That way, if we later change the confidence threshold or output location, 
we only need to change it in one place.
'''

from pathlib import Path

# Project root: the folder containing this src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input data
SOURCE_DIR = PROJECT_ROOT / "downloaded_newspapers"
WEIGHTS_PATH = PROJECT_ROOT / "model_weights" / "model_final.pth"
CONFIG_PATH = (
    PROJECT_ROOT
    / "detectron2"
    / "configs"
    / "COCO-Detection"
    / "faster_rcnn_R_50_FPN_3x.yaml"
)

# Output directories
DATASET_DIR = PROJECT_ROOT / "dataset"
CROPS_DIR = DATASET_DIR / "images"
METADATA_DIR = DATASET_DIR / "metadata" / "per_page"
COMPLETED_DIR = DATASET_DIR / "completed_pages"

# Model settings
NUM_CLASSES = 7
CONFIDENCE_THRESHOLD = 0.5
DEVICE = "mps"

# Newspaper Navigator class names
CLASS_NAMES = [
    "Illustration/Photograph",
    "Photograph",
    "Comics/Cartoon",
    "Editorial Cartoon",
    "Map",
    "Headline",
    "Advertisement",
]

