'''
will load the Newspaper Navigator model using the configuration.
What this file does
- Loads the Faster R-CNN model architecture.
- Loads the trained weights from model_final.pth.
- Sets the seven Newspaper Navigator classes.
- Applies the 0.5 confidence threshold.
- Uses your Mac's MPS device.
'''

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from src.config import (
    CONFIG_PATH,
    WEIGHTS_PATH,
    NUM_CLASSES,
    CONFIDENCE_THRESHOLD,
    DEVICE,
)


def load_model():
    """Load the trained Newspaper Navigator object detection model."""

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {WEIGHTS_PATH}"
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Detectron2 config not found: {CONFIG_PATH}"
        )

    if DEVICE == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available. Check your PyTorch installation."
        )

    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIG_PATH))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.WEIGHTS = str(WEIGHTS_PATH)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.DEVICE = DEVICE

    predictor = DefaultPredictor(cfg)

    print("Newspaper Navigator model loaded successfully.")
    print(f"Architecture: Faster R-CNN R50-FPN")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Device: {DEVICE}")

    return predictor