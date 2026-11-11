
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Project paths
CONFIG_PATH = "detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
WEIGHTS_PATH = "model_weights/model_final.pth"

# Configure model
cfg = get_cfg()
cfg.merge_from_file(CONFIG_PATH)

# Newspaper Navigator has 7 classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

# Load trained checkpoint
cfg.MODEL.WEIGHTS = WEIGHTS_PATH

# Start with CPU for compatibility
cfg.MODEL.DEVICE = "cpu"

# Same confidence threshold as our Colab test
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Build predictor
predictor = DefaultPredictor(cfg)

print("Newspaper Navigator model loaded successfully!")
print("Architecture: Faster R-CNN R50-FPN")
print("Classes: 7")
print("Confidence threshold: 0.5")
print("Device:", cfg.MODEL.DEVICE)
print("MPS available:", torch.backends.mps.is_available())
