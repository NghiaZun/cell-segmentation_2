# Config file for cell segmentation/classification pipeline

# Paths
IMAGES_DIR = "images"
LABELS_DIR = "labels_yolo"
CSV_PATH = "train.csv"
MODEL_DIR = "models"
YOLO_WEIGHTS = "models/yolov8n.pt"
MASK_RCNN_WEIGHTS = "models/mask_rcnn.pth"

# Data
IMAGE_SIZE = (704, 520)
NUM_CLASSES = 4  # 3 cell types + background
CLASS_NAMES = ['shsy5y', 'astro', 'cort']

# Training
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda"  # or "cpu"

# WBF
WBF_IOU_THR = 0.5
WBF_SKIP_BOX_THR = 0.0
WBF_WEIGHTS = [1, 1]  # [YOLO, MaskRCNN]