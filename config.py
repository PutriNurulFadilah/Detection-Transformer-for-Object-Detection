import os
import torch

# Path configurations
HOME = os.getcwd()
MODEL_PATH = os.path.join(HOME, 'detr')
ANNOTATION_FILE_NAME = "_annotations.coco.json"

# Model configurations
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50-dc5'
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.8

# Training configurations
MAX_EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 3

# API Keys
ROBOFLOW_API_KEY = "your roboflow API key"