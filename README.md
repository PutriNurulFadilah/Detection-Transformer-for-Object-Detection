# Automatic Parasite Detection on Bivalve Shells Using Detection Transformer

A deep learning-based system for detecting parasitic worms on bivalve shells using the Detection Transformer (DETR) with ResNet backbones.

---

## 📊 Dataset
- Total: **521 images**
  - 381 images (secondary data from Google Images)
  - 140 images (primary data collected in TPI Bondet, Cirebon, Indonesia)
- Classes:
  - Parasite
  - Non-parasite
- Image size: **224 × 224**

---

## Model
Download DETR model from HuggingFace (or other resources):  
https://huggingface.co/docs/transformers/model_doc/detr

```python
from transformers import DetrForObjectDetection
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```

you can also clone this repository: 
- if you have annotated data, you can directly use without roboflow API key
- if you have raw data, you should annotate it first using roboflow (https://app.roboflow.com/) or another platform
- if use this code, you should adjust the image size to 224x224

# HOW TO USE THIS REPOSITORY: 
1. Run config.py (provide your API key from Roboflow)
2. Adjust your model configuration
3. Run train.py, then modify your workspace name and project name
4. Run testing.py
5. Results will be saved in the evaluation_result folder

---
## Detection Pipeline



1. Input annotated images
2. Feature extraction using CNN backbone (ResNet)
3. Feature flattening + positional encoding
4. Transformer encoder-decoder processing
5. Object queries predict bounding boxes
6. Feed Forward Network (FFN) produces:
  - Bounding box coordinates
  - Class labels (parasite / non-parasite)
  - Confidence scores
In addition to standard DETR, this study explores dilated convolution applied to ResNet backbones:
