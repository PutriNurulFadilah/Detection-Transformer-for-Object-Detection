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

## 🧠 Model
Download DETR model from HuggingFace (or other resources):  
https://huggingface.co/docs/transformers/model_doc/detr

```python
from transformers import DetrForObjectDetection
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")```

you can also clone this repository: - if you have annotated data, you can directly use without roboflow API key - if you have raw data, you should annotate it first using roboflow (https://app.roboflow.com/) or another platform - if use this code, you should adjust the image size to 224x224
