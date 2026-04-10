download detr model from huggingface (you can download from another resources) 	https://huggingface.co/docs/transformers/model_doc/detr
```python
from transformers import DetrForObjectDetection
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```
	

you can also clone this repository:
- if you have annotated data, you can directly use without roboflow API key
- if you have raw data, you should annotate it first using roboflow (https://app.roboflow.com/) or another platform 
- if use this code, you should adjust the image size to 224x224

# HOW TO USE THIS REPO:
1. run config.py file (provide your API key from roboflow) 
2. adjust your own model configuration
3. run train.py file, then modify your workspace name, and project name
4. run testing.py file
5. the result will be save in "evaluation_result" folder 
