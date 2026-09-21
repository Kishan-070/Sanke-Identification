import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np

# Load MobileNetV2 model
def load_model():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

# Image transformation
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# Feature extraction
def extract_features(img_path, model, transform):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.inference_mode():
        features = model(img)
    return features.flatten().numpy()