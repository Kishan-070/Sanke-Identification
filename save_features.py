import os
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Load MobileNetV2
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
model.eval()



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)

    with torch.inference_mode():
        features = model(img)

    return features.flatten().numpy()

dataset_path = "dataset"
features_list = []
labels = []

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        features = extract_features(img_path)
        features_list.append(features)
        labels.append(folder)

#features_list = np.array(features_list)

# Save
os.makedirs("features", exist_ok=True)

np.save("features/features.npy", features_list)
np.save("features/labels.npy", labels)

print("Features saved successfully!")