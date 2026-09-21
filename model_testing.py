import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics.pairwise import cosine_similarity
from snakes_details import snake_info
import model_utility


model =model_utility.load_model()

transform = model_utility.get_transform()

features_list = np.load("features/features.npy")
labels = np.load("features/labels.npy")

print("Features loaded successfully!")


test_image = "test_images/test.jpg"


test_features = model_utility.extract_features(test_image,model,transform).reshape(1, -1)


similarities = cosine_similarity(test_features, features_list)


label_scores = {}

for i, score in enumerate(similarities[0]):
    label = labels[i]
    
    if label not in label_scores or score > label_scores[label]:
        label_scores[label] = score


sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)

print("\nTop Snake Predictions:")
for label, score in sorted_labels[:3]:
    print(f"{label} (Similarity: {score:.4f})")


predicted_label = sorted_labels[0][0] 

print("\nPredicted Snake:", predicted_label)

info = snake_info.get(predicted_label, {})

print("\nDetails:")
for key, value in info.items():
    print(f"{key.capitalize()}: {value}")

