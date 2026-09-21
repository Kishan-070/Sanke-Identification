from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from snakes_details import snake_info
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import model_utility
import uuid

app = Flask(__name__)

# Load model
model =model_utility.load_model()

transform =model_utility.get_transform()

# Load saved features
features_list = np.load("features/features.npy")
labels = np.load("features/labels.npy")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)

        filename = str(uuid.uuid4())+"_" + file.filename
        path = os.path.join(upload_folder,filename)
        file.save(path)

        test_features = model_utility.extract_features(path,model,transform).reshape(1, -1)
        similarities = cosine_similarity(test_features, features_list)

        label_scores ={}
        for i,score in enumerate(similarities[0]):
            label = labels[i]
            if label not in label_scores or score > label_scores[label]:
                label_scores[label] =score

        sorted_labels = sorted(label_scores.items(), key = lambda x:x[1], reverse=True)
        top_3 = sorted_labels[:3]

        top_label = top_3[0][0]

        info = snake_info.get(top_label, {})

        return render_template(
            'index.html', 
            predictions = top_3,
            info=info,
            image = filename
            )

    return render_template('index.html', predictions = None)

if __name__ == '__main__':
    app.run(debug=True)