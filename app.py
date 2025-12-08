from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import json

app = Flask(__name__)

# Create upload folder if not exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pretrained MobileNet model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Load ImageNet labels
with open("imagenet_class_index.json") as f:
    idx_to_label = json.load(f)

# Expanded animal keywords
animal_keywords = [
    # Dogs
    "dog", "retriever", "husky", "terrier", "beagle", "pug", "chihuahua", "bulldog", "dalmatian", "labrador",
    # Cats
    "cat", "siamese", "persian", "tabby", "maine coon", "egyptian", "sphynx",
    # Birds
    "bird", "parrot", "cockatoo", "macaw", "penguin", "ostrich", "flamingo", "robin", "sparrow", "canary",
    # Other animals
    "lion", "tiger", "horse", "sheep", "cow", "elephant", "bear", "monkey", "panda", "zebra", "giraffe",
    "rabbit", "fox", "wolf", "deer", "kangaroo", "otter", "alligator", "crocodile", "hippopotamus"
]

# Preprocess function
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/prediction')
def prediction_page():
    return render_template("prediction.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Open image
    img = Image.open(file_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_idxs = probs.topk(3)

    # Check if top-3 predictions contain an animal
    is_animal = False
    top_labels = []

    for prob, idx in zip(top_probs, top_idxs):
        # Normalize label
        label = idx_to_label[str(idx.item())][1].lower().replace("_", " ")
        top_labels.append({"label": label, "probability": float(prob)})

        # Check if any word in label matches our animal keywords
        words = label.split()
        if any(a in word for a in animal_keywords for word in words) and prob >= 0.3:
            is_animal = True

    if not is_animal:
        return jsonify({"error": "❌ Please upload an animal image only."}), 400

    # Use first matching animal label as prediction
    prediction_label = None
    for pred in top_labels:
        words = pred["label"].split()
        if any(a in word for a in animal_keywords for word in words):
            prediction_label = pred["label"]
            break

    return jsonify({
        "prediction": prediction_label if prediction_label else "Unknown Animal ❓",
        "filename": file.filename
    })


if __name__ == "__main__":
    app.run(debug=True)
