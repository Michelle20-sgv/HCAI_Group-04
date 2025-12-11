from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('templates'))
template = env.get_template('prediction.html')

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

@app.route('/explanability')
def explanability():
    return render_template("explanability.html")

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')


@app.route("/predict", methods=["POST","GET"])
def predict():
    if request.method == "POST":
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

        #example preprocess steps
        exampleDir = "examples/"
        if not os.path.exists(exampleDir):
            os.makedirs(exampleDir)



        resizeFunction = transforms.Resize(256)
        imgStep1 = resizeFunction(img)
        centerCropFunction = transforms.CenterCrop(224)
        imgStep2 = centerCropFunction(imgStep1)
        toTensorFuction = transforms.ToTensor()
        imgStep3 = toTensorFuction(imgStep2)
        normalFunction = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        imgStep4 = normalFunction(imgStep3)

        imgStep1.save(os.path.join(exampleDir,"Resized.png"))
        imgStep2.save(os.path.join(exampleDir, "CenterCrop.png"))
        #imgStep4.save(os.path.join(exampleDir, "Normalized.png"))


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

        most_likely = top_labels[0]['label']
        most_likely_prob = int(top_labels[0]['probability']*100)
        second_most = top_labels[1]['label']
        second_most_prob = int(top_labels[1]['probability']*100)
        third_most = top_labels[2]['label']
        third_most_prob = int(top_labels[2]['probability']*100)



        if not is_animal:
            return jsonify({"error": "❌ Please upload an animal image only."}), 400

        # Use first matching animal label as prediction
        prediction_label = None
        for pred in top_labels:
            words = pred["label"].split()
            if any(a in word for a in animal_keywords for word in words):
                prediction_label = pred["label"]
                break

        print(most_likely_prob)
        print(second_most_prob)

        return jsonify({
            "prediction": prediction_label if prediction_label else "Unknown Animal ❓",
            "filename": file.filename,
            "most_likely": most_likely,
            "most_likely_prob": most_likely_prob,
            "second_most": second_most,
            "second_most_prob": second_most_prob,
            "third_most": third_most,
            "third_most_prob": third_most_prob
        })
    else:
        return render_template("prediction.html")

if __name__ == "__main__":
    app.run(debug=True)