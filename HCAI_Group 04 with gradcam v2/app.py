from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
import numpy as np
import cv2

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




        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_idxs = probs.topk(3)
            most_likely_idx = top_idxs[0].item()

        # Check if top-3 predictions contain an animal
        is_animal = True
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
        most_likely = most_likely[0].upper() + most_likely[1:]
        most_likely_prob = int(top_labels[0]['probability']*100)
        second_most = top_labels[1]['label']
        second_most = second_most[0].upper() + second_most[1:]
        second_most_prob = int(top_labels[1]['probability']*100)
        third_most = top_labels[2]['label']
        third_most = third_most[0].upper() + third_most[1:]
        third_most_prob = int(top_labels[2]['probability']*100)

        def get_conv_layer(model, conv_layer_name):
            for name, layer in model.named_modules():
                if name == conv_layer_name:
                    return layer

        def create_gradcam(model, input_tensor, class_index):
            conv_layer = get_conv_layer(model, "features.18")
            activations = None
            def forward_hook(module, input, output):
                nonlocal activations
                activations = output

            hook = conv_layer.register_forward_hook(forward_hook)

            input_tensor.requires_grad = True
            preds = model(input_tensor)
            loss = preds[0, int(class_index)]
            model.zero_grad()
            loss.backward()

            grads = input_tensor.grad.cpu().numpy()
            pooled_grads = np.mean(grads, axis=(0, 2, 3))
            hook.remove()

            activations = activations.detach().cpu().numpy()[0]
            for i in range(pooled_grads.shape[0]):
                activations[i, ...] *= pooled_grads[i]

            heatmap = np.maximum(np.mean(activations, axis=0), 0)
            heatmap /= np.max(heatmap)

            return heatmap

        def coloured_heatmap(img_original, heatmap, alpha=0.4):
            if isinstance(img_original, Image.Image):
                img = cv2.cvtColor(np.array(img_original), cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(img_original, cv2.IMREAD_COLOR)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            weighted_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)
            weighted_img = cv2.resize(weighted_img, (512, 512))
            return weighted_img

        heatmap = create_gradcam(model, input_tensor, most_likely_idx)
        gradcam = coloured_heatmap(file_path, heatmap)
        gradcam_file = f"gradcam_{file.filename}"
        gradcam_path = f"examples/{gradcam_file}"
        cv2.imwrite(gradcam_path, gradcam)

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
            "filename": file.filename,
            "most_likely": most_likely,
            "most_likely_prob": most_likely_prob,
            "second_most": second_most,
            "second_most_prob": second_most_prob,
            "third_most": third_most,
            "third_most_prob": third_most_prob,
            "gradcam_image": gradcam_file

        })
    else:
        return render_template("prediction.html")

@app.route('/examples/<filename>')
def serve_gradcam(filename):
    return send_from_directory('examples', filename)

if __name__ == "__main__":
    app.run(debug=True)