from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
import requests
import google.generativeai as genai
import numpy as np
import cv2


#LLM INITIALISATION

genai.configure(api_key="API KEY GOES HERE")
LLM_ON = False # toggles LLM api call
PRINT_MODELS = False # toggles printing available llm models to console

# print list of models if toggled on
def printModelList():
    if PRINT_MODELS:
        print("=== AVAILABLE MODELS ===")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"Model: {model.name}")
printModelList()

# use api to generate llm explanation if toggled on
def generate_explanation(top_predictions, original_image_path, gradcam_image_path):
    if LLM_ON:

        prompt = f"""You are explaining an AI image classifier's prediction to a user.

        The classifier's top 3 predictions were:
        1. {top_predictions[0]['label']} - {top_predictions[0]['probability']}%
        2. {top_predictions[1]['label']} - {top_predictions[1]['probability']}%
        3. {top_predictions[2]['label']} - {top_predictions[2]['probability']}%

        I'm showing you the original image and a GradCAM heatmap where red/yellow areas show where the AI focused its attention.

        Based on what you see in both images:
        1. What visual features did the model focus on (look at the heatmap)?
        2. Does the model's attention make sense for predicting "{top_predictions[0]['label']}"?
        3. Could the model be making a mistake based on what it's focusing on?

        Keep your explanation 3-4 sentences, simple and clear."""

        try:
            print("=== USING GOOGLE GEMINI WITH VISION ===")

            from PIL import Image

            # load images
            original_img = Image.open(original_image_path)
            gradcam_img = Image.open(gradcam_image_path)

            # set model
            model = genai.GenerativeModel('gemini-2.5-flash')

            # send prompt + images
            response = model.generate_content([prompt, original_img, gradcam_img])

            explanation = response.text
            print(f"Generated explanation: {explanation}")
            return explanation

        except Exception as e:
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return 'Unable to generate explanation at this time.'
    else:
        return ("LLM turned off")



# initialise Flask
app = Flask(__name__)

# Create upload folder if not exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# create gradcam folder
gradcam_dir = "gradcams/"
if not os.path.exists(gradcam_dir):
    os.makedirs(gradcam_dir)

# create examples folder
exampleDir = "examples/"
if not os.path.exists(exampleDir):
    os.makedirs(exampleDir)


# Load pretrained MobileNet model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Load ImageNet labels
with open("imagenet_class_index.json") as f:
    idx_to_label = json.load(f)


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

        top_labels = []
        for prob, idx in zip(top_probs, top_idxs):
            # Normalize label
            label = idx_to_label[str(idx.item())][1].lower().replace("_", " ")
            top_labels.append({"label": label, "probability": round(float(prob) * 100, 1)})




        most_likely = top_labels[0]['label']
        most_likely_prob = int(top_labels[0]['probability'])
        second_most = top_labels[1]['label']
        second_most_prob = int(top_labels[1]['probability'])
        third_most = top_labels[2]['label']
        third_most_prob = int(top_labels[2]['probability'])



        # Gradcam functions
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


        #run gradcam functions
        heatmap = create_gradcam(model, input_tensor, most_likely_idx)
        gradcam = coloured_heatmap(file_path, heatmap)

        #set gradcam filename and path
        gradcam_file = f"gradcam_{file.filename}"
        gradcam_path = f"gradcams/{gradcam_file}"
        cv2.imwrite(gradcam_path, gradcam)

        # generate explanation using LLM
        explanation = generate_explanation(top_labels, file_path, gradcam_path)
        print("Explanation: ", explanation)


        return jsonify({
            "filename": file.filename,
            "most_likely": most_likely,
            "most_likely_prob": most_likely_prob,
            "second_most": second_most,
            "second_most_prob": second_most_prob,
            "third_most": third_most,
            "third_most_prob": third_most_prob,
            "explanation": explanation,
            "gradcam_filename": gradcam_file
        })
    else:
        return render_template("prediction.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/gradcams/<filename>")
def serve_gradcam(filename):
    return send_from_directory("gradcams", filename)


if __name__ == "__main__":
    app.run(debug=True)