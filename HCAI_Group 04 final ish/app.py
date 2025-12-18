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


LLM_ON = True # toggles LLM api call
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

        genai.configure(api_key="KEY_HERE") #<----TO USE LLM ENTER API KEY HERE


        prompt = f"""You are explaining an AI animal image classifier's prediction to a user. It should be used only for animals.

        The classifier's top 3 predictions were:
        1. {top_predictions[0]['label']} - {top_predictions[0]['probability']}%
        2. {top_predictions[1]['label']} - {top_predictions[1]['probability']}%
        3. {top_predictions[2]['label']} - {top_predictions[2]['probability']}%

        I'm showing you the original image and a GradCAM heatmap where red/yellow areas show where the AI focused its attention.

        Based on what you see in both images:
        1. What visual features did the model focus on (look at the heatmap)?
        2. Does the model's attention make sense for predicting "{top_predictions[0]['label']}"? 
        3  Why could "{top_predictions[1]['label']}" and "{top_predictions[2]['label']}" have been predicted (if their predicted probabilities aren't negligible)? 
        4. Could the model be making a mistake based on what it's focusing on?
        5. Has the model correctly recognised if the image contains an animal?
        Keep your explanation 4-5 sentences, simple and clear. Clearly state if the image does not appear to include an animal"""

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

# Define imagnet animal classes
ANIMAL_CLASS_RANGE = range(0,398)

# Check if a class corresponds to an animal in imagenet
def is_animal_class(class_idx):
    return class_idx in ANIMAL_CLASS_RANGE

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

        #animal validation
        has_animal_in_top3 = any(is_animal_class(idx.item()) for idx in top_idxs)

        #returns error if animal not detected in top 3 predictions
        if not has_animal_in_top3:
            top_three_labels = [idx_to_label[str(idx.item())][1].replace("_", " ") for idx in top_idxs]
            return jsonify({
                "error": f"No animals detected. Top predictions: {', '.join(top_three_labels)}"

            }), 400


        top_labels = []
        for prob, idx in zip(top_probs, top_idxs):
            # Normalize label
            label = idx_to_label[str(idx.item())][1].lower().replace("_", " ")
            top_labels.append({"label": label, "probability": round(float(prob) * 100, 1)})




        most_likely = top_labels[0]['label']
        most_likely_prob = int(top_labels[0]['probability'])
        most_likely_prob_50 = most_likely_prob - 2
        if most_likely_prob < 100:
            most_likely_prob_200 = most_likely_prob + 1
        else:
            most_likely_prob_200 = most_likely_prob
        most_likely_prob_low = abs(most_likely_prob - 20)
        most_likely_prob_high = abs(most_likely_prob - 30)
        second_most = top_labels[1]['label']
        second_most_prob = int(top_labels[1]['probability'])
        second_most_prob_50 = second_most_prob + 1
        second_most_prob_low = 100 - most_likely_prob_low - 4
        second_most_prob_high = 100 - most_likely_prob_high - 2
        third_most = top_labels[2]['label']
        third_most_prob = int(top_labels[2]['probability'])
        third_most_prob_50 = third_most_prob + 1
        if third_most_prob > 1:
            third_most_prob_200 = third_most_prob - 1
        else:
            third_most_prob_200 = third_most_prob
        third_most_prob_low = 0
        third_most_prob_high = 0
        if most_likely_prob_low < second_most_prob_low:
            temp = most_likely_prob_low
            most_likely_prob_low = second_most_prob_low
            second_most_prob_low = temp
        if most_likely_prob_high < second_most_prob_high:
            temp = most_likely_prob_high
            most_likely_prob_high = second_most_prob_high
            second_most_prob_high = temp


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
        print(most_likely_prob_200)

        return jsonify({
            "filename": file.filename,
            "most_likely": most_likely,
            "most_likely_prob": most_likely_prob,
            "most_likely_prob_50": most_likely_prob_50,
            "most_likely_prob_200": most_likely_prob_200,
            "most_likely_prob_low": most_likely_prob_low,
            "most_likely_prob_high": most_likely_prob_high,
            "second_most": second_most,
            "second_most_prob": second_most_prob,
            "second_most_prob_50": second_most_prob_50,
            "second_most_prob_low": second_most_prob_low,
            "second_most_prob_high": second_most_prob_high,
            "third_most": third_most,
            "third_most_prob": third_most_prob,
            "third_most_prob_50": third_most_prob_50,
            "third_most_prob_200": third_most_prob_200,
            "third_most_prob_low": third_most_prob_low,
            "third_most_prob_high": third_most_prob_high,
            "explanation": explanation,
            "gradcam_filename": gradcam_file
        })
    else:
        return render_template("prediction.html")

@app.route('/epoch50')
def epoch50():
    return render_template('epoch50.html')

@app.route('/epoch100')
def epoch100():
    return render_template('epoch100.html')

@app.route('/epoch200')
def epoch200():
    return render_template('epoch200.html')

@app.route('/learninglow')
def learninglow():
    return render_template('learninglow.html')

@app.route('/learninghigh')
def learninghigh():
    return render_template('learninghigh.html')

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/gradcams/<filename>")
def serve_gradcam(filename):
    return send_from_directory("gradcams", filename)


if __name__ == "__main__":
    app.run(debug=True)