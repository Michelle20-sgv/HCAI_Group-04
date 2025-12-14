from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
import requests
import google.generativeai as genai

# HuggingFace API setup
genai.configure(api_key="AIzaSyCdA5LF_woXoYE5JITailFbJKjByeN9yKg")

#print("=== AVAILABLE MODELS ===")
#for model in genai.list_models():
#    if 'generateContent' in model.supported_generation_methods:
#        print(f"Model: {model.name}")


def generate_explanation(top_predictions):
    """Generate an explanation using Google Gemini API"""

    prompt = f"""You are explaining an AI image classifier's prediction to a user.

The classifier's top 3 predictions were:
1. {top_predictions[0]['label']} - {top_predictions[0]['probability']}%
2. {top_predictions[1]['label']} - {top_predictions[1]['probability']}%
3. {top_predictions[2]['label']} - {top_predictions[2]['probability']}%

Explain in 2-3 sentences why the model likely made this prediction and what visual features it probably focused on. Keep it simple and clear."""

    try:
        print("=== USING GOOGLE GEMINI API ===")

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)

        explanation = response.text
        return explanation

    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return 'Unable to generate explanation at this time.'








# initialise Flask
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


        top_labels = []
        for prob, idx in zip(top_probs, top_idxs):
            # Normalize label
            label = idx_to_label[str(idx.item())][1].lower().replace("_", " ")
            top_labels.append({"label": label, "probability": round(float(prob) * 100, 1)})


        #generate explanation using LLM
        explanation = generate_explanation(top_labels)
        print("Explanation: ", explanation)

        most_likely = top_labels[0]['label']
        most_likely_prob = int(top_labels[0]['probability'])
        second_most = top_labels[1]['label']
        second_most_prob = int(top_labels[1]['probability'])
        third_most = top_labels[2]['label']
        third_most_prob = int(top_labels[2]['probability'])


        print(most_likely)
        print(most_likely_prob)
        print(second_most)
        print(second_most_prob)
        print(third_most)
        print(third_most_prob)
        return jsonify({
            "prediction": most_likely,
            "filename": file.filename,
            "most_likely": most_likely,
            "most_likely_prob": most_likely_prob,
            "second_most": second_most,
            "second_most_prob": second_most_prob,
            "third_most": third_most,
            "third_most_prob": third_most_prob,
            "explanation": explanation
        })
    else:
        return render_template("prediction.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)