import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = keras.models.load_model(r"C:\Users\Dhanush G R\Documents\mango_leaf_web\mango_leaf_disease_model.h5")

# Define image size
image_size = (224, 224)

# Class labels (modify if needed)
class_labels = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
]

def predict_image(img_path):
    """Preprocess and predict the uploaded image"""
    img = image.load_img(img_path, target_size=image_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["file"]
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Make prediction
            predicted_class, confidence = predict_image(file_path)

            return render_template("result.html", filename=filename, prediction=predicted_class, confidence=confidence)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"))

if __name__ == "__main__":
    app.run(debug=True)
