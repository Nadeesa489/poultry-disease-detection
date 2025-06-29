import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("poultry_disease_model.h5")

# Labels for prediction
labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Main route: upload + prediction
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    img_path = None

    if request.method == "POST":
        if "image" not in request.files:
            return "No file part", 400

        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        # Save uploaded image
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)
        img_path = os.path.join(upload_folder, file.filename)
        file.save(img_path)

        # Preprocess image
        img = load_img(img_path, target_size=(224, 224))
        image_array = img_to_array(img)
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction_index = np.argmax(model.predict(image_array), axis=1)[0]
        prediction = labels[prediction_index]

    return render_template("page.html", predict=prediction, image_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
