from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import os
from werkzeug.utils import secure_filename
from PIL import Image

application = Flask(__name__, template_folder='templates')
app = application
# UPLOAD_FOLDER = 'static/uploads'
# application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

alex_loaded = load_model("artifacts/mri_classifier_local_v3.h5")

# Define your class labels based on your model training
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

def process_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = img_array.reshape((1, 256, 256, 3))
    
    # img = image.load_img(image_path, target_size=(256, 256))
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array = preprocess_input(img_array)
    return img_array

def predict_image(image_array):
    # Perform prediction
    prediction = alex_loaded.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    return predicted_class


@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file to the uploads folder
    # filename = secure_filename(file.filename)
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # file.save(file_path)
    
    # Process the uploaded image
    # img_array = process_image(file_path)
    img_array = process_image(file.stream)
    
    # Perform prediction
    predicted_class = predict_image(img_array)
    
    # Prepare the result to display in the HTML page
    prediction_result = f'Predicted class: {predicted_class}'
    # image_url = f'/static/uploads/{filename}'  # Assuming 'uploads' folder is inside 'static'

    return render_template('index.html', prediction=prediction_result) #, image_url=image_url)


if __name__ == '__main__':
    app.run(host="0.0.0.0")