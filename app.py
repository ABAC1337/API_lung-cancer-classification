from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger, swag_from
import numpy as np
import tensorflow as tf
import os
import logging
import cv2
from datetime import datetime

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load Model
model = tf.keras.models.load_model('CNN_model.h5')

# Classes
class_labels = ['Benign', 'Malignant', 'Normal']

# Declare Flask
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# Swagger config
swagger = Swagger(app, config={
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "swagger_ui": True,
    "specs_route": "/apidocs/"
})

# Preprocess Before Predict
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1) 
    img = np.expand_dims(img, axis=0)  
    return img

# Test API
@app.route('/')
def home():
    return jsonify({"message": "Flask API is running for Lung Cancer Classification!"})

# API Predict
@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Upload CT-Scan image file'
        }
    ],
    'responses': {
        200: {
            'description': 'Prediction result',
            'examples': {
                'application/json': {
                    'prediction': 'Benign',
                    'probabilities': {
                        'Benign': '95%',
                        'Malignant': '4%',
                        'Normal': '1%'
                    }
                }
            }
        }
    }
})
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img)
        probabilities = predictions[0]

        # Konversi ke index dan label prediksi
        predicted_class_index = int(np.argmax(probabilities))
        predicted_label = class_labels[predicted_class_index]

        # Ubah probabilitas ke persentase
        probability_dict = {
            class_labels[i]: f"{probabilities[i] * 100:.2f}%" for i in range(len(class_labels))
        }

        return jsonify({
            'prediction': predicted_label,
            'probabilities': probability_dict
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)