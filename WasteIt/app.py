import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

IMG_SIZE = 224
CLASS_NAMES = ["cardboard", "metal", "inorganic", "plastic", "paper", "glass", "organic"]

model = None
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'model_waste.h5')
    
    model = load_model(MODEL_PATH, compile=False) 
    print("Model loaded")
except Exception as e:
    print(f"Error while loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo not available'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400
    
    file = request.files['image']
    try:
        processed_img = prepare_image(file)
        predictions = model.predict(processed_img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        result = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown"
        
        return jsonify({
            'class': result,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port)