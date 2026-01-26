import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model_waste.h5' 
IMG_SIZE = 224
CLASS_NAMES = [
    "cardboard",
    "metal",
    "inorganic",
    "plastic",
    "paper",
    "glass",
    "organic",
    "battery"
  ]

print("Cargando modelo... espera un momento...")
try:
    model = load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input})
    print("¡Modelo listo!")
except Exception as e:
    print(f"Error cargando el modelo: {e}")

def prepare_image(file_stream):
    """
    Replica EXACTAMENTE la función load_and_preprocess_image de tu script funcional.
    """
    # 1. Read image
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 2. Resize + Padding 
    h, w = img.shape[:2]
    top, bottom, left, right = 0, 0, 0, 0
    if w >= h:
        top = (w - h) // 2
        bottom = (w - h) - top
    else:
        left = (h - w) // 2
        right = (h - w) - left
    
    if any([top, bottom, left, right]):
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 3. BGR -> RGB 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400
    
    file = request.files['image']
    
    try:
        processed_img = prepare_image(file)
        predictions = model.predict(processed_img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        # Name x Index
        if class_idx < len(CLASS_NAMES):
            result = CLASS_NAMES[class_idx]
        else:
            result = "Desconocido"
        
        print(f"Predicción: {result} ({confidence:.2f}%)")

        return jsonify({
            'class': result,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)