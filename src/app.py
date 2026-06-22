import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
from keras.models import load_model
import threading 
import gc         

app = Flask(__name__)
CORS(app)

IMG_SIZE = 224

CLASS_NAMES = [
    "cardboard", "metal", "inorganic", "plastic", 
    "paper", "glass", "organic", "battery"
]

model = None
load_error = None
lock = threading.Lock() # Candado para asegurar la predicción segura entre hilos

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'model.h5') 
    
    # Se remueve el custom_preprocess de la carga si el modelo original no lo usaba
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    load_error = str(e)

def prepare_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    
    # CORRECCIÓN 2: Eliminamos custom_preprocess(img) porque el código original NO lo usaba.
    # Esto asegura que los píxeles vayan de 0 a 255 tal como el modelo los espera.
    
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model could not be loaded.', 'Details': load_error}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400
    
    file = request.files['image']
    processed_img = None
    
    try:
        processed_img = prepare_image(file)
        
        # Sincronización segura para evitar colisiones en Flask
        with lock:
            predictions = model.predict(processed_img, verbose=0)
            
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        result = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown"
        
        return jsonify({
            'class': result,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': f"Error: {str(e)}"}), 500
    finally:
        # CORRECCIÓN 3: Limpieza segura. Borramos la variable, pero dejamos que Python 
        # maneje el ciclo de vida de manera natural en lugar de forzar gc.collect() 
        # de forma inmediata y violenta, previniendo caídas en el renderizado.
        if processed_img is not None:
            del processed_img

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)