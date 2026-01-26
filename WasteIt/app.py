import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

IMG_SIZE = 224
CLASS_NAMES = [
    "cardboard", "metal", "inorganic", "plastic", 
    "paper", "glass", "organic", "battery"
]

# --- BLOQUE DE CARGA DE MODELO BLINDADO ---
model = None
load_error = None
model_path_used = "Desconocido"

try:
    # 1. Buscamos la ruta exacta donde está este archivo app.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 2. Construimos la ruta al modelo
    MODEL_PATH = os.path.join(BASE_DIR, 'model_waste.h5')
    model_path_used = MODEL_PATH
    
    print(f"📂 Intentando cargar modelo desde: {MODEL_PATH}")
    
    # 3. Cargamos
    model = load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input})
    print("✅ ¡Modelo cargado exitosamente!")

except Exception as e:
    # Si falla, guardamos el error en una variable para mostrarlo después
    load_error = str(e)
    print(f"❌ ERROR FATAL CARGANDO MODELO: {e}")

def prepare_image(file_stream):
    # Lectura de imagen robusta
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
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    # --- DIAGNÓSTICO DE ERROR ---
    # Si el modelo no cargó, le decimos al usuario POR QUÉ
    if model is None:
        return jsonify({
            'error': 'El modelo no se pudo cargar al iniciar el servidor.',
            'razon_del_error': load_error,
            'ruta_intentada': model_path_used
        }), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400
    
    file = request.files['image']
    
    try:
        processed_img = prepare_image(file)
        predictions = model.predict(processed_img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        if class_idx < len(CLASS_NAMES):
            result = CLASS_NAMES[class_idx]
        else:
            result = "Desconocido"
        
        return jsonify({
            'class': result,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': f"Error procesando imagen: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)