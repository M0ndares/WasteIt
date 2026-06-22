import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import gc  # Mantenemos el recolector para asegurar que Render libere RAM inmediatamente
from tensorflow.lite.python.interpreter import Interpreter as tflite

app = Flask(__name__)
CORS(app)

IMG_SIZE = 224
CLASS_NAMES = [
    "cardboard", "metal", "inorganic", "plastic", 
    "paper", "glass", "organic", "battery"
]

# Inicialización de variables para TF Lite
interpreter = None
input_details = None
output_details = None
load_error = None
lock = threading.Lock()

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'model.tflite')  # <-- Apuntamos al nuevo archivo
    
    # Cargar el motor ultraligero de TF Lite
    interpreter = tflite(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Guardar las referencias de los túneles de entrada y salida de datos
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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
    
    # Importante: Como la normalización de ResNet ya se grabó dentro del .tflite,
    # solo pasamos la matriz limpia (0-255).
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'error': 'TF Lite Interpreter could not be loaded.', 'Details': load_error}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400
    
    file = request.files['image']
    processed_img = None
    
    try:
        processed_img = prepare_image(file)
        
        # Flujo de ejecución seguro con hilos (Threading Lock) para TF Lite
        with lock:
            # 1. Colocar la imagen en la ranura de entrada del intérprete
            interpreter.set_tensor(input_details[0]['index'], processed_img)
            
            # 2. Ejecutar la predicción matemática express
            interpreter.invoke()
            
            # 3. Extraer el resultado desde la ranura de salida
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        result = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown"
        
        return jsonify({
            'class': result,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': f"Error durante la inferencia: {str(e)}"}), 500
    finally:
        if processed_img is not None:
            del processed_img
        gc.collect() # Obligamos a limpiar cualquier residuo en la RAM de Render

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)