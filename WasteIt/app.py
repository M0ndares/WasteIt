import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
# Importamos esto SOLO para que load_model no falle al leer el archivo .h5
from tensorflow.keras.applications.resnet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model_waste.h5' # Asegúrate que el nombre del archivo sea correcto aquí
IMG_SIZE = 224

# Usamos la lista de clases EXACTAMENTE como la tienes en tu script que funciona
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
    # Cargamos el modelo pasando 'preprocess_input' en custom_objects 
    # para que Keras reconozca la capa interna, igual que en tu script.
    model = load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input})
    print("¡Modelo listo!")
except Exception as e:
    print(f"Error cargando el modelo: {e}")

def prepare_image(file_stream):
    """
    Replica EXACTAMENTE la función load_and_preprocess_image de tu script funcional.
    """
    # 1. Leer imagen desde la memoria (equivalente a cv.imread)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 2. Resize + Padding (Tu lógica exacta de square_resize_image)
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

    # 3. Convertir a RGB (OpenCV carga en BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. NO ESCALAMOS MANUALMENTE (La clave del éxito)
    # Solo convertimos a float32 manteniendo valores 0-255
    img = img.astype('float32')

    # 5. Expandir dimensiones
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400
    
    file = request.files['image']
    
    try:
        # Procesar imagen
        processed_img = prepare_image(file)
        
        # Predecir
        predictions = model.predict(processed_img, verbose=0)
        
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        # Obtener nombre de la clase
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