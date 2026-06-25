import tensorflow as tf
import keras
from keras.applications.resnet_v2 import preprocess_input

@keras.saving.register_keras_serializable()
def custom_preprocess(x):
    return preprocess_input(x)

custom_dict = {
    'preprocess_input': custom_preprocess,
    'function': custom_preprocess
}

model = keras.models.load_model('model.h5', custom_objects=custom_dict, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

