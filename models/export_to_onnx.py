# Description: Convert a TensorFlow SavedModel to ONNX
# Imports
import os
import tensorflow as tf
import tf2onnx
from utilities.tools import suppress_tf_warnings

# Supress TF warnings
suppress_tf_warnings()


# Function to convert a SavedModel to ONNX
def convert_saved_model_to_onnx(saved_model_path, onnx_output_path):
    # Load the SavedModel
    model = tf.keras.models.load_model(saved_model_path)

    # Convert the TensorFlow model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model)

    # Save the ONNX model
    if not os.path.exists(onnx_output_path):
        print('Creating new file')
        with open(onnx_output_path, 'w'): pass
    with open(onnx_output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())


if __name__ == '__main__':
    saved_model_path = 'car_types/best_model/efficientnet-car-type-2_best_model.h5'
    onnx_output_path = 'onnx/car_types/efficientnet-car-type.onnx'
    convert_saved_model_to_onnx(saved_model_path, onnx_output_path)
