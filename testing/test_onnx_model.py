import os

from testing.class_names import *
from utilities.tools import suppress_tf_warnings
import onnxruntime as ort
import numpy as np
import tensorflow as tf
from export_helper import export

# Supress TF warnings
suppress_tf_warnings()


# Functions to load and run an ONNX model
def load_onnx_model(onnx_model_path):
    session = ort.InferenceSession(onnx_model_path)
    return session


def predict(session, input_data):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Check if the input data is already a numpy array
    if isinstance(input_data, np.ndarray):
        input_feed = {input_name: input_data}
    # If the input data is a list of dictionaries, directly pass it as input_feed
    elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
        input_feed = input_data
    else:
        raise ValueError("Input data must be a list of dictionaries or a single numpy array")

    predictions = session.run([output_name], input_feed)[0]
    return predictions


def preprocess_image_keras(image_path, input_shape):
    img = tf.keras.utils.load_img(image_path, target_size=(input_shape[1], input_shape[2]))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add a new dimension for the batch size
    return img_array


def get_top_class_and_percentage(predictions, class_labels):
    top_class_idx = np.argmax(predictions)
    percentages = tf.nn.softmax(predictions)
    return class_labels[top_class_idx], percentages[0, top_class_idx]


# Prepare inference
img_height = 300
img_width = 300
# Set specific_model_variants to True if you want to test the model with specific Porsche model variants and years.
# Set specific_model_variants to False if you want to test the model with broad Porsche model types.
specific_model_variants = True
model_path = '../models/model_variants/vgg16-pretrained-model-variants.onnx'
img_folder = 'test_pic'

# Load model
session = load_onnx_model(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Input name and shape:", input_name, session.get_inputs()[0].shape)
print("Output name and shape:", output_name, session.get_outputs()[0].shape)

# Load images
images = []
img_names = []
for image in os.listdir('test_pic'):
    img_names.append(image)
    img_array = input_data = preprocess_image_keras(f"{img_folder}/{image}", session.get_inputs()[0].shape)
    images.append(img_array)

# Predict
all_predictions = {}
class_names = MODEL_VARIANT if specific_model_variants else CAR_TYPE
for img_array, name in zip(images, img_names):
    predictions = predict(session, img_array)

    top_class, top_percentage = get_top_class_and_percentage(predictions, MODEL_VARIANT)
    print(f"Ground truth: {name} | Predicted: {top_class} | Confidence: {100 * top_percentage: .2f}%")
    all_predictions[name] = [top_class, 100 * top_percentage]

# Export predictions to CSV or text file
export(all_predictions, export_to_csv=False, export_folder='results/onnx/')
