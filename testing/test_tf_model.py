import os
from utilities.class_names import CAR_TYPE, MODEL_VARIANT
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utilities.tools import suppress_tf_warnings
from utilities.export_helper import export

# Define config
img_height = 300
img_width = 300
img_folder = 'test_images'
model_path = '../models/car_types/best_model/vgg16-pretrained.h5'
export_folder = 'results/'
# Set specific_model_variants to True if you want to test the model with specific Porsche model variants and years.
# Set specific_model_variants to False if you want to test the model with broad Porsche model types.
specific_model_variants = False
# Supress TF warnings
suppress_tf_warnings()
# Load model (If loading a model fails, compile=False and compile it manually
model = keras.models.load_model(model_path)  # , compile=False)
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# Load images
images = []
img_names = []
for image in os.listdir(img_folder):
    img_names.append(image)
    img = tf.keras.utils.load_img(f"{img_folder}/{image}", target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    images.append(img_array)

# Predict
all_predictions = {}
class_names = MODEL_VARIANT if specific_model_variants else CAR_TYPE

for img_array, name in zip(images, img_names):
    predictions = model.predict(img_array)

    for pred in predictions:
        score = tf.nn.softmax(pred)
        print(np.argmax(score))
        print(f"Ground truth: {name} | Predicted: {class_names[np.argmax(score)]} | Confidence: {100 * np.max(score): .2f}%")
        all_predictions[name] = [class_names[np.argmax(score)], 100 * np.max(score)]

# Export predictions to CSV or text file
export(all_predictions, export_folder=export_folder, export_to_csv=False)
