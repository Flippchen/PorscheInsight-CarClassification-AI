import os
from class_names import FEW_CLASSES, MORE_CLASSES
import numpy as np
import tensorflow as tf
from tensorflow import keras
from training.tools import suppress_tf_warnings
from export_helper import export

# Define config
img_height = 300
img_width = 300
img_folder = 'test_pic'
# Set true if you want to test the model with more classes
more_classes = True
# Supress TF warnings
suppress_tf_warnings()
# Load model
model = keras.models.load_model('../models/more_classes/vgg16-pretrained-more-classes.h5')

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
class_names = MORE_CLASSES if more_classes else FEW_CLASSES

for img_array, name in zip(images, img_names):
    predictions = model.predict(img_array)

    for pred in predictions:
        score = tf.nn.softmax(pred)
        print(f"Ground truth: {name} | Predicted: {class_names[np.argmax(score)]} | Confidence: {100 * np.max(score): .2f}%")
        all_predictions[name] = [class_names[np.argmax(score)], 100 * np.max(score)]

# Export predictions to CSV or text file
export(all_predictions, export_to_csv=False)
