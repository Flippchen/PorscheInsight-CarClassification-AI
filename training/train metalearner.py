import numpy as np
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
import onnxruntime as ort
import tensorflow as tf
import matplotlib.pyplot as plt

from utilities.tools import load_image_subset, get_data_path_addon

# Set model Type to 'all_specific_model_variants' or 'car_type' or "specific_model_variants"
model_type = 'car_type'
name = "car_type_meta"
path_addon = get_data_path_addon(model_type)
img_height = 300
img_width = 300
config = {
    "path": f"C:/Users\phili/.keras/datasets/resized_DVM/{path_addon}",
    "batch_size": 1,
    "img_height": img_height,
    "img_width": img_width,
}

# Load models
model_paths = ["../models/onnx/car_types/efficientnet-car-type.onnx",
               "../models/onnx/car_types/vgg16-pretrained-car-types.onnx"]
models = [ort.InferenceSession(model_path, providers=['CPUExecutionProvider']) for model_path in model_paths]


def ensemble_predictions(models, filter_images):
    # Initialize a list to hold all predictions
    all_predictions = []

    # Iterate over each image in the batch
    for image in filter_images:
        # Make predictions with each model and store it
        yhats = [np.squeeze(model.run(None, {model.get_inputs()[0].name: image})) for model in models]
        # Append the predictions to the list
        all_predictions.append(yhats)

    # Stack the list along a new axis to create a 2D numpy array
    return np.stack(all_predictions, axis=0)


# Load 500 training images
dataset = load_image_subset(**config, shuffle=1000, number_images=50)
# Extract images and labels from the dataset
all_images = []
all_labels = []
for images, labels in dataset:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())

all_labels = np.array(all_labels)
# Get the predictions of your models on the training data
ensemble_predictions_train = ensemble_predictions(models, all_images)

# Reshape the predictions to be 2D
ensemble_predictions_train = ensemble_predictions_train.reshape(-1, len(models) * 10)

# Define your meta-learner
meta_model = Sequential()
meta_model.add(Dense(10, input_dim=len(models), activation='relu'))
meta_model.add(Dense(1))

# Compile your meta-learner
meta_model.compile(loss='mean_squared_error', optimizer='adam')

# Train your meta-learner
meta_model.fit(ensemble_predictions_train, all_labels, epochs=10)
