import numpy as np
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
import onnxruntime as ort
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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
num_classes = 10

meta_model = Sequential()
meta_model.add(Dense(128, input_dim=len(models) * 10, activation='relu'))
meta_model.add(Dropout(0.5))
meta_model.add(Dense(64, activation='relu'))
meta_model.add(Dropout(0.5))
meta_model.add(Dense(num_classes, activation='softmax'))
meta_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam',
                   metrics=['accuracy'])

# Train your meta-learner
history = meta_model.fit(ensemble_predictions_train, all_labels, epochs=100)


# Define Tree/Regression Model
meta_model_tree = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
meta_model_regression = LogisticRegression(solver='liblinear')
meta_model_tree.fit(ensemble_predictions_train, all_labels.ravel())

# Train Tree/Regression Model
meta_model_tree.fit(ensemble_predictions_train, all_labels.ravel())
meta_model_regression.fit(ensemble_predictions_train, all_labels.ravel())
# TODO: Try a Regression/Tree Based Model

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

plt.tight_layout()
plt.show()
