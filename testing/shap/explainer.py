import shap
import numpy as np
import keras
from training.tools import suppress_tf_warnings
import tensorflow as tf
from testing.class_names import CAR_TYPE,MODEL_VARIANT
suppress_tf_warnings()


# Define a function to preprocess the input image
def preprocess_input(image):
    return tf.keras.applications.vgg16.preprocess_input(image)


background_data = ...  # Replace this with your actual dataset
background_data = preprocess_input(background_data)

model = keras.models.load_model("../../models/model_variants/vgg16-pretrained-model-variants.h5")
# Create a wrapper function for the model to handle preprocessing
def wrapped_model(images):
    images = preprocess_input(images)
    return model(images)


# Initialize the SHAP explainer
explainer = shap.DeepExplainer(wrapped_model, background_data)

# Choose a specific image to explain
image_to_explain = ...  # Replace this with an image from your dataset
image_to_explain = np.expand_dims(image_to_explain, axis=0)

# Generate SHAP values for the selected image
shap_values = explainer.shap_values(image_to_explain)

# Plot the SHAP values
shap.image_plot(shap_values, image_to_explain, class_names=MODEL_VARIANT)

## create a Background dataset (random data with the same shape as your input data)
# background = np.random.rand(10, 300, 300, 3)
#
## create a Test dataset (one specific example you want to explain)
# test = np.random.rand(1, 300, 300, 3)
#
## load your Keras model
# model_path = "../../models/car_types/without-augmentation.h5"
# model = keras.models.load_model(model_path)
#
## create a SHAP explainer object
# explainer = shap.DeepExplainer(model, background)
#
## generate SHAP values for your Test dataset
# shap_values = explainer.shap_values(test)
#
## visualize the SHAP values for your Test dataset
# shap.image_plot(shap_values, test)
# Initialize the SHAP explainer
# shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
#explainer = shap.GradientExplainer(model, background_data)
#img = tf.keras.utils.load_img(f"{img_folder}/{image}", target_size=(img_height, img_width))
#img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create a batch
# img, label in train_ds.shuffle(1000).take(2).as_numpy_iterator():

#image_to_explain = np.expand_dims(img, axis=0)
# Generate SHAP values for the selected image
#shap_values = explainer.shap_values(image_to_explain)
# Plot the SHAP values
#shap.image_plot(shap_values, image_to_explain)