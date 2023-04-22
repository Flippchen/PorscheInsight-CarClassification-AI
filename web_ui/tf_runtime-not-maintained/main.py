from typing import List, Tuple

import eel
import base64
from io import BytesIO
from PIL import Image
import keras
from utilities.tools import suppress_tf_warnings
from utilities.class_names import get_classes_for_model
from testing.prepare_images import replace_background
import tensorflow as tf
import pooch

suppress_tf_warnings()

# Load local Keras models
# models = {
#    "car_type": keras.models.load_model("../models/car_types/best_model/vgg16-pretrained.h5"),
#    "all_specific_model_variants": keras.models.load_model("../models/all_model_variants/best_model/efficientnet-old-head-model-variants-full_best_model.h5"),
# }

# Initiate models
models = {
    "car_type": None,
    "all_specific_model_variants": None,
    "specific_model_variants": None,
}


def load_model(model_name: str) -> keras.Model:
    if model_name == "car_type":
        url = "https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases/download/v.0.1/vgg16-pretrained-car-types.h5"
        md5 = "e7c79ac2d2855e7a65e2ec728fe1d178"
    elif model_name == "all_specific_model_variants":
        url = "https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases/download/v.0.1/efficientnet-old-head-model-all-variants-full_best_model.h5"
        md5 = "564a7d21468c6de78d7ac7a8b7896a28"
    elif model_name == "specific_model_variants":
        url = "https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases/download/v.0.1/efficientnet-model-variants_best_model.h5"
        md5 = "ead5b6ca6a89bb2b6df4dd1fc4f6a583"
    else:
        raise ValueError("invalid Model name")

    # Show the loading notification
    eel.showLoading()

    # Download and cache the model using Pooch
    model_path = pooch.retrieve(
        url,
        f"md5:{md5}",
        fname=model_name + ".h5",
        progressbar=True,
    )
    print("Model downloaded to: ", model_path)
    # Hide the loading notification
    eel.hideLoading()

    return keras.models.load_model(model_path)


def prepare_image(image_data: Image, target_size: Tuple):
    image = image_data.resize(target_size)
    image = replace_background(image)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def get_top_3_predictions(prediction: List, model_name: str) -> List[Tuple[str, float]]:
    top_3 = prediction[0].argsort()[-3:][::-1]
    classes = get_classes_for_model(model_name)
    top_3 = [(classes[i], round(prediction[0][i] * 100, 2)) for i in top_3]
    return top_3


@eel.expose
def classify_image(image_data: str, model_name: str) -> List[Tuple[str, float]]:
    if models[model_name] is None:
        models[model_name] = load_model(model_name)
    # Decode image and open it
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))
    # Prepare image and predict
    prepared_image = prepare_image(image, models[model_name].input_shape[1:3])
    prediction = models[model_name].predict(prepared_image)
    # Get top 3 predictions
    top_3 = get_top_3_predictions(prediction, model_name)

    return top_3


eel.init("../web")
eel.start("index.html", size=(1000, 800))
