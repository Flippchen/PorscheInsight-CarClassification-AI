from typing import List, Tuple

import eel
import base64
from io import BytesIO
from PIL import Image
import keras
from utilities.tools import suppress_tf_warnings, get_classes_for_model
from testing.prepare_images import replace_background
import tensorflow as tf

suppress_tf_warnings()


# Load your Keras models
models = {
    "car_type": keras.models.load_model("../models/car_types/best_model/vgg16-pretrained.h5"),
    "all_specific_model_variants": keras.models.load_model("../models/model_variants/best_model/efficientnet-old-head-model-variants-full_best_model.h5"),
}


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
    # Decode image and open it
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))
    # Prepare image and predict
    prepared_image = prepare_image(image, models[model_name].input_shape[1:3])
    prediction = models[model_name].predict(prepared_image)
    # Get top 3 predictions
    top_3 = get_top_3_predictions(prediction, model_name)

    return top_3


eel.init("web")
eel.start("index.html", size=(1000, 800))
