import numpy as np
import onnxruntime as ort
from typing import List, Tuple
# Needs to be imported before eel to not crash when using --noconsole
import sys, io

sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

import eel
import base64
from io import BytesIO
from PIL import Image
from utilities.class_names import get_classes_for_model
from testing.prepare_images import replace_background, resize_and_pad_image
import pooch
from rembg import new_session

# Initiate models
models = {
    "car_type": None,
    "all_specific_model_variants": None,
    "specific_model_variants": None,
    "pre_filter": None,
}

session = new_session("isnet-general-use")


def load_model(model_name: str) -> ort.InferenceSession:
    if model_name == "car_type":
        url = "https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases/download/v.0.1/vgg16-pretrained-car-types.onnx"
        md5 = "7c42a075ab9ca1a2a198e5cd241a06f7"
    elif model_name == "all_specific_model_variants":
        url = "https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases/download/v.0.1/efficientnet-old-head-all-model-variants-full_best_model.onnx"
        md5 = "c54797cf92974c9ec962842e7ecd515c"
    elif model_name == "specific_model_variants":
        url = "https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases/download/v.0.1/efficientnet-model-variants_best_model.onnx"
        md5 = "3de16b8cf529dc90f66c962a1c93a904"
    elif model_name == "pre_filter":
        url = "https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases/download/v.0.1/efficientnet-pre-filter_best_model.onnx"
        md5 = "b70e531f5545afc66551c58f85d6694a"
    else:
        raise ValueError("Invalid model name")

    # Show the loading notification
    eel.showLoading()

    # Download and cache the model using Pooch
    model_path = pooch.retrieve(
        url,
        f"md5:{md5}",
        fname=model_name + ".onnx",
        progressbar=True,
    )
    print("Model downloaded to: ", model_path)
    # Hide the loading notification
    eel.hideLoading()

    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


def prepare_image(image_data: Image, target_size: Tuple, remove_background: bool) -> np.ndarray:
    image = resize_and_pad_image(image_data, target_size)
    if remove_background:
        image = replace_background(image, session=session)
    img_array = np.array(image).astype('float32')
    img_array = np.expand_dims(img_array, 0)
    return img_array


def get_top_3_predictions(prediction: np.ndarray, model_name: str) -> List[Tuple[str, float]]:
    top_3 = prediction[0].argsort()[-3:][::-1]
    classes = get_classes_for_model(model_name)
    top_3 = [(classes[i], round(prediction[0][i] * 100, 2)) for i in top_3]
    return top_3


def get_pre_filter_prediction(image_data: np.ndarray, model_name: str):
    if models[model_name] is None:
        models[model_name] = load_model(model_name)
    input_name = models[model_name].get_inputs()[0].name
    prediction = models[model_name].run(None, {input_name: image_data})
    fitler_names = get_top_3_predictions(prediction[0], "pre_filter")
    return fitler_names


@eel.expose
def classify_image(image_data: str, model_name: str) -> List[Tuple[str, float]]:
    # Load model if not loaded yet
    if models[model_name] is None:
        models[model_name] = load_model(model_name)

    # Decode image and open it
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))

    # Convert image to RGB if not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Get correct input size for model
    input_size = models[model_name].get_inputs()[0].shape[1:3]

    # Prepare image for filtering and predict
    filter_image = prepare_image(image, input_size, remove_background=False)
    filter_predictions = get_pre_filter_prediction(filter_image, "pre_filter")

    # If the pre_filter predicts porsche or other_car_brand, predict the correct model
    # FIXME: If image contains a porsche the prediction can be false negative because the background was not removed
    if filter_predictions[0][0] == "porsche":
        prepared_image = prepare_image(image, input_size, remove_background=True)
        input_name = models[model_name].get_inputs()[0].name
        prediction = models[model_name].run(None, {input_name: prepared_image})
        # Get top 3 predictions
        top_3 = get_top_3_predictions(prediction[0], model_name)

        return top_3

    return filter_predictions


eel.init("web")
eel.start("index.html", size=(1000, 800), mode="default")
