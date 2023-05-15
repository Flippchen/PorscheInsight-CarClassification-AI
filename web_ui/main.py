import io
import sys
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
# Needs to be imported before eel to not crash when using --noconsole

sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

import eel
import base64
from io import BytesIO
from PIL import Image

from utilities.class_names import get_classes_for_model
from utilities.prepare_images import replace_background, resize_and_pad_image, fix_image, convert_mask
import pooch
from rembg import new_session

# Initiate models
models = {
    "car_type": None,
    "all_specific_model_variants": None,
    "specific_model_variants": None,
    "pre_filter": None,
}

session = new_session("u2net")


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


def prepare_image(image_data: Image, target_size: Tuple, remove_background: bool, show_mask: bool) -> Tuple[np.ndarray, Image.Image]:
    if remove_background and show_mask:
        image, mask = replace_background(image_data, session=session)
    elif remove_background:
        image, _ = replace_background(image_data, session=session)
        mask = None
    else:
        image = resize_and_pad_image(image_data, target_size)
        mask = None

    img_array = np.array(image).astype('float32')
    img_array = np.expand_dims(img_array, 0)

    if mask is None:
        mask = Image.fromarray(np.zeros((1, 1), dtype=np.uint8))
        mask = mask.convert("RGBA")

    return img_array, mask


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
    filter_names = get_top_3_predictions(prediction[0], "pre_filter")
    return filter_names


@eel.expose
def classify_image(image_data: str, model_name: str, show_mask: str = "no") -> tuple[list[tuple[str, float]], str] | list[list[tuple[str, float]]]:
    # Loading the model if it's not already loaded
    if models[model_name] is None:
        models[model_name] = load_model(model_name)

    # Decoding the base64 image data and opening the image
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))
    show_mask = True if show_mask == "yes" else False

    # Correcting image orientation and color mode if necessary
    image = fix_image(image)

    # Retrieving the required input size for the specified model
    input_size = models[model_name].get_inputs()[0].shape[1:3]

    # Preparing image for processing and prediction
    # FIXME: Currently, the background is removed prior to prediction. This is a workaround,
    # as predictions seem to be better with a black background.
    filter_image, mask = prepare_image(image, input_size, remove_background=True, show_mask=show_mask)

    # Converting the mask for processing
    mask = convert_mask(mask)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Getting initial predictions before applying filters
    filter_predictions = get_pre_filter_prediction(filter_image, "pre_filter")

    # If the initial prediction is 'porsche' or other car brands, run the specified model's prediction
    if filter_predictions[0][0] == "porsche":
        input_name = models[model_name].get_inputs()[0].name
        prediction = models[model_name].run(None, {input_name: filter_image})

        # Retrieving the top 3 predictions
        top_3 = get_top_3_predictions(prediction[0], model_name)

        if show_mask:
            return top_3, mask_base64
        else:
            return [top_3]

    # Returning the initial predictions and an optional mask if show_mask is set to 'yes'
    if show_mask:
        return filter_predictions, mask_base64
    return [filter_predictions]


eel.init("web")
eel.start("index.html", size=(1000, 800), mode="default")
