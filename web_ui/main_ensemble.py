import io
import sys
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Union

# Needs to be imported before eel to not crash when using --noconsole

# sys.stdout = io.StringIO()
# sys.stderr = io.StringIO()

import eel
import base64
from io import BytesIO
from PIL import Image

from utilities.class_names import get_classes_for_model
from utilities.prepare_images import replace_background, resize_and_pad_image, fix_image, convert_mask
import pooch
from rembg import new_session

# Initiate models
models: dict[str, Union[None, ort.InferenceSession]] = {
    "car_type": None,
    "all_specific_model_variants": None,
    "specific_model_variants": None,
    "pre_filter": None,
    "car_type_2": None,
}

# Initiate session
session: new_session = new_session("u2net")


def load_model(model_name: str) -> ort.InferenceSession | List[ort.InferenceSession]:
    """
    Load a specific model from a set of predefined models.

    This function downloads a model from a remote URL based on the given model name.
    After the model is downloaded, an ONNX Inference Session is initialized with the model
    and the session is returned.

    Args:
        model_name (str): Name of the model to be loaded. Valid model names include 'car_type',
        'all_specific_model_variants', 'specific_model_variants' and 'pre_filter'.

    Returns:
        ort.InferenceSession: The initialized ONNX inference session for the loaded model.
    """

    if model_name == "car_type":
        url_model_1 = "https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases/download/v.0.1/vgg16-pretrained-car-types.onnx"
        md5_model_1 = "7c42a075ab9ca1a2a198e5cd241a06f7"

        url_model_2 = "https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases/download/v.0.1/efficientnet-car-type.onnx"
        md5_model_2 = "69909729a5480bdeaf1929046d76e183"

        # Download and cache the models using Pooch
        model_path_1 = pooch.retrieve(
            url_model_1,
            f"md5:{md5_model_1}",
            fname="car_type_1.onnx",
            progressbar=True,
        )

        model_path_2 = pooch.retrieve(
            url_model_2,
            f"md5:{md5_model_2}",
            fname="car_type_2.onnx",
            progressbar=True,
        )

        print("Models downloaded to: ", model_path_1, model_path_2)

        # Return the inference sessions for both models
        return [ort.InferenceSession(model_path_1, providers=['CPUExecutionProvider']),
                ort.InferenceSession(model_path_2, providers=['CPUExecutionProvider'])]

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
    """
    Prepare image data for prediction.

    This function applies background removal, resizing and padding as required.
    If remove_background is set to True, it uses the U2Net model to remove the image background.
    If show_mask is set to True, it will also return the mask of the removed background.

    Args:
        image_data (Image): Input image data to be processed.
        target_size (Tuple): Target size to resize the input image data.
        remove_background (bool): Flag indicating whether to remove background from image.
        show_mask (bool): Flag indicating whether to show the mask of removed background.

    Returns:
        Tuple[np.ndarray, Image.Image]: A tuple containing processed image data as a numpy array and the mask of the image.
    """

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


def get_top_n_predictions(prediction: np.ndarray, model_name: str, n: int = 3) -> List[Tuple[str, float]]:
    """
    Get top n predictions from the model output.

    Args:
        prediction (np.ndarray): Output prediction from a model.
        model_name (str): Name of the model that produced the prediction.
        n (int, optional): Number of top predictions to retrieve. Defaults to 3.

    Returns:
        List[Tuple[str, float]]: A list of top n predictions along with their respective scores.
    """

    # Ensure that n does not exceed the total number of classes
    n = min(n, prediction[0].shape[0])

    top_n_indices = prediction[0].argsort()[-n:][::-1]
    classes = get_classes_for_model(model_name)
    top_n_predictions = [(classes[i], round(prediction[0][i] * 100, 2)) for i in top_n_indices]

    return top_n_predictions


def get_pre_filter_prediction(image_data: np.ndarray, model_name: str):
    """
    Get pre-filter prediction results from the model.

    This function loads a pre-filter model if it has not been loaded already.
    Then it runs this model on the given image data and returns top 3 prediction results.

    Args:
        image_data (np.ndarray): Image data to run the model on.
        model_name (str): Name of the pre-filter model.

    Returns:
        filter_names: Top 3 prediction results from the pre-filter model.
    """

    if models[model_name] is None:
        models[model_name] = load_model(model_name)
    input_name = models[model_name].get_inputs()[0].name
    prediction = models[model_name].run(None, {input_name: image_data})
    filter_names = get_top_n_predictions(prediction[0], "pre_filter")
    return filter_names


def ensemble_predictions_weighted(models, filter_image):
    assert len(models) == 2, "This function is designed for two models only."

    # make predictions
    yhats = [model.run(None, {model.get_inputs()[0].name: filter_image}) for model in models]
    yhats = np.array(yhats)
    # weighted average across ensemble members
    weighted_avg = 0.35 * yhats[0] + 0.65 * yhats[1]
    return weighted_avg


@eel.expose
def classify_image(image_data: str, model_name: str, show_mask: bool = False) -> Tuple[List[Tuple[str, float]], str] | List[List[Tuple[str, float]]]:
    """
    Classify an image using a specified model.

    This function loads the specified model if it has not been loaded already.
    Then it decodes the base64 image data, fixes the image orientation and color,
    and prepares the image for processing and prediction. Depending on the specified
    show_mask option, it also returns the mask of the processed image.

    Args:
        image_data (str): Base64 encoded image data.
        model_name (str): Name of the model to use for classification.
        show_mask (str): Flag indicating whether to show the mask of processed image. Default is "no".

    Returns:
        Tuple[List[Tuple[str, float]], str] | List[List[Tuple[str, float]]]: If show_mask is "yes",
        it returns a tuple containing the top 3 predictions along with their scores and the mask
        of the processed image as base64 encoded string. If show_mask is "no", it returns only
        the top 3 predictions.
    """

    # Loading the model if it's not already loaded
    model = models.get(model_name)
    if not model:
        model = load_model(model_name)
        models[model_name] = load_model(model_name)

    # Decoding the base64 image and fix image orientation/color
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = fix_image(image)

    # Retrieving the required input size for the specified model
    input_size = model.get_inputs()[0].shape[1:3] if model_name != "car_type" else model[0].get_inputs()[0].shape[1:3]

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
    pre_filter_predictions = get_pre_filter_prediction(filter_image, "pre_filter")

    # If the pre-filter model doesn't predict a Porsche, we can skip the specific model
    if pre_filter_predictions[0][0] != "porsche":
        return (pre_filter_predictions, mask_base64) if show_mask else [pre_filter_predictions]

    if model_name != "car_type" and model_name != "specific_model_variants":
        input_name = model.get_inputs()[0].name
        prediction = model.run(None, {input_name: filter_image})
    elif model_name == "specific_model_variants":
        if models["car_type"] is None:
            models["car_type"] = load_model("car_type")

        pre_prediction = ensemble_predictions_weighted(models["car_type"], filter_image)
        top_pre_prediction = get_top_n_predictions(pre_prediction[0], "car_type")[0][0]

        input_name = model.get_inputs()[0].name
        prediction = model.run(None, {input_name: filter_image})

        # Filter the second model's predictions based on the top result of the first model
        classes = get_classes_for_model("specific_model_variants")

        print(top_pre_prediction)
        valid_indices = [i for i, class_name in enumerate(classes) if class_name.startswith(top_pre_prediction)]
        filtered_predictions = [prediction[0][i] for i in valid_indices]

        # Get top 3 predictions after filtering
        top_3_predictions = sorted(zip(filtered_predictions, valid_indices), key=lambda x: x[0], reverse=True)[:3]
        top_3_predictions = [(classes[i], round(score * 100, 2)) for score, i in top_3_predictions]

        return (top_3_predictions, mask_base64) if show_mask else [top_3_predictions]
    else:
        prediction = ensemble_predictions_weighted(model, filter_image)

    # Retrieving the top 3 predictions
    top_3_predictions = get_top_n_predictions(prediction[0], model_name)

    return (top_3_predictions, mask_base64) if show_mask else [top_3_predictions]


eel.init("web")
eel.start("index.html", size=(1000, 800), mode="default")
