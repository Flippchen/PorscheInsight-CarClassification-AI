# PorscheInsight-CarClassification-AI
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Flippchen/PorscheInsight-CarClassification-AI?include_prereleases&style=flat-square) ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Flippchen/PorscheInsight-CarClassification-AI/python.yaml?logoColor=blue&style=flat-square) ![GitHub repo size](https://img.shields.io/github/repo-size/Flippchen/PorscheInsight-CarClassification-AI?style=flat-square)

<a href='https://play.google.com/store/apps/details?id=com.flippchen.porsche_classifier'><img alt='Get it on Google Play' src='https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png' height="70"/></a>
## Description
This repository contains scripts to train models to classify pictures of Porsche cars.
It is not ment to be used in production (yet).

## Web UI

The Web UI is a simple local website that enables users to upload images of Porsche cars and get classification results from the models. The app is built using Eel, which allows Python and HTML/JS to communicate with each other.
For a free online version of the Web UI, check out [PorscheInsight](https://classify.autos).

Using the Web UI, non-car images led to random predictions. I trained a model to classify Porsche, other car brands, and others, resulting in a two-step [Web UI](web_ui) architecture for Porsche identification and classification.
### Installation
```bash
pip install -m build_requirements.txt
```
### Usage
```bash
python web_app/main.py
```
or download the [executable](https://github.com/Flippchen/PorscheInsight-CarClassification-AI/actions).

### Screenshot
<img alt="Screenshot of the Web UI" src="assets/web_app/example_web_ui.png"  width="600" height="400">

### Architecture
The Web UI employs a two-step process involving two models. Initially, the pre_filter model determines if an image contains a Porsche. If a Porsche is detected, the image proceeds to the second model, which classifies the car according to the user's input.

<img alt="Architecture of the Web UI" src="assets/architecture.png"  height="400">

### ToDos
- [ ] Add release 1.0.0
- [ ] Add docker support
- [ ] Evaluate feature engineering/ More data augmentation
- [ ] Add Taycans to images/models

<details>
<summary>Completed Tasks</summary>

- [x] Add ONNX export
- [x] Implement better Testing
- [x] Implement shap for feature importance
- [x] Add confusion matrix
- [x] Try a deeper/wider or different pretrained model to improve accuracy on the more classes dataset
- [x] Add a (web) app to use the models
- [X] Train on cleaned classes
- [x] Add onnx models to web UI to speedup and reduce size
- [x] Isnet model for bg remove
- [x] Bundle in one Binary
- [x] Add django web app
- [x] Train on cleaned classes with Vision Transformer
- [x] Add Android App
- [x] Implement new Architecture: One model classifies if a car is present and a second model classifies the car
- [x] Implement new architecture to online version
</details>


## Installation
### Requirements
- Python >3.8, <3.11
- [optional] CUDA 11.1
- [optional] cuDNN 8.0.5

Install tensorflow, keras and the other dependencies with pip:
```bash
pip install -m requirements.txt
```
## Models
The first version of the model was trained to predict 10 classes, which correspond to broad Porsche car model types. These classes include popular models such as the 911, Cayman, and Panamera, among others. The accuracy of this model on the training set was 99%, and the accuracy on the validation set was 95%.

After achieving satisfactory results with the 10-class model, a second model was trained to predict 88 classes, which correspond to specific Porsche build years. For example, this model can predict whether an image is a 911 from 2008. The accuracy of this model on the training set was 80%, and the accuracy on the validation set was 46%.

For the third model I bundled several years together to imitate the Porsche car series like the 911 991 or 911 992. The model was trained to predict 30 classes, the accuracy on the validation set was 85%.

The fourth model was trained to predict 3 classes (porsche, other_car_brand and other). The model is used for the new architecture in the [web_app](web_ui).

| Model                                | Total params  | Trainable params | Non-trainable params  | Batch size | Accuracy Train % | Accuracy Val % | Number of classes |
|--------------------------------------|---------------|------------------|-----------------------|------------|------------------|----------------|-------------------|
| without augmentation*                | 11,239,850    | 11,239,850       | 0                     | 32         | 98               | 78             | 10                |
| with augmentation*                   | 11,239,850    | 11,239,850       | 0                     | 32         | 79               | 74             | 10                |
| old_pretrained*                      | 20,027,082    | 5,311,114        | 14,715,968            | 32         | 74               | 72             | 10                |
| VGG16 pretrained*                    | 20,027,082    | 12,390,538       | 7,636,544             | 32         | 99               | 95             | 10                |
| VGG16 pretrained                     | 20,027,082    | 12,390,538       | 7,636,544             | 32         | 80               | 46             | 88                |
| efficientnetv2-b1(new head & faster) | 7,106,956     | 993,416          | 6,113,640             | 32         | 47               | 46             | 88                |
| efficientnetv2-b1                    | 7,099,474     | 1985,934         | 6,113,540             | 32         | 49               | 46             | 88                |
| efficientnetv2-b1 (cleaned classes)  | 7,099,474     | 985,934          | 6,113,540             | 32         | 82               | 85             | 30                |
| vit_b16 (cleaned classes)            | 85,901,470    | 102,558          | 85,798,912            | 32         | 45               | 49             | 30                |
| efficientnetv2-b1-pre-filter         | 7,095,991     | 982,451          | 6,113,540             | 32         | 98               | 99             | 3                 |

The models with * were trained on the pre cleaned dataset.

Have a look at the [releases](https://github.com/Flippchen/PorscheInsight-CarClassification-AI/releases) to see the models and [results](models/car_types/results) folder to see the results.

## Usage
### Train a model
To train a model you can use the [train](training) folder. You can choose the model, the dataset and the number of epochs.
You can use the build in Discord Callback to get notfications on Discord after every epoch. You need to change the discord webhook url in the training file.
### Predict with a model (Inference)
To predict with a model you can use the [test_tf_model.py](testing/test_tf_model.py) script. You can choose the model and the image you want to predict.
If you want to predict with an onnx model you can use the [test_onnx_model.py](testing/test_onnx_model.py) script.

I recommend to prepare the images with [prepare_images.py](testing/prepare_images.py) before. Thus, an error-free and improved prediction is guaranteed.
### Explain a model
To explain a model you can use the [explainer.py](model_insights/shap/explainer.py) script. You can choose the model and the image(folder) you want to get explanations.
I recommend to prepare the images with [prepare_images.py](testing/prepare_images.py) before.

After using shap values on the new efficientnet model and the vgg16 model, both on the old head, I found out that the vgg16 model found "better" spots to distinguish between classes, at least sometimes.

### Confusion matrix of a model
The confusion matrix shows the performance of a classification model. It shows the number of correct and incorrect predictions made by a classifier.

To get a confusion matrix of a model you can use the [confusion_matrix.py](model_insights/confusion_matrix/confusion_matrix.py) script. You can choose the model and the test set you want to get the confusion matrix from.
<details>
<summary><b>Confusion Matrix</b></summary>

<img alt="Confusion matrix for cat types" src="model_insights/confusion_matrix/results/cm_car_type.png"  width="700" height="700">

</details>

The confusion matrix of the specific model variants is too big to show it here. You can find it in the [results](model_insights/confusion_matrix/results) folder.

### Sample images from the shap values
<details>
<summary><b>Explained images</b></summary>


<img alt="Shap values for 911_1980" src="model_insights/shap/results/car_types/shap_values_911_1980.png"  width="700" height="280">
<img alt="Shap values for Cayman_2009" src="model_insights/shap/results/all_model_variants/shap_values_Cayman_2009.png"  width="700" height="280">
</details>


### Convert a model to ONNX
You can use the [convert_to_onnx.py](models/export_to_onnx.py) script to convert a keras(.h5) model to ONNX. You can choose the model you want to convert and a save path.
# Dataset
The dataset is from [Github](https://github.com/Flippchen/porsche-pictures) and contains ~32.000 pictures of potential Porsche cars.
Since the source of the dataset is public the quality and the arrangement of the images was also not great.
After the data was cleaned, there are ~30.300 pictures left. Several pictures were removed because they were not of a Porsche car or the picture was not clear enough.

<details>
<summary><b>Have a look at the data:</b></summary>

<img alt="Sample images from Dataset" src="models/car_types/results/sample_images.png"  width="700" height="700">
</details>

For the training of the <b>pre_filter</b> model a mixture of the [porsche-pictures](https://github.com/Flippchen/porsche-pictures) dataset, other Open Source datasets like [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) were used.