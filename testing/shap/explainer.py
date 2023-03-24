import shap
import numpy as np
from testing.class_names import MODEL_VARIANT, CAR_TYPE
from training.tools import *


# tf.compat.v1.disable_v2_behavior()
suppress_tf_warnings()
# model = keras.models.load_model("../../models/car_types/with_augmentation.h5", compile=False)
model = keras.models.load_model("../../models/model_variants/vgg16-pretrained-model-variants.h5", compile=False)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

img_height = 300
img_width = 300
img_folder = '../test_pic'
# image = '911_1980.jpg'
path_addon = "Porsche"


# Define a function to preprocess the input image
def preprocess_input(image):
    return tf.keras.applications.vgg16.preprocess_input(image)


config = {
    "path": f"C:/Users\phili/.keras/datasets/resized_DVM/{path_addon}",
    "batch_size": 32,
    "img_height": img_height,
    "img_width": img_width,
}

# Load dataset and classes
train_ds, val_ds, class_names = load_dataset(**config)
# Load a small set of images for the SHAP background dataset
# You can use a small sample from your training or validation set


# sample = tf.keras.utils.load_img(f"C:/Users\phili/.keras/datasets/resized_DVM/{path_addon}/911/1990/Green/Porsche$$911$$1990$$Green$$71_4$$1388$$image_1.jpg", target_size=(img_height, img_width))
## background_data = preprocess_input(background_data)
# img_array = tf.keras.utils.img_to_array(sample)
## img_array = tf.expand_dims(img_array, 0)
# background_data = np.expand_dims(img_array, axis=0)

background_data = []
sample = train_ds.shuffle(10000).take(1000)
for img, label in sample.as_numpy_iterator():
    # img = img / 255
    background_data.append(img)

# Load images
images = []
img_names = []
for image in os.listdir(img_folder):
    img_names.append(image)
    img = tf.keras.utils.load_img(f"{img_folder}/{image}", target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create a batch
    images.append(img_array)

explainer = shap.GradientExplainer(model, background_data)  # local_smoothing=0.1
classes = MODEL_VARIANT
with tf.device('/CPU:0'):
    for image, name in zip(images, img_names):
        shap_values, indexes = explainer.shap_values(image, ranked_outputs=3)
        # get the names for the classes
        print(indexes)
        image = image/255
        index_names = np.vectorize(lambda x: classes[x])(indexes)
        shap.image_plot(shap_values, image, index_names, show=False)
        plt.suptitle("SHAP values for " + name)
        plt.show()
