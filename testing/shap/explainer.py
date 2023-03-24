import shap
import numpy as np
from testing.class_names import MODEL_VARIANT, CAR_TYPE
from training.tools import *

# Load model
suppress_tf_warnings()

# Load model (Compiling failed, so I compiled it manually)
#model = keras.models.load_model("../../models/model_variants/vgg16-pretrained-model-variants.h5", compile=False)
model = keras.models.load_model("../../models/car_types/best_model/vgg16-pretrained.h5", compile=False)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Set config
img_height = 300
img_width = 300
img_folder = '../test_pic'
# Set specific_model_variants to True if you want to test the model with specific Porsche model variants and years.
specific_model_variants = False
path_addon = "Porsche_more_classes" if specific_model_variants else "Porsche"
classes = MODEL_VARIANT if specific_model_variants else CAR_TYPE

config = {
    "path": f"C:/Users\phili/.keras/datasets/resized_DVM/{path_addon}",
    "batch_size": 32,
    "img_height": img_height,
    "img_width": img_width,
}

# Load background dataset
background_data = load_explainer_background(**config, shuffle=1000, take=1000)

# Load test images
images = []
img_names = []
for image in os.listdir(img_folder):
    img_names.append(image)
    img = tf.keras.utils.load_img(f"{img_folder}/{image}", target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create a batch
    images.append(img_array)

# Create explainer
explainer = shap.GradientExplainer(model, background_data)  # local_smoothing=0.1

# Explain and plot the results ( You can change CPU to GPU if you have a good enough GPU,
# on large models the GPU must have a big amount of memory)
with tf.device('/CPU:0'):
    for image, name in zip(images, img_names):
        shap_values, indexes = explainer.shap_values(image, ranked_outputs=3)
        # get the names for the classes
        print(indexes)
        image = image / 255
        index_names = np.vectorize(lambda x: classes[x])(indexes)
        shap.image_plot(shap_values, image, index_names, show=False)
        plt.suptitle("SHAP values for " + name)
        fig1 = plt.gcf()
        plt.show()
        # Remove file extension from image name
        name = name.split(".")[0]
        if specific_model_variants:
            fig1.savefig(f"results/model_variants/shap_values_{name}.png")
        else:
            fig1.savefig(f"results/car_types/shap_values_{name}.png")
