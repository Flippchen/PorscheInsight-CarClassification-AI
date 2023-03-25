from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from utilities.tools import *
from testing.class_names import MODEL_VARIANT, CAR_TYPE

suppress_tf_warnings()

# Load your saved Keras model and configure variables
specific_model_variants = False
saved_model_path = "../../models/model_variants/vgg16-pretrained-model-variants.h5" if specific_model_variants \
            else "../../models/car_types/best_model/vgg16-pretrained.h5"
path_addon = "Porsche_more_classes" if specific_model_variants else "Porsche"
img_height = 300
img_width = 300
config = {
    "path": f"C:/Users\phili/.keras/datasets/resized_DVM/{path_addon}",
    "batch_size": 32,
    "img_height": img_height,
    "img_width": img_width,
}

# Load model and (compile it)
model = load_model(saved_model_path, compile=False)
# Needs to be recompiled after loading because from_logits is set to True.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
# Load subset of data
data = load_image_subset(**config, shuffle=10000, number_images=1024)

# Extract images and labels from the dataset
all_images = []
all_labels = []
for images, labels in data:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())

all_images = np.vstack(all_images)
all_labels = np.hstack(all_labels)

# Evaluate the model on the test set ( I used CPU because my GPU is not powerful enough)
with tf.device('/CPU:0'):
    loss, accuracy = model.evaluate(all_images, all_labels, verbose=0)
    print("Test set accuracy: {:.2f}%".format(accuracy * 100))

    # Make predictions on the test set
    y_pred_probs = model.predict(all_images)

    # Convert the predicted probabilities into class labels
    y_pred = np.argmax(y_pred_probs, axis=1)

# Create the confusion matrix
cm = confusion_matrix(all_labels, y_pred)

# Print the confusion matrix to terminal
print("Confusion Matrix:")
print(cm)

# Get the names of the classes
class_names = MODEL_VARIANT if specific_model_variants else CAR_TYPE

# Plot the confusion matrix
plot_confusion_matrix(cm, class_names, specific_model_variants)
