import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from training.tools import *
from testing.class_names import MODEL_VARIANT, CAR_TYPE
suppress_tf_warnings()

# Load your saved Keras model
saved_model_path = "../../models/car_types/best_model/vgg16-pretrained.h5"
specific_model_variants = False
path_addon = "Porsche_more_classes" if specific_model_variants else "Porsche"
classes = MODEL_VARIANT if specific_model_variants else CAR_TYPE
img_height = 300
img_width = 300
config = {
    "path": f"C:/Users\phili/.keras/datasets/resized_DVM/{path_addon}",
    "batch_size": 32,
    "img_height": img_height,
    "img_width": img_width,
}

model = load_model(saved_model_path)
# Needs to be recompiled after loading because from_logits is set to True.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

train, val, class_names = load_dataset(**config)


# Extract images and labels from the dataset
all_images = []
all_labels = []
for images, labels in val:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())

all_images = np.vstack(all_images)
all_labels = np.hstack(all_labels)

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Evaluate the model on the test set
with tf.device('/CPU:0'):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test set accuracy: {:.2f}%".format(accuracy * 100))

    # Make predictions on the test set
    y_pred_probs = model.predict(X_test)

    # Convert the predicted probabilities into class labels
    y_pred = np.argmax(y_pred_probs, axis=1)

# Create the confusion matrix
cm = confusion_matrix(X_test, y_pred)

print("Confusion Matrix:")
print(cm)
