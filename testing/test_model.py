import numpy as np
import tensorflow as tf
from tensorflow import keras

img_height = 300
img_width = 300

model = tf.keras.models.load_model('../models/keras_model.h5')
img = tf.keras.utils.load_img(
    "../test_pic/panamera3.jpg", target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
class_names = ['718 Boxster', '718 Cayman', '911', '918', 'Boxster', 'Carrera Gt', 'Cayenne', 'Cayman', 'Macan', 'Panamera']
print(class_names)
print(predictions)
for pred in predictions:
    score = tf.nn.softmax(pred)
    print(class_names[np.argmax(score)], 100 * np.max(score))

score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
