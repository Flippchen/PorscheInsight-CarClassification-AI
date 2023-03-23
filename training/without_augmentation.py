# This file contains the code for training a model without data augmentation.
# Import libraries
import numpy as np
from keras.models import Sequential
from tools import *

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
suppress_tf_warnings()

# Set variables and config
AUTOTUNE = tf.data.AUTOTUNE
img_height = 300
img_width = 300
name = "without-augmentation"
config = {
    "path": "C:/Users\phili/.keras/datasets/resized_DVM/Porsche",
    "batch_size": 32,
    "img_height": img_height,
    "img_width": img_width,
}

# Load dataset and classes
train_ds, val_ds, class_names = load_dataset(**config)
print(class_names)

# Show sample batch
show_sample_batch(train_ds, class_names)
show_batch_shape(train_ds)

# Shuffle/cache and set prefetch buffer size
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalize the data
normalization_layer = layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

# Create a new model and initialize model
num_classes = len(class_names)
model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Train model
epochs = 20
with tf.device('/GPU:1'):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot and save model score
plot_model_score(history, epochs, name)

# Save model
model.save(f"../models/cat_types/best_model/{name}.h5")
