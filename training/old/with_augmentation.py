# This file contains the code for training a model with data augmentation.
# Import libraries
from keras.models import Sequential
from utilities.tools import *

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
suppress_tf_warnings()

# Set variables and config
AUTOTUNE = tf.data.AUTOTUNE
img_height = 300
img_width = 300
name = "with-augmentation"
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


# Create data augmentation layer and show augmented batch
data_augmentation = create_augmentation_layer(img_height, img_width)
show_augmented_batch(train_ds, data_augmentation)

# Create a new model and initialize model
num_classes = len(class_names)
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Train model
epochs = 20
device = tf.test.gpu_device_name() if tf.test.is_gpu_available() else '/CPU:0'
print("Using Device:", device)
with tf.device(device):
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

# Plot and save model score
plot_model_score(history, epochs, name)

# Save model
model.save(f"../models/car_types/best_model/{name}.h5")
