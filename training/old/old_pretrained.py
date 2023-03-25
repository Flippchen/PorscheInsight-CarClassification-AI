# This file contains the code for training a model with data augmentation and a pretrained base.
# Import libraries
from keras.models import Sequential
from keras.applications import VGG16
from utilities.tools import *
from keras import layers
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set variables and config
AUTOTUNE = tf.data.AUTOTUNE
img_height = 300
img_width = 300
name = "vgg16-pretrained"
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


# Load the pre-trained VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_height,img_width,3))

# Set the trainable flag of the pre-trained model to False
for layer in vgg16.layers:
    layer.trainable = False

# Create a new head and initialize model
num_classes = len(class_names)
#model = Sequential([
#    data_augmentation,
#    vgg16,
#    layers.Flatten(),
#    layers.Dense(128, activation='relu'),
#    layers.Dense(num_classes, name="outputs")
#])
model = Sequential([
    data_augmentation,
    vgg16,
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dense(num_classes, activation='softmax', name="outputs")
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
# Plot and save model score
plot_model_score(history, epochs, name)

# Save model
model.save(f"../best_model/{name}.h5")