# This file contains the code for training a model with data augmentation and a pretrained base.
# Import libraries
import numpy as np
from keras.models import Sequential
from keras.applications import VGG16
from tools import *
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')
suppress_tf_warnings()

# Set variables and config
AUTOTUNE = tf.data.AUTOTUNE
img_height = 300
img_width = 300
name = "vgg16-pretrained-more-classes"
# Variables to control training flow
# Set to True to use the more_classes dataset
more_classes = True
# Set to True to load trained model
load_model = True
load_path = "../models/more_classes/vgg16-pretrained-more-classes.h5"
# Config
path_addon = "Porsche_more_classes" if more_classes else "Porsche"
config = {
    "path": f"C:/Users\phili/.keras/datasets/resized_DVM/{path_addon}",
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
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Set the trainable flag of the pre-trained model to False
for layer in vgg16.layers:
    layer.trainable = False

# Fine-tuning: unfreeze some layers of VGG16
for layer in vgg16.layers[-4:]:
    layer.trainable = True

# Create a new head and initialize model
num_classes = len(class_names)
model = Sequential([
    data_augmentation,
    vgg16,
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Flatten(),
    layers.Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dense(num_classes, activation='softmax', name="outputs")
]) if not load_model else keras.models.load_model(load_path)

# Define optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# Compile model
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Define callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', cooldown=0, min_lr=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto', restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath="../models/more_classes/best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
# Train model
epochs = 1
with tf.device('/GPU:1'):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[lr_scheduler, early_stopping, model_checkpoint]
    )
# Plot and save model score
plot_model_score(history, epochs, name, more_classes)

# Save model
model.save(f"../models/more_classes/{name}.h5")

