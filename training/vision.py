# This file contains the code for training a model with data augmentation and a pretrained base.
# Import libraries
from keras.models import Sequential
from vit_keras import vit, utils
from utilities.tools import *
from utilities.discord_callback import DiscordCallback
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')
suppress_tf_warnings()

# Set variables and config
AUTOTUNE = tf.data.AUTOTUNE
img_height = 224
img_width = 224
name = "vit-model-variants"
# Variables to control training flow
# Set model Type to 'all_specific_model_variants' or 'car_type or 'specific_model_variants'
model_type = 'specific_model_variants'
# Don't forget to change the save paths in the model checkpoint and model save
save_path = f"../models/model_variants/"
# Set to True to load trained model
load_model = False
load_path = "../models/all_model_variants/vit-model-variants.h5"
# Config
path_addon = get_data_path_addon(model_type)
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

# Create data augmentation layer and show augmented batch
data_augmentation = create_augmentation_layer(img_height, img_width)
show_augmented_batch(train_ds, data_augmentation)

# Create a new head and initialize model
num_classes = len(class_names)
# Load the pre-trained Vision Transformer model
vit = vit.vit_b16(
    image_size=img_height,
    activation='softmax',
    pretrained=True,  # Use pre-trained weights
    include_top=False,  # Exclude the classification head
    pretrained_top=False,
    classes=num_classes,
)
# Set the trainable flag of the pre-trained model to False
for layer in vit.layers:
    layer.trainable = False

# Fine-tuning: unfreeze some layers of pretrained model
# for layer in vit.layers[-20:]:
#    layer.trainable = True


model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),
    vit,
    layers.Flatten(),
    layers.Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax', name="outputs")
]) if not load_model else keras.models.load_model(load_path)

# Define optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Define learning rate scheduler
initial_learning_rate = 0.001
lr_decay_steps = 1000
lr_decay_rate = 0.96
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=lr_decay_steps,
    decay_rate=lr_decay_rate,
    staircase=True)

# Compile model
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Define callbacks
lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto', restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath=f"{save_path}{name}_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
# Discord callback ( If you want to use this, you need to set the environment variable "WEBHOOK_URL",
# otherwise comment it out and also remove the callback from the model callbacks)
webhook_url = os.environ.get('WEBHOOK_URL')
discord_callback = DiscordCallback(webhook_url)

# Train model
epochs = 20
with tf.device('/GPU:0'):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[lr, early_stopping, model_checkpoint, discord_callback]
    )
# Plot and save model score
plot_model_score(history, name, model_type)

# Save model
model.save(f"{save_path}{name}.h5")
