# This file contains the code for training a model with data augmentation and a pretrained base.
# Import libraries
from keras.models import Sequential
from keras.applications import EfficientNetV2B1
from utilities.tools import *
from utilities.discord_callback import DiscordCallback
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')
suppress_tf_warnings()

# Set variables and config
AUTOTUNE = tf.data.AUTOTUNE
img_height = 300
img_width = 300
name = "vgg16-pretrained-deeper-model-variants"
# Variables to control training flow
# Set specific_model_variants to True if you want to test the model with specific Porsche model variants and years.
# Set specific_model_variants to False if you want to test the model with broad Porsche model types.
specific_model_variants = True
# Set to True to load trained model
load_model = False
load_path = "../models/model_variants/vgg16-pretrained-model-variants.h5"
# Config
path_addon = "Porsche_more_classes" if specific_model_variants else "Porsche"
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

# Load the pre-trained VGG16/EfficientNet model
# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
efficientnet = EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
# Set the trainable flag of the pre-trained model to False
# for layer in vgg16.layers:
# layer.trainable = False
for layer in efficientnet.layers:
    layer.trainable = False

# Fine-tuning: unfreeze some layers of pretrained model
# for layer in vgg16.layers[-4:]:
# layer.trainable = True
for layer in efficientnet.layers[-20:]:
    layer.trainable = True

# Create a new head and initialize model
num_classes = len(class_names)
# model = Sequential([
#    data_augmentation,
#    #vgg16,
#    layers.BatchNormalization(),
#    layers.LeakyReLU(),
#    layers.Flatten(),
#    layers.Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
#    layers.Dropout(0.5),
#    layers.BatchNormalization(),
#    layers.LeakyReLU(),
#    layers.Dense(num_classes, activation='softmax', name="outputs")
# ]) if not load_model else keras.models.load_model(load_path)
# Create a new head and initialize model
num_classes = len(class_names)
model = Sequential([
    data_augmentation,
    efficientnet,
    layers.GlobalAveragePooling2D(),
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
#lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', cooldown=0, min_lr=0)
lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto', restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath="../models/model_variants/best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
webhook_url = "YOUR WEBHOOK URL"
discord_callback = DiscordCallback(webhook_url)
# Train model
epochs = 15
with tf.device('/GPU:1'):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[lr,early_stopping, model_checkpoint, discord_callback]
    )
# Plot and save model score
plot_model_score(history, epochs, name, specific_model_variants)

# Save model
model.save(f"../models/model_variants/{name}.h5")

# TODO: Try efficientNEt instead of VGG16, different head
# TODO: Try effiecientNEt on old head
# TODO: Implement a DataGenerator
# TODO: Different data augmentation (vertical, ..), Augmentation before training

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
#train_datagen = ImageDataGenerator(
#    rescale=1./255,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    fill_mode='nearest')
#
#validation_datagen = ImageDataGenerator(rescale=1./255)
#
## Flow the data
#train_generator = train_datagen.flow_from_directory(train_dir,
#                                                    target_size=(img_height, img_width),
#                                                    batch_size=batch_size,
#                                                    class_mode='c